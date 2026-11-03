import os
import re
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Prevent Hugging Face from hanging in offline / firewalled environments
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Explicitly load .env from project root with override
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fake_news_detector")

DEFAULT_MODEL_PATH = BASE_DIR / "model" / "best_model.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# Classification Labels (Default: 0 -> Fake, 1 -> Real)
LABEL_0 = os.getenv("LABEL_0", "Fake")
LABEL_1 = os.getenv("LABEL_1", "Real")
LABEL_MAP = {0: LABEL_0, 1: LABEL_1}
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
).strip()

# Ensure os.environ reflects SUPABASE_KEY if defined via SUPABASE_ANON_KEY
if SUPABASE_KEY and not os.getenv("SUPABASE_KEY"):
    os.environ["SUPABASE_KEY"] = SUPABASE_KEY

# Lazy imports / conditional libraries
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"
    logger.warning("PyTorch or NumPy not installed in current environment.")

try:
    from transformers import (
        AutoTokenizer,
        DistilBertTokenizerFast,
        DistilBertConfig,
        DistilBertForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Hugging Face Transformers not installed in current environment.")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None
    logger.warning("supabase-py library not installed.")

# Global state holders
model = None
tokenizer = None
model_loaded = False
model_mode = "uninitialized"
supabase_client: Optional[Any] = None


# -----------------------------------------------------------------------------
# Supabase Initialization Helper
# -----------------------------------------------------------------------------
def init_supabase() -> Optional[Any]:
    global supabase_client, SUPABASE_URL, SUPABASE_KEY
    if not SUPABASE_AVAILABLE:
        logger.info("Supabase client skipped: supabase-py package not found.")
        return None
    url = SUPABASE_URL or os.getenv("SUPABASE_URL", "").strip()
    key = (
        SUPABASE_KEY
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY", "")
    ).strip()
    if not url or not key:
        logger.info("Supabase credentials not configured in environment (SUPABASE_URL / SUPABASE_KEY).")
        return None
    try:
        supabase_client = create_client(url, key)
        logger.info(f"Supabase client initialized successfully for: {url}")
        return supabase_client
    except Exception as exc:
        logger.error(f"Failed to initialize Supabase client: {exc}")
        return None


def log_prediction_to_supabase(news_text: str, prediction: str, confidence: float):
    """
    Synchronous helper designed to run inside FastAPI BackgroundTasks
    or worker thread to record inference evaluations to Supabase.
    """
    global supabase_client
    if not supabase_client:
        return

    try:
        record = {
            "news_text": news_text,
            "prediction": prediction,
            "confidence": round(float(confidence), 2),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        response = supabase_client.table("prediction_logs").insert(record).execute()
        logger.info(f"Logged prediction to Supabase 'prediction_logs' successfully. (Verdict: {prediction})")
    except Exception as exc:
        logger.error(f"Error recording log to Supabase: {exc}")


# -----------------------------------------------------------------------------
# Model & Tokenizer Loader
# -----------------------------------------------------------------------------
def load_model_and_tokenizer():
    """
    Loads DistilBERT model weights and tokenizer with GPU/CPU compatibility fallback.
    If best_model.pt does not exist yet, initializes fallback/simulation mode immediately
    without blocking cold starts or network connections.
    """
    global model, tokenizer, model_loaded, model_mode

    logger.info(f"Target execution device: {DEVICE}")
    logger.info(f"Checking model path: {MODEL_PATH}")

    # If model file does not exist yet, initialize simulation placeholder mode immediately
    if not (TORCH_AVAILABLE and MODEL_PATH.exists()):
        logger.warning(
            f"Notice: '{MODEL_PATH}' was not found. "
            "Server initialized in resilient SIMULATION / PLACEHOLDER mode. "
            "Place your trained 'best_model.pt' into the model/ folder to enable live PyTorch inference."
        )
        model_loaded = False
        model_mode = "simulation_placeholder"
        return

    # Model file exists! Proceed with loading tokenizer and weights
    if TRANSFORMERS_AVAILABLE:
        try:
            logger.info("Loading DistilBERT tokenizer ('distilbert-base-uncased')...")
            # Try locally cached tokenizer first
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            logger.info("DistilBERT tokenizer loaded successfully.")
        except Exception as exc:
            logger.warning(f"Could not load Hugging Face tokenizer: {exc}. Using fallback tokenization.")
            tokenizer = None

    try:
        logger.info(f"Loading custom weights from {MODEL_PATH} on {DEVICE}...")
        checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE)

        if TRANSFORMERS_AVAILABLE:
            if isinstance(checkpoint, dict):
                config = DistilBertConfig(num_labels=2)
                base_model = DistilBertForSequenceClassification(config)
                if "state_dict" in checkpoint:
                    base_model.load_state_dict(checkpoint["state_dict"], strict=False)
                elif "model_state_dict" in checkpoint:
                    base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    base_model.load_state_dict(checkpoint, strict=False)
                model = base_model
            elif isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                config = DistilBertConfig(num_labels=2)
                base_model = DistilBertForSequenceClassification(config)
                model = base_model
        else:
            model = checkpoint

        model.to(DEVICE)
        model.eval()
        model_loaded = True
        model_mode = "distilbert_weights"
        logger.info("Custom pre-trained DistilBERT model successfully loaded and set to eval mode.")
    except Exception as exc:
        logger.error(f"Failed to load weights from {MODEL_PATH}: {exc}")
        model_loaded = False
        model_mode = "simulation_placeholder"


# -----------------------------------------------------------------------------
# Heuristic Fallback Engine (Runs when best_model.pt is pending placement)
# -----------------------------------------------------------------------------
def simulate_inference(text: str) -> Dict[str, Any]:
    """
    Computes an intelligent linguistic heuristic score when custom weights
    have not yet been placed in model/best_model.pt.
    Analyzes sensationalism, capitalization, clickbait keywords, and punctuation.
    """
    clean_text = text.strip()
    words = clean_text.split()
    word_count = len(words)

    # Clickbait / Fake news indicators
    clickbait_patterns = [
        r"\bshocking\b", r"\byou won't believe\b", r"\bsecret cure\b",
        r"\bmiracle\b", r"\bthey don't want you to know\b", r"\b100% guaranteed\b",
        r"\bconspiracy\b", r"\balien\b", r"\billuminati\b", r"\bbombshell\b",
        r"\bmainstream media won't show\b", r"\bproof that\b", r"\burip\b",
        r"\bdeep state\b", r"\bunbelievable\b", r"\bgone wrong\b"
    ]
    
    sensational_hits = sum(1 for pattern in clickbait_patterns if re.search(pattern, clean_text, re.IGNORECASE))
    
    # Capitalization & punctuation metrics
    all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
    exclamation_count = clean_text.count("!")
    question_count = clean_text.count("?")

    suspicion_score = 0.0
    suspicion_score += min(sensational_hits * 0.25, 0.50)
    suspicion_score += min((all_caps_words / max(word_count, 1)) * 1.5, 0.30)
    suspicion_score += min(exclamation_count * 0.08, 0.20)

    # Determine simulated probabilities
    if suspicion_score > 0.35:
        verdict = "Fake"
        fake_prob = min(0.65 + suspicion_score * 0.35, 0.985)
        real_prob = 1.0 - fake_prob
        confidence = fake_prob * 100.0
    else:
        verdict = "Real"
        real_prob = min(0.70 + (1.0 - suspicion_score) * 0.28, 0.978)
        fake_prob = 1.0 - real_prob
        confidence = real_prob * 100.0

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "probabilities": {
            "Real": round(real_prob * 100.0, 2),
            "Fake": round(fake_prob * 100.0, 2)
        },
        "model_mode": "simulation_placeholder"
    }


# -----------------------------------------------------------------------------
# Lifespan Context Manager
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Fake News Detector backend...")
    init_supabase()
    load_model_and_tokenizer()
    yield
    logger.info("Shutting down Fake News Detector backend...")


# -----------------------------------------------------------------------------
# FastAPI Application Declaration
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Fake News & Misinformation Detector API",
    description="High-throughput Serverless DistilBERT inference API with Supabase persistence for misinformation analysis.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=100000,
        description="The news article text, headline, or excerpt to evaluate.",
        example="A breakthrough study published today in Nature Medicine indicates promising clinical trial results for a novel cancer therapy."
    )

class PredictResponse(BaseModel):
    verdict: str = Field(..., description="Classification category: 'Real' or 'Fake'")
    confidence: float = Field(..., description="Confidence score percentage (0.0 to 100.0)")
    probabilities: Dict[str, float] = Field(..., description="Probabilities breakdown for Real vs Fake")
    latency_ms: float = Field(..., description="Inference execution time in milliseconds")
    model_mode: str = Field(..., description="Model state: distilbert_weights or simulation_placeholder")
    logged_to_supabase: bool = Field(..., description="Whether entry was queued for Supabase recording")
    created_at: str = Field(..., description="UTC ISO timestamp of evaluation")


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the front-end dashboard if running locally or directly via Uvicorn."""
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "name": "Fake News Detector API",
        "status": "online",
        "docs_url": "/docs"
    })


@app.get("/api/health")
async def health_check():
    """Provides system diagnostics, model status, and database connectivity."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available() if TORCH_AVAILABLE else False,
        "model_loaded": model_loaded,
        "model_mode": model_mode,
        "model_file_found": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "supabase_connected": supabase_client is not None
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Evaluates news article text for misinformation.
    Computes DistilBERT sequence classification probabilities (or fallback simulation),
    and asynchronously writes the record to Supabase's prediction_logs table.
    """
    start_time = time.perf_counter()
    raw_text = request.text.strip()

    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provided news text cannot be blank or whitespace."
        )

    # 1. DistilBERT Inference Execution (if model and tokenizer are active)
    if model_loaded and model is not None and tokenizer is not None and TORCH_AVAILABLE:
        try:
            inputs = tokenizer(
                raw_text,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            verdict = LABEL_MAP.get(pred_idx, "Unknown")
            confidence = round(float(probs[pred_idx]) * 100.0, 2)

            # Map probabilities for label output
            prob_dict = {
                LABEL_MAP.get(i, f"Class_{i}"): round(float(probs[i]) * 100.0, 2)
                for i in range(len(probs))
            }
            active_mode = "distilbert_weights"

        except Exception as exc:
            logger.error(f"Inference computation error: {exc}. Falling back to simulation.")
            sim = simulate_inference(raw_text)
            verdict = sim["verdict"]
            confidence = sim["confidence"]
            prob_dict = sim["probabilities"]
            active_mode = "simulation_placeholder"
    else:
        # Fallback simulation when best_model.pt has not been uploaded yet
        sim = simulate_inference(raw_text)
        verdict = sim["verdict"]
        confidence = sim["confidence"]
        prob_dict = sim["probabilities"]
        active_mode = "simulation_placeholder"

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    timestamp_str = datetime.now(timezone.utc).isoformat()

    # 2. Asynchronously persist evaluation entry to Supabase
    logged = False
    if supabase_client is not None:
        background_tasks.add_task(log_prediction_to_supabase, raw_text, verdict, confidence)
        logged = True

    return PredictResponse(
        verdict=verdict,
        confidence=confidence,
        probabilities=prob_dict,
        latency_ms=latency_ms,
        model_mode=active_mode,
        logged_to_supabase=logged,
        created_at=timestamp_str
    )


@app.get("/api/logs")
async def get_recent_logs(limit: int = 10):
    """
    Fetches recent prediction log entries from the Supabase prediction_logs table.
    Enables live dashboard inspection of previous test verifications.
    """
    if not supabase_client:
        return {
            "status": "unconfigured",
            "message": "Supabase client is not connected or credentials are missing.",
            "logs": []
        }

    try:
        query_limit = min(max(limit, 1), 50)
        res = (
            supabase_client.table("prediction_logs")
            .select("id, news_text, prediction, confidence, created_at")
            .order("created_at", desc=True)
            .limit(query_limit)
            .execute()
        )
        return {
            "status": "success",
            "count": len(res.data),
            "logs": res.data
        }
    except Exception as exc:
        logger.error(f"Failed to fetch logs from Supabase: {exc}")
        return {
            "status": "error",
            "message": str(exc),
            "logs": []
        }
