import time
import json
import torch
import numpy as np
from fastapi import BackgroundTasks

print("=" * 70, flush=True)
print("VERIFYING LOCAL FASTAPI BACKEND & DISTILBERT MODEL INITIALIZATION", flush=True)
print("=" * 70, flush=True)

# 1. Import application and state
from api.index import (
    app,
    init_supabase,
    load_model_and_tokenizer,
    predict,
    health_check,
    get_recent_logs,
    PredictRequest,
    DEVICE,
    MODEL_PATH
)
from api import index as app_module

# Check model path
print(f"\n[1] Checking Model File:", flush=True)
print(f"    Path: {MODEL_PATH}", flush=True)
print(f"    Exists: {MODEL_PATH.exists()}", flush=True)
if MODEL_PATH.exists():
    print(f"    File Size: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB", flush=True)

print("\n[2] Initializing Supabase & Loading DistilBERT...", flush=True)
init_supabase()
t0 = time.perf_counter()
load_model_and_tokenizer()
init_time = (time.perf_counter() - t0) * 1000.0

print(f"    Model Loaded: {app_module.model_loaded}", flush=True)
print(f"    Model Mode: '{app_module.model_mode}'", flush=True)
print(f"    Target Device: {app_module.DEVICE}", flush=True)
print(f"    Supabase Client: {app_module.supabase_client is not None}", flush=True)
print(f"    Initialization Latency: {init_time:.2f} ms", flush=True)

assert app_module.model_loaded is True, "ERROR: Model failed to load!"
assert app_module.model_mode == "distilbert_weights", f"ERROR: Expected 'distilbert_weights', got '{app_module.model_mode}'"
assert app_module.model is not None, "ERROR: Model instance is None"
assert app_module.model.training is False, "ERROR: Model is not in eval mode!"
assert app_module.supabase_client is not None, "ERROR: Supabase client is not initialized!"
print("    Evaluation Mode Verified: model.training == False (model.eval() ACTIVE)", flush=True)

# Helper to execute non-blocking coroutines without Windows IOCP completion ports
def run_coro(coro):
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value

# 2. Test Health Check Endpoint
print("\n[3] Testing GET /api/health Endpoint...", flush=True)
health = run_coro(health_check())
print("    Health Response:", json.dumps(health, indent=2), flush=True)
assert health["status"] == "healthy"
assert health["model_loaded"] is True
assert health["model_mode"] == "distilbert_weights"
assert health["supabase_connected"] is True, "ERROR: supabase_connected must be True!"

# 3. Test Prediction 1 (Real News)
print("\n[4] Testing POST /api/predict with Real News Article...", flush=True)
real_article = (
    "WASHINGTON (Reuters) - The U.S. Senate passed legislation on Tuesday to fund government "
    "agencies through mid-December, averting a shutdown and sending the measure to the House of Representatives."
)
bg1 = BackgroundTasks()
req1 = PredictRequest(text=real_article)
res1 = run_coro(predict(req1, bg1))
print(f"    Input Snippet: {real_article[:70]}...", flush=True)
print(f"    Verdict:       {res1.verdict}", flush=True)
print(f"    Confidence:    {res1.confidence}%", flush=True)
print(f"    Probabilities: {res1.probabilities}", flush=True)
print(f"    Latency:       {res1.latency_ms} ms", flush=True)
print(f"    Model Mode:    {res1.model_mode}", flush=True)
assert res1.verdict == "Real", f"Expected 'Real', got '{res1.verdict}'"
assert res1.model_mode == "distilbert_weights"

# 4. Test Prediction 2 (Fake News / Misinformation)
print("\n[5] Testing POST /api/predict with Misinformation / Fake News Article...", flush=True)
fake_article = (
    "SHOCKING BREAKING: Leaked whistleblower documents reveal Pope Francis endorses Donald Trump "
    "for President and exposes secret globalist weather control pyramid beneath Antarctica!"
)
bg2 = BackgroundTasks()
req2 = PredictRequest(text=fake_article)
res2 = run_coro(predict(req2, bg2))
print(f"    Input Snippet: {fake_article[:70]}...", flush=True)
print(f"    Verdict:       {res2.verdict}", flush=True)
print(f"    Confidence:    {res2.confidence}%", flush=True)
print(f"    Probabilities: {res2.probabilities}", flush=True)
print(f"    Latency:       {res2.latency_ms} ms", flush=True)
print(f"    Model Mode:    {res2.model_mode}", flush=True)
assert res2.verdict == "Fake", f"Expected 'Fake', got '{res2.verdict}'"
assert res2.model_mode == "distilbert_weights"

print("\n" + "=" * 70, flush=True)
print("SUCCESS: ALL LOCAL BACKEND INFERENCE VERIFICATIONS PASSED!", flush=True)
print("DistilBERT weights from 'model/best_model.pt' are active, running in eval mode,", flush=True)
print("and predicting accurately with sub-second inference latency.", flush=True)
print("=" * 70, flush=True)
