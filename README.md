# 🛡️ VeriNews AI — Fake News & Misinformation Detector

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E.svg?logo=supabase&logoColor=white)](https://supabase.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VeriNews AI** is a production-grade, full-stack misinformation detection system engineered with a custom **Bidirectional LSTM (BiLSTM)** deep learning model trained for **50 full epochs** on the ISOT Fake News Dataset. The application combines a high-performance **FastAPI** backend, asynchronous **Supabase** audit logging, and a modern, responsive **light-mode SaaS interface**.

---

## 📌 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Deep Learning Architecture (BiLSTM)](#-deep-learning-architecture-bilstm)
- [FastAPI Backend & API Endpoints](#-fastapi-backend--api-endpoints)
- [Supabase Asynchronous Audit Logging](#-supabase-asynchronous-audit-logging)
- [Modern SaaS Frontend UI](#-modern-saas-frontend-ui)
- [Local Setup & Installation](#-local-setup--installation)
- [Running Backend Verification Tests](#-running-backend-verification-tests)
- [Running the Local Development Server](#-running-the-local-development-server)
- [Deployment on Vercel](#-deployment-on-vercel)

---

## 🌐 Overview & Architecture

Modern digital platforms face unprecedented volumes of fabricated news and misleading editorial content. VeriNews AI provides real-time verification and credibility scoring for news articles, press releases, and editorial snippets.

```
┌────────────────────────────────────────────────────────┐
│               VeriNews AI Client (UI)                  │
│   Tailwind CSS • Lucide Icons • Live Telemetry Cards   │
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST /api/predict
                           ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI Backend Service                │
│             (Vercel Serverless / Uvicorn)              │
│                                                        │
│   ┌────────────────────────────────────────────────┐   │
│   │ SimpleTokenizer (vocab.json)                   │   │
│   │ Text cleaning, token indexing, padding/truncate │   │
│   └──────────────────────┬─────────────────────────┘   │
│                          ▼                             │
│   ┌────────────────────────────────────────────────┐   │
│   │ PyTorch FakeNewsBiLSTM (best_model.pt)         │   │
│   │ 50 Epochs Trained • Bidirectional LSTM • BCE   │   │
│   └──────────────────────┬─────────────────────────┘   │
│                          │                             │
│                          ├──────────────┐              │
│                          ▼              ▼              │
│                 Inference Response   BackgroundTasks   │
│                 (JSON Verdict/Conf)         │          │
└─────────────────────────────────────────────┼──────────┘
                                              ▼
                                 ┌────────────────────────┐
                                 │   Supabase Database    │
                                 │    (prediction_logs)   │
                                 └────────────────────────┘
```

---

## 📁 Project Directory Structure

```
fakenewsdetector/
│
├── api/
│   └── index.py             # Production FastAPI serverless backend handler
│
├── data/
│   ├── True.csv             # Verified credible news reference samples
│   └── Fake.csv             # Misinformation reference samples
│
├── model/
│   ├── best_model.pt        # Trained 50-epoch FakeNewsBiLSTM weights checkpoint (13.22 MB)
│   ├── vocab.json           # Vocabulary mapping dictionary for tokenization
│   └── README.md            # Checkpoint metadata & layer specifications
│
├── train.py                 # Self-contained PyTorch training pipeline (50 epochs)
├── test_backend.py          # Automated verification test suite
├── index.html               # Advanced light-mode SaaS frontend interface
├── requirements.txt         # Production Python dependencies
├── schema.sql               # Supabase PostgreSQL schema for prediction_logs
├── vercel.json              # Serverless deployment & routing configuration
├── .env.example             # Template for environment credentials
└── README.md                # Project documentation
```

---

## 🧠 Deep Learning Architecture (BiLSTM)

The model is built from scratch in PyTorch without depending on oversized black-box transformers that rapidly overfit past 3–5 epochs. Instead, `FakeNewsBiLSTM` utilizes bidirectional recurrent pathways trained for **50 full epochs** on the ISOT dataset.

### Layer Specifications

| Component | Layer / Operation | Dimensions / Details |
| :--- | :--- | :--- |
| **Input** | Cleaned Text Sequence | Padded/truncated to `max_len = 100` |
| **Embedding** | `nn.Embedding` | `vocab_size = 25,000`, `embed_dim = 128`, `padding_idx = 0` |
| **Recurrent** | Bidirectional `nn.LSTM` | `input_size = 128`, `hidden_size = 128`, `bidirectional = True` |
| **Hidden States**| Concatenation | $\vec{h}_n \oplus \overleftarrow{h}_n \rightarrow 256$ dimensions |
| **Regularization**| `nn.Dropout` | $p = 0.3$ |
| **Classification**| `nn.Linear` Head | $256 \rightarrow 1$ (Sigmoid Binary Cross-Entropy) |

### Custom Tokenization Pipeline (`SimpleTokenizer`)
1. **Normalization**: Lowercases input, removes URL strings (`https?://\S+`), strips HTML tags, and replaces non-alphanumeric punctuation with whitespace.
2. **Indexing**: Maps word tokens to integer indices using [`model/vocab.json`](model/vocab.json). Special tokens include `<PAD>` (`0`) and `<UNK>` (`1`).
3. **Padded Sequences**: Automatically pads shorter texts or truncates longer articles to exactly 100 tokens for uniform tensor evaluation.

---

## 📡 FastAPI Backend & API Endpoints

The backend in [`api/index.py`](api/index.py) provides high-throughput, low-latency inference with automated CPU/CUDA hardware detection.

### 1. Health & Diagnostics: `GET /api/health`
Returns live service diagnostics, model status, device target, and database connectivity.

**Sample Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-11-13T11:54:54.866118",
  "model_loaded": true,
  "model_mode": "bilstm_50_epochs",
  "architecture": "FakeNewsBiLSTM",
  "epochs_trained": 50,
  "best_val_acc": 100.0,
  "device": "cpu",
  "supabase_connected": true
}
```

### 2. Prediction Inference: `POST /api/predict`
Evaluates text credibility under `torch.no_grad()` and schedules an asynchronous audit log.

**Sample Request:**
```json
{
  "text": "WASHINGTON (Reuters) - The U.S. Senate passed legislation on Tuesday to fund government agencies through mid-December, averting a shutdown and sending the measure to the House of Representatives."
}
```

**Sample Response:**
```json
{
  "verdict": "Real",
  "confidence": 87.98,
  "probabilities": {
    "Real": 87.98,
    "Fake": 12.02
  },
  "latency_ms": 318.07,
  "model_mode": "bilstm_50_epochs",
  "architecture": "FakeNewsBiLSTM (50 Epochs Trained)",
  "logged_to_supabase": true,
  "timestamp": "2026-11-13T11:54:55.184188"
}
```

### 3. Audit History: `GET /api/logs?limit=10`
Queries recent evaluation entries directly from the Supabase `prediction_logs` table.

---

## 💾 Supabase Asynchronous Audit Logging

Every prediction is persisted asynchronously using FastAPI `BackgroundTasks` so that database latency never blocks the inference response.

### PostgreSQL Table Schema
Execute the following in your Supabase SQL Editor (provided in [`schema.sql`](schema.sql)):

```sql
CREATE TABLE IF NOT EXISTS public.prediction_logs (
    id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    news_text TEXT NOT NULL,
    prediction VARCHAR(50) NOT NULL,
    confidence NUMERIC(5, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Enable Row Level Security (RLS)
ALTER TABLE public.prediction_logs ENABLE ROW LEVEL SECURITY;

-- Allow public read/insert access for API service roles
CREATE POLICY "Allow public read access"
ON public.prediction_logs FOR SELECT USING (true);

CREATE POLICY "Allow service insert access"
ON public.prediction_logs FOR INSERT WITH CHECK (true);
```

---

## 🎨 Modern SaaS Frontend UI

The frontend interface ([`index.html`](index.html)) provides a light-mode aesthetic:
- **Tailwind CSS & Lucide Icons**: Responsive layout with high-contrast slate typography and indigo accenting.
- **Dynamic Diagnostics**: Live navbar pills reporting API status, BiLSTM 50-epoch weights, and Supabase connection.
- **Visual Probability Breakdown**: Emerald (Real) and Rose (Fake) animated percentage bars.
- **Real-Time Audit Trail**: Dynamic table displaying recent evaluations logged into Supabase.
- **Preset Benchmarks**: One-click sample headlines for instant testing.

---

## 🛠️ Local Setup & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/reethika1-ai/fake_news_detector.git
cd fake_news_detector
```

### Step 2: Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the project root:
```env
SUPABASE_URL="https://your-project-id.supabase.co"
SUPABASE_KEY="your-anon-or-service-role-key"
```

---

## 🧪 Running Backend Verification Tests

An automated test suite ([`test_backend.py`](test_backend.py)) verifies model checkpoint integrity, `model.eval()` mode, Supabase connectivity, and inference on both real and fake news articles.

Run the test suite:
```bash
python -u test_backend.py
```

**Expected Output:**
```
======================================================================
VERIFYING LOCAL FASTAPI BACKEND & BiLSTM (50 EPOCHS) MODEL
======================================================================
[Supabase] Connected successfully to https://your-project-id.supabase.co...
[Tokenizer] Successfully loaded vocabulary (619 tokens) from model/vocab.json
[Model Loader] Loading checkpoint from: model/best_model.pt on cpu
[Model Loader] Successfully loaded FakeNewsBiLSTM (50 Epochs Trained, 25000 vocab, 128 hidden)!

[1] Checking Model File:
    Path: model/best_model.pt
    Exists: True | File Size: 13.22 MB

[2] Initializing Supabase & Loading BiLSTM (50 Epochs)...
    Model Loaded: True
    Model Mode: 'bilstm_50_epochs'
    Target Device: cpu
    Supabase Client: True
    Evaluation Mode Verified: model.training == False (model.eval() ACTIVE)

[3] Testing GET /api/health Endpoint...
    Health Response: { "status": "healthy", "model_loaded": true, "model_mode": "bilstm_50_epochs" }

[4] Testing POST /api/predict with Real News Article...
    Verdict: Real | Confidence: 87.98% | Latency: 318.07 ms

[5] Testing POST /api/predict with Misinformation / Fake News Article...
    Verdict: Fake | Confidence: 89.68% | Latency: 84.02 ms

======================================================================
SUCCESS: ALL LOCAL BACKEND INFERENCE VERIFICATIONS PASSED!
======================================================================
```

---

## 🚀 Running the Local Development Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn api.index:app --reload --host 0.0.0.0 --port 8000
```

Once running:
- **Interactive UI**: Open `index.html` in your browser or visit `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **ReDoc API Documentation**: `http://localhost:8000/redoc`

---

## ☁️ Deployment on Vercel

The repository is pre-configured for Vercel serverless deployment via [`vercel.json`](vercel.json):

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    },
    {
      "src": "index.html",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api/index.py"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ]
}
```

Deploy using the Vercel CLI:
```bash
vercel --prod
```
*Remember to configure `SUPABASE_URL` and `SUPABASE_KEY` in your Vercel Project Settings under Environment Variables.*

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
