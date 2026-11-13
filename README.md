# 🛡️ VeriNews AI — Fake News & Misinformation Detector

A production-ready, serverless fake news and misinformation detection platform engineered with **FastAPI**, a custom scratch-built **Bidirectional LSTM (BiLSTM)** trained for **50 Epochs** on the ISOT Fake News Dataset, **Supabase** asynchronous persistence, **Vercel**, and an advanced **light-mode SaaS UI**.

---

## 📁 Project Architecture

```
fake-news-detector/
│
├── api/
│   └── index.py             # FastAPI serverless backend handler for Vercel & local dev
│
├── data/
│   ├── True.csv             # ISOT dataset: Real / Credible news articles
│   └── Fake.csv             # ISOT dataset: Fake / Misinformation articles
│
├── model/
│   ├── best_model.pt        # Trained 50-epoch FakeNewsBiLSTM checkpoint
│   ├── vocab.json           # Vocabulary mapping dictionary
│   └── README.md            # Checkpoint specification & retraining guide
│
├── train.py                 # Self-contained 50-epoch PyTorch training pipeline
├── test_backend.py          # Automated verification test suite
├── .env                     # Supabase & app credentials
├── index.html               # Advanced light-mode SaaS frontend dashboard
├── requirements.txt         # Production Python dependencies
├── schema.sql               # Supabase SQL schema for prediction_logs table
├── vercel.json              # Vercel serverless function & routing configuration
└── README.md                # Project documentation
```

---

## 🚀 Key Features

- **⚡ Custom Bidirectional LSTM (BiLSTM) Engine**: Scratch-built PyTorch `FakeNewsBiLSTM` featuring an embedding layer, 2-layer Bidirectional LSTM, Dropout regularization, and a dense classification head, trained for **50 full epochs**.
- **🧠 Custom Word-Level Tokenizer**: Word tokenizer that extracts and cleans tokens, builds vocabulary maps, and pads/truncates sequences up to `max_len=100`.
- **🛡️ Resilient Placeholder / Heuristic Mode**: If `model/best_model.pt` is ever missing, the backend boots smoothly in simulation mode, allowing end-to-end testing without crashes.
- **💾 Asynchronous Supabase Persistence**: Every evaluated article is asynchronously written into the Supabase `prediction_logs` table via background tasks, recording `news_text`, `prediction`, `confidence`, and `created_at`.
- **🎨 Modern SaaS Dashboard (`index.html`)**: Clean light-mode aesthetic using Tailwind CSS and Lucide icons, responsive text input, benchmark sample presets, live telemetry, and real-time audit feed.
- **☁️ Zero-Config Vercel Deployment**: Pre-configured `vercel.json` routing both serverless Python endpoints (`/api/*`) and static dashboard assets.

---

## 🛠️ Quickstart & Local Setup

### 1. Set Up Python Environment
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model for 50 Epochs
To train or retrain the custom BiLSTM for 50 epochs:
```bash
python -u train.py --epochs 50 --batch-size 64 --lr 0.001
```
This trains the model, tracks validation accuracy, and exports `model/best_model.pt` and `model/vocab.json`.

### 4. Run Backend Tests
Run the verification test script:
```bash
python -u test_backend.py
```

### 5. Start Local Development Server
```bash
uvicorn api.index:app --reload --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` or open `index.html` in your browser.

---

## 📡 API Endpoints

### 1. `GET /api/health`
Returns service diagnostic telemetry and model loading status.
```json
{
  "status": "healthy",
  "timestamp": "2026-11-13T11:28:00.645378",
  "model_loaded": true,
  "model_mode": "bilstm_50_epochs",
  "architecture": "FakeNewsBiLSTM",
  "epochs_trained": 50,
  "best_val_acc": 100.0,
  "device": "cpu",
  "supabase_connected": true
}
```

### 2. `POST /api/predict`
Evaluates text and classifies as Real or Fake.
```json
// Request:
{
  "text": "WASHINGTON (Reuters) - The U.S. Senate passed legislation to fund government..."
}

// Response:
{
  "verdict": "Real",
  "confidence": 99.99,
  "probabilities": {
    "Real": 99.99,
    "Fake": 0.01
  },
  "latency_ms": 22.62,
  "model_mode": "bilstm_50_epochs",
  "architecture": "FakeNewsBiLSTM (50 Epochs, Acc: 100.0%)",
  "logged_to_supabase": true,
  "timestamp": "2026-11-13T11:28:00.675000"
}
```

### 3. `GET /api/logs?limit=10`
Retrieves recent evaluations directly from the Supabase `prediction_logs` table.
