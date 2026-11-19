# VIGIL AI – Intelligent Video Violence Detection & Analysis System

> A complete, production-quality deep learning web application for detecting violence in video clips and analysing false positive predictions.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| Node.js | 18+ |
| pip | latest |

### 1. Install Dependencies

```bat
install.bat
```

Or manually:

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2. Configure Environment

```bash
cp .env.example backend/.env
# Edit backend/.env as needed
```

### 3. Start the Application

**Terminal 1 – Backend:**
```bat
start_backend.bat
# FastAPI runs on http://localhost:8000
# API docs: http://localhost:8000/docs
```

**Terminal 2 – Frontend:**
```bat
start_frontend.bat
# React runs on http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## 📁 Project Structure

```
vigil-ai/
├── backend/
│   ├── config.py               # Pydantic settings
│   ├── database.py             # SQLAlchemy async SQLite
│   ├── main.py                 # FastAPI entrypoint
│   ├── models/
│   │   ├── cnn_lstm.py         # CNN+LSTM architecture (MobileNetV2)
│   │   └── model_3dcnn.py      # Lightweight 3D-CNN architecture
│   ├── routers/
│   │   ├── predict.py          # POST /api/predict
│   │   ├── history.py          # GET/DELETE /api/history
│   │   ├── dashboard.py        # GET /api/dashboard/stats
│   │   ├── analysis.py         # GET /api/analysis/false-positives
│   │   ├── dataset.py          # GET/POST /api/dataset
│   │   └── training.py         # POST /api/train, GET /api/model/info
│   ├── schemas/
│   │   └── schemas.py          # Pydantic request/response models
│   └── services/
│       ├── inference.py        # Model inference + demo mode
│       ├── trainer.py          # Background training loop
│       ├── video_processor.py  # OpenCV frame extraction
│       ├── evaluator.py        # Scikit-learn metrics
│       └── false_positive.py   # FP analysis engine
├── frontend/
│   └── src/
│       ├── api/client.js       # Axios API client
│       ├── components/
│       │   ├── Sidebar.jsx     # Navigation
│       │   ├── StatCard.jsx    # Metric card
│       │   └── ConfusionMatrix.jsx
│       └── pages/
│           ├── Landing.jsx     # Hero landing page
│           ├── Dashboard.jsx   # Analytics dashboard
│           ├── VideoDetection.jsx   # Upload & classify
│           ├── FalsePositiveAnalysis.jsx
│           ├── DatasetExplorer.jsx
│           ├── ModelTraining.jsx
│           ├── PredictionHistory.jsx
│           └── About.jsx
├── datasets/
│   └── annotations/
│       └── category_annotations.csv   # Template
├── models/
│   └── weights/                # Trained model weights go here
├── requirements.txt
└── .env.example
```

---

## 🗂️ Datasets

### RWF-2000
Download from: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

Place in:
```
datasets/rwf2000/
  train/
    Fight/       *.avi
    NonFight/    *.avi
  val/
    Fight/
    NonFight/
```

### Hockey Fight
Download from the NTHU VIPLab page.

Place in:
```
datasets/hockey/
  Fight/      *.avi
  NonFight/   *.avi
```

### Demo Mode
If no datasets are present, the system automatically enters **Demo Mode** — predictions still work using an untrained model, clearly labelled as DEMO. No results are fabricated.

---

## 📊 Category Annotations (False Positive Analysis)

Create `datasets/annotations/category_annotations.csv`:

```csv
filename,activity_category,ground_truth,notes
video01.avi,Sports Activities,Non-Violent,Hockey tackle
video02.avi,Hugging and Friendly Physical Contact,Non-Violent,Airport reunion
video03.avi,Crowded Scenes,Non-Violent,Concert crowd
```

**Valid categories:**
- Sports Activities
- Hugging and Friendly Physical Contact
- Crowded Scenes
- Dancing and Group Movement
- Normal Physical Interactions
- Other Non-Violent Activities

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Upload and classify a video |
| POST | `/api/predict/annotate` | Add ground-truth label |
| GET | `/api/history` | Get prediction history |
| DELETE | `/api/history/{id}` | Delete a record |
| GET | `/api/history/export` | Export history as CSV |
| GET | `/api/dashboard/stats` | Dashboard statistics |
| GET | `/api/analysis/false-positives` | FP analysis report |
| GET | `/api/dataset/info` | Dataset availability |
| POST | `/api/dataset/load` | Scan/reload a dataset |
| POST | `/api/train` | Start training job |
| GET | `/api/train/status` | Poll training progress |
| GET | `/api/model/info` | Model metadata |
| GET | `/health` | Health check |

Full interactive docs: **http://localhost:8000/docs**

---

## 🧠 Model Architecture

### CNN + LSTM (Default)
- Backbone: MobileNetV2 (pretrained on ImageNet)
- 16 frames uniformly sampled per clip
- Bidirectional LSTM (2 layers, 512 hidden)
- Binary classification head with Sigmoid

### 3D-CNN (Alternative)
- 3 × Conv3D blocks with BatchNorm and MaxPool3D
- Global average pooling
- Input: `(B, 3, 16, 112, 112)`

---

## ⚠️ Important Notes

- **Demo mode results are not meaningful** — predictions from untrained models are random.
- **False positive category analysis requires annotation CSV** — raw predictions alone cannot determine activity type.
- **Training requires dataset files** — the backend will return an error if no dataset is found.
- Accuracy values are computed only from labelled test data; no values are fabricated.

---

## 📄 License

Academic / research use. See individual dataset licenses for data usage terms.
