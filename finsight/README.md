# FinSight — Financial Document RAG Chatbot

A production-grade RAG (Retrieval-Augmented Generation) chatbot for financial document analysis. Upload annual reports, 10-Ks, earnings transcripts and ask natural-language questions. Features fraud detection, evaluation metrics, and MLflow tracking.

## Stack

- **Backend**: FastAPI · LangChain · ChromaDB · OpenAI · XGBoost · RAGAS · MLflow
- **Frontend**: Streamlit
- **Deployment**: Docker Compose

## Quick Start

```bash
# 1. Clone and setup
cd finsight/backend
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Install backend deps
pip install -r requirements.txt

# 3. Run backend
uvicorn main:app --reload --port 8000

# 4. Run frontend (new terminal)
cd ../frontend
pip install -r requirements.txt
streamlit run app.py

# 5. Open browser
# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
# MLflow:   http://localhost:5000
```

## Docker Compose

```bash
cd finsight
docker-compose up --build
```

## Features

- **PDF Ingestion**: SHA-256 deduplication, financial-tuned chunking
- **RAG Pipeline**: OpenAI embeddings + GPT-4o-mini generation
- **Evaluation**: RAGAS faithfulness & answer relevancy scores
- **Fraud Detection**: Rule-based + XGBoost scoring layer
- **MLflow Integration**: Track metrics over time

## Project Structure

```
finsight/
├── backend/          FastAPI + services
├── frontend/         Streamlit UI
├── notebooks/        Evaluation notebooks
└── docker-compose.yml
```

## Resume Bullet Points

- Built end-to-end RAG: PDF → chunking → embeddings → ChromaDB → GPT-4o-mini
- Integrated RAGAS evaluation framework with MLflow logging
- Extended fraud detection with rule-based red-flag detection
- Deployed via Docker Compose with full stack
