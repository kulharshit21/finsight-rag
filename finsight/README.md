<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=200&section=header&text=FinSight&fontSize=72&fontColor=e94560&animation=fadeIn&fontAlignY=38&desc=Financial%20Document%20RAG%20Chatbot&descAlignY=60&descColor=a8b2d8" width="100%"/>

<a href="https://readme-typing-svg.demolab.com">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=E94560&center=true&vCenter=true&width=700&lines=Upload+any+financial+document.;Ask+natural-language+questions.;Get+grounded%2C+cited+answers.;Zero+hallucinations.;Real-time+fraud+risk+detection." alt="Typing SVG" />
</a>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-Large-FF7000?style=for-the-badge&logo=mistral&logoColor=white)](https://mistral.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-8A2BE2?style=for-the-badge)](https://docs.ragas.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Fraud_Detection-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)

</div>

---

## What is FinSight?

**FinSight** is a production-grade RAG (Retrieval-Augmented Generation) system built specifically for financial documents — annual reports, 10-K filings, earnings call transcripts, and balance sheets.

Upload a PDF. Ask anything. Get a grounded, page-cited answer — never hallucinated. Every response is automatically evaluated for faithfulness and scanned for financial fraud signals.

---

## Features

```
📄  Upload PDFs / TXT         Drag-and-drop. Automatic SHA-256 dedup.
🔍  Semantic Retrieval         Mistral embeddings + ChromaDB vector search.
🤖  Grounded LLM Answers       mistral-large-latest with strict no-hallucination prompt.
📌  Page-level Citations        Every fact cited with filename + page number.
📊  RAGAS Evaluation           Faithfulness & answer relevancy scored per query.
📈  MLflow Tracking            All metrics logged for quality drift detection.
🚨  Fraud Detection            Regex red-flags + XGBoost risk score 0–1.
🐳  One-command Deploy         Full stack via Docker Compose.
📡  Swagger API Docs           Auto-generated at /docs.
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FinSight Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   User Browser                                                    │
│       │                                                           │
│       ▼                                                           │
│   ┌──────────────────┐                                           │
│   │  Streamlit UI     │  Chat · Upload · Library · Dashboard     │
│   └────────┬─────────┘                                           │
│            │  HTTP                                                │
│            ▼                                                      │
│   ┌──────────────────────────────────────────────────────┐      │
│   │              FastAPI Backend  /api/v1                  │      │
│   │                                                        │      │
│   │  POST /upload   →  ingestion.py  →  vectorstore.py    │      │
│   │                      SHA-256         Mistral Embed     │      │
│   │                      PyPDF           ChromaDB          │      │
│   │                      Chunking                          │      │
│   │                                                        │      │
│   │  POST /query    →  rag_chain.py                        │      │
│   │                      Similarity Search                 │      │
│   │                      mistral-large-latest              │      │
│   │                      Grounding Prompt                  │      │
│   │                         │                              │      │
│   │                         ├──→  evaluator.py (RAGAS)     │      │
│   │                         │       faithfulness           │      │
│   │                         │       answer_relevancy       │      │
│   │                         │       → MLflow               │      │
│   │                         │                              │      │
│   │                         └──→  fraud_scorer.py          │      │
│   │                                 regex red-flags        │      │
│   │                                 XGBoost risk 0–1       │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Fraud Red-Flag Detection

FinSight scans every answer and retrieved chunk for these signals:

| Flag | Pattern |
|---|---|
| 🔴 Related party transactions | `related.*party` |
| 🔴 Going concern doubt | `going.*concern` |
| 🟠 Financial restatement | `restatement\|restated` |
| 🟠 Qualified/adverse auditor opinion | `qualified.*opinion\|adverse.*opinion` |
| 🟠 Material weakness | `material.*weakness` |
| 🟡 Goodwill impairment | `goodwill.*impairment` |
| 🟡 Off-balance-sheet items | `off.*balance.*sheet` |
| 🟡 Revenue recognition change | `revenue.*recognition` |

Each triggered flag contributes to a **fraud risk score 0–1** shown inline with every answer.

---

## Project Structure

```
finsight/
├── backend/
│   ├── core/
│   │   └── config.py           ← All settings from environment variables
│   ├── models/
│   │   └── schemas.py          ← Pydantic request / response contracts
│   ├── services/
│   │   ├── ingestion.py        ← PDF → financial-tuned chunks + metadata
│   │   ├── vectorstore.py      ← ChromaDB CRUD + similarity search
│   │   ├── rag_chain.py        ← Retrieval → Mistral LLM → cited answer
│   │   ├── evaluator.py        ← RAGAS evaluation + MLflow logging
│   │   └── fraud_scorer.py     ← Regex red-flags + XGBoost classifier
│   ├── api/
│   │   └── routes.py           ← FastAPI endpoints (upload / query / docs)
│   ├── main.py                 ← App factory + CORS + lifespan
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app.py                  ← Streamlit UI (4 pages)
│   ├── requirements.txt
│   └── Dockerfile
├── notebooks/
│   └── evaluate_rag.ipynb      ← Batch evaluation + visualisation
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### Option 1 — Local (no Docker)

```bash
# 1. Clone
git clone https://github.com/kulharshit21/finsight-rag.git
cd finsight-rag/finsight/backend

# 2. Configure
cp .env.example .env
# Edit .env — add your MISTRAL_API_KEY

# 3. Install and run backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 4. In a new terminal — run frontend
cd ../frontend
pip install -r requirements.txt
streamlit run app.py
```

### Option 2 — Docker Compose (recommended)

```bash
git clone https://github.com/kulharshit21/finsight-rag.git
cd finsight-rag/finsight

cp backend/.env.example backend/.env
# Edit backend/.env — add your MISTRAL_API_KEY

docker-compose up --build
```

| Service | URL |
|---|---|
| 🖥️  Streamlit UI | http://localhost:8501 |
| 📡  Swagger API docs | http://localhost:8000/docs |
| 📈  MLflow tracking | http://localhost:5000 |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | — | **Required** — get one at [console.mistral.ai](https://console.mistral.ai) |
| `LLM_MODEL` | `mistral-large-latest` | Chat/generation model |
| `EMBEDDING_MODEL` | `mistral-embed` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `MIN_RELEVANCE_SCORE` | `0.3` | Minimum similarity threshold (0–1) |
| `RUN_EVALUATION` | `true` | Enable RAGAS scoring per query |
| `FRAUD_MODEL_PATH` | _(empty)_ | Path to XGBoost `.json` — heuristic used if blank |
| `FRAUD_THRESHOLD` | `0.65` | Risk score above which to show alert |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/upload` | Upload PDF or TXT document |
| `POST` | `/api/v1/query` | Ask a question, get cited answer + scores |
| `GET` | `/api/v1/documents` | List all ingested documents |
| `DELETE` | `/api/v1/documents/{doc_id}` | Remove a document and its chunks |
| `GET` | `/api/v1/health` | Health check |

Full interactive docs available at **`/docs`** (Swagger UI).

---

## Resume Bullets

```
FinSight — Financial Document RAG Chatbot                          May 2026
Mistral AI · FastAPI · LangChain · ChromaDB · XGBoost · MLflow · Docker · RAGAS

• Built end-to-end RAG pipeline: PDF ingestion → recursive financial chunking →
  Mistral embeddings (mistral-embed) → ChromaDB → mistral-large-latest generation

• Engineered SHA-256 deduplication and financial-tuned RecursiveCharacterTextSplitter
  with custom separator ordering to preserve table rows and numbered lists

• Integrated RAGAS evaluation framework for per-query faithfulness and answer
  relevancy scoring; metrics streamed to MLflow for quality drift detection

• Built two-layer fraud detection: regex red-flag scanner + optional XGBoost
  classifier producing calibrated risk score 0–1 alongside every answer

• Deployed full stack via Docker Compose: FastAPI + Streamlit + MLflow;
  REST API fully documented via OpenAPI/Swagger at /docs
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=100&section=footer" width="100%"/>

<sub>Built by <strong>Harshit Kulkarni</strong> · May 2026</sub>

</div>
