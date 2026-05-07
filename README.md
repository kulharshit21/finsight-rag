<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=220&section=header&text=FinSight&fontSize=80&fontColor=e94560&animation=fadeIn&fontAlignY=38&desc=Financial%20Document%20RAG%20Chatbot&descAlignY=60&descColor=a8b2d8" width="100%"/>

<a href="https://readme-typing-svg.demolab.com">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=E94560&center=true&vCenter=true&width=700&lines=Upload+any+financial+document.;Ask+natural-language+questions.;Get+grounded%2C+cited+answers.;Zero+hallucinations.;Real-time+fraud+risk+detection." alt="Typing SVG" />
</a>

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-Large-FF7000?style=for-the-badge&logo=mistral&logoColor=white)](https://mistral.ai)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-8A2BE2?style=for-the-badge)](https://docs.ragas.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Fraud_Scoring-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)

<br/>

[![GitHub stars](https://img.shields.io/github/stars/kulharshit21/finsight-rag?style=social)](https://github.com/kulharshit21/finsight-rag/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/kulharshit21/finsight-rag?style=social)](https://github.com/kulharshit21/finsight-rag/network/members)

</div>

---

## What is FinSight?

**FinSight** is a production-grade RAG (Retrieval-Augmented Generation) system built specifically for financial documents — annual reports, 10-K filings, earnings call transcripts, and balance sheets.

Upload a PDF. Ask anything. Get a grounded, page-cited answer — never hallucinated. Every response is automatically scored for faithfulness and scanned for financial fraud signals in real time.

---

## Core Features

| Feature | Description |
|---|---|
| 📄 **Smart Ingestion** | SHA-256 dedup · page-aware PDF loading · financial-tuned chunking |
| 🔍 **Semantic Search** | Mistral embeddings (`mistral-embed`) + ChromaDB persistent vector store |
| 🤖 **Grounded Answers** | `mistral-large-latest` with strict no-hallucination system prompt |
| 📌 **Page Citations** | Every fact cited with filename + page number |
| 📊 **RAGAS Evaluation** | Faithfulness + answer relevancy scored per query, shown live |
| 📈 **MLflow Tracking** | All metrics logged — detect quality drift over time |
| 🚨 **Fraud Detection** | Regex red-flags + XGBoost classifier · risk score 0–1 per answer |
| 🐳 **One-command Deploy** | Full stack via Docker Compose |
| 📡 **Swagger Docs** | Auto-generated interactive API at `/docs` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FinSight — Data Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📄 PDF Upload                                               │
│      │                                                       │
│      ▼                                                       │
│  SHA-256 Dedup ──► PyPDFLoader ──► Chunker ──► Mistral Embed│
│                                                    │         │
│                                              ChromaDB ◄──────│
│                                                              │
│  ❓ User Question                                            │
│      │                                                       │
│      ▼                                                       │
│  ChromaDB Search ──► Top-K Chunks ──► Mistral Large ──► Answer
│                                              │               │
│                              ┌───────────────┤              │
│                              │               │              │
│                              ▼               ▼              │
│                         RAGAS Eval     Fraud Scorer         │
│                         faithfulness   regex + XGBoost      │
│                         relevancy      risk score 0–1       │
│                              │               │              │
│                              └──────┬────────┘             │
│                                     ▼                       │
│                                 MLflow Log                  │
│                                     +                       │
│                               UI Response                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Local (no Docker)

```bash
git clone https://github.com/kulharshit21/finsight-rag.git
cd finsight-rag/finsight/backend

cp .env.example .env
# Add your MISTRAL_API_KEY to .env

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

In a second terminal:

```bash
cd finsight-rag/finsight/frontend
pip install -r requirements.txt
streamlit run app.py
```

### Docker Compose

```bash
cd finsight-rag/finsight
cp backend/.env.example backend/.env
# Add your MISTRAL_API_KEY

docker-compose up --build
```

| URL | Service |
|---|---|
| http://localhost:8501 | Streamlit UI |
| http://localhost:8000/docs | Swagger API |
| http://localhost:5000 | MLflow tracking |

---

## Fraud Red-Flag Scanner

Every response is scanned for these financial risk signals:

```
🔴  Related party transactions     going concern doubts
🟠  Financial restatements         auditor qualified opinions
🟠  Material weaknesses            goodwill impairments
🟡  Off-balance-sheet items        revenue recognition changes
```

Risk signals aggregate into a calibrated **fraud score 0–1** displayed next to every answer.

---

## Project Structure

```
finsight/
├── backend/
│   ├── core/config.py          ← env-var driven settings
│   ├── models/schemas.py       ← Pydantic contracts
│   ├── services/
│   │   ├── ingestion.py        ← PDF → chunks
│   │   ├── vectorstore.py      ← ChromaDB wrapper
│   │   ├── rag_chain.py        ← retrieval + Mistral generation
│   │   ├── evaluator.py        ← RAGAS + MLflow
│   │   └── fraud_scorer.py     ← red-flags + XGBoost
│   ├── api/routes.py           ← FastAPI endpoints
│   └── main.py
├── frontend/app.py             ← Streamlit (Chat/Upload/Library/Dashboard)
├── notebooks/evaluate_rag.ipynb
└── docker-compose.yml
```

---

## Resume

```
FinSight — Financial Document RAG Chatbot                    May 2026
Mistral AI · FastAPI · LangChain · ChromaDB · XGBoost · MLflow · Docker · RAGAS

• End-to-end RAG: PDF → financial chunking → mistral-embed → ChromaDB → mistral-large-latest
• SHA-256 dedup, financial-tuned splitter (preserves table rows and numbered lists)
• RAGAS per-query evaluation logged to MLflow for quality drift detection
• Two-layer fraud scoring: regex red-flags + calibrated XGBoost risk score 0–1
• Deployed via Docker Compose; REST API documented via OpenAPI/Swagger
```

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,50:16213e,100:0f3460&height=120&section=footer" width="100%"/>

**Built by Harshit Kulkarni · May 2026**

</div>
