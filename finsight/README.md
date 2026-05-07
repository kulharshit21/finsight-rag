# FinSight — Financial Document RAG Chatbot

Upload any company's annual report, 10-K, balance sheet, or earnings transcript and ask natural-language questions. Every answer is grounded in the document, cited by page, evaluated for quality using RAGAS, and scanned for financial fraud signals.

## Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| RAG pipeline | LangChain + ChromaDB + OpenAI |
| Evaluation | RAGAS + MLflow |
| Fraud detection | Rule-based regex + XGBoost |
| Frontend | Streamlit |
| Deployment | Docker Compose |

## Quick Start (local)

```bash
cd finsight/backend
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

In a second terminal:
```bash
cd finsight/frontend
pip install -r requirements.txt
streamlit run app.py
```

| URL | Service |
|---|---|
| http://localhost:8501 | Streamlit UI |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:5000 | MLflow tracking |

## Docker (one command)

```bash
cd finsight
cp backend/.env.example backend/.env
# Edit backend/.env — add OPENAI_API_KEY

docker-compose up --build
```

## How It Works

1. **Upload** — PDF bytes → SHA-256 dedup → PyPDFLoader → RecursiveCharacterTextSplitter (financial-tuned) → OpenAI embeddings → ChromaDB
2. **Query** — question → ChromaDB similarity search → top-k chunks → GPT-4o-mini with strict grounding prompt → cited answer
3. **Evaluate** — RAGAS measures faithfulness and answer relevancy per query; metrics logged to MLflow
4. **Fraud scan** — regex patterns + optional XGBoost classifier on retrieved text; risk score 0–1 shown alongside every answer

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `MIN_RELEVANCE_SCORE` | `0.3` | Minimum similarity threshold |
| `RUN_EVALUATION` | `true` | Enable RAGAS per query |
| `FRAUD_MODEL_PATH` | _(empty)_ | Path to XGBoost model; heuristic used if blank |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |

## Project Structure

```
finsight/
├── backend/
│   ├── core/config.py          # All settings via environment variables
│   ├── models/schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── ingestion.py        # PDF → chunks with metadata
│   │   ├── vectorstore.py      # ChromaDB wrapper
│   │   ├── rag_chain.py        # Retrieval + LLM generation
│   │   ├── evaluator.py        # RAGAS evaluation
│   │   └── fraud_scorer.py     # Red-flag detection
│   ├── api/routes.py           # FastAPI endpoints
│   └── main.py                 # App entry point
├── frontend/app.py             # Streamlit UI
├── notebooks/evaluate_rag.ipynb
└── docker-compose.yml
```
