# FinSight Build Context Summary

## Build Date: 2026-05-08
## Total Commits: 12
## Repository: https://github.com/kulharshit21/finsight-rag

---

## Project Structure

```
finsight/
├── backend/
│   ├── main.py                 # FastAPI entry point
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Backend container
│   ├── .env.example            # Environment template
│   ├── core/
│   │   ├── config.py           # Pydantic settings
│   │   └── __init__.py
│   ├── api/
│   │   ├── routes.py           # REST endpoints
│   │   └── __init__.py
│   ├── services/
│   │   ├── ingestion.py        # PDF → chunks
│   │   ├── vectorstore.py      # ChromaDB wrapper
│   │   ├── rag_chain.py        # LLM generation
│   │   ├── evaluator.py        # RAGAS metrics
│   │   ├── fraud_scorer.py     # XGBoost + rules
│   │   └── __init__.py
│   └── models/
│       ├── schemas.py          # Pydantic models
│       └── __init__.py
├── frontend/
│   ├── app.py                  # Streamlit UI
│   ├── requirements.txt        # Frontend deps
│   └── Dockerfile              # Frontend container
├── notebooks/
│   └── evaluate_rag.ipynb      # Evaluation notebook
├── docker-compose.yml          # Full stack orchestration
└── README.md                   # Project documentation
```

---

## Commit History

1. `4e26ad4` - feat: add backend core config module
2. `71b2ada` - feat: add Pydantic schemas for API models
3. `a6eaa06` - feat: add PDF ingestion service with financial chunking
4. `b701a48` - feat: add ChromaDB vector store service
5. `9e0a37b` - feat: add RAG chain with financial system prompt
6. `8b88e68` - feat: add RAGAS evaluator and XGBoost fraud scorer
7. `9f0d1ff` - feat: add FastAPI routes for upload, query, documents
8. `0c4e263` - feat: add FastAPI main entry, requirements, Dockerfile, env example
9. `b0f9ae0` - feat: add Streamlit frontend with chat, upload, library, dashboard
10. `770fc17` - feat: add docker-compose and project README
11. `4188775` - feat: add RAG evaluation notebook with metrics visualization

---

## Key Features Implemented

### Backend Services
- **Config Management**: Pydantic-settings with env file support
- **PDF Ingestion**: PyPDFLoader + RecursiveCharacterTextSplitter with financial separators
- **Vector Store**: ChromaDB with OpenAI embeddings, dedup by SHA-256
- **RAG Chain**: GPT-4o-mini with financial system prompt, source citations
- **Evaluation**: RAGAS faithfulness + answer_relevancy with MLflow logging
- **Fraud Scoring**: Rule-based patterns + XGBoost (optional) for risk scoring

### API Endpoints
- `POST /api/v1/upload` - Upload PDF/TXT documents
- `POST /api/v1/query` - RAG query with evaluation metrics
- `GET /api/v1/documents` - List ingested documents
- `DELETE /api/v1/documents/{doc_id}` - Remove document
- `GET /api/v1/health` - Health check

### Frontend Pages
- **Chat**: Conversational interface with metrics display
- **Upload**: Drag-and-drop file ingestion
- **Library**: Document management with delete
- **Dashboard**: MLflow integration info

---

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- LangChain + LangChain-OpenAI + LangChain-Chroma
- ChromaDB 0.5.0
- OpenAI API (GPT-4o-mini, text-embedding-3-small)
- RAGAS 0.1.9
- XGBoost 2.0.3
- MLflow 2.13.0
- Streamlit 1.35.0
- Docker + Docker Compose

---

## Next Steps for Graphify

To enable semantic graph extraction, set one of these API keys:
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` (Google Gemini)
- `MOONSHOT_API_KEY` (Kimi)
- `ANTHROPIC_API_KEY` (Claude)
- `OPENAI_API_KEY` (OpenAI)
- `OLLAMA_API_KEY` (local Ollama)

Then run: `graphify extract . --out graphify-context/`
