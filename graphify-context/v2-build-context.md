# FinSight v2 — Graphify Build Context

Generated: 2026-05-08 | Repo: https://github.com/kulharshit21/finsight-rag

## Architecture Map

```
HTTP Request
    │
    ▼
FastAPI (main.py)
    │  prefix: /api/v1
    ▼
api/routes.py
    ├── POST /upload   → VectorStoreService.ingest()
    │                       └── ingestion.load_and_split()
    │                               └── PyPDFLoader → RecursiveCharacterTextSplitter
    │
    ├── POST /query    → RAGChain.query()
    │                       ├── VectorStoreService.similarity_search()
    │                       ├── ChatOpenAI (GPT-4o-mini)
    │                       ├── evaluate_response()  [RAGAS + MLflow]
    │                       └── FraudScorer.score()  [regex + XGBoost]
    │
    ├── GET /documents → VectorStoreService.list_documents()
    └── DELETE /documents/{id} → VectorStoreService.delete_document()

VectorStoreService
    ├── OpenAIEmbeddings (text-embedding-3-small)
    └── Chroma (persisted to ./chroma_db)
```

## File → Responsibility Map

| File | Single Responsibility |
|---|---|
| `core/config.py` | All env-var driven settings via pydantic-settings |
| `models/schemas.py` | Pydantic request/response contracts for all endpoints |
| `services/ingestion.py` | PDF bytes → LangChain Document chunks with metadata |
| `services/vectorstore.py` | ChromaDB CRUD + similarity search |
| `services/rag_chain.py` | Retrieval → prompt → LLM → answer + sources |
| `services/evaluator.py` | RAGAS faithfulness + relevancy + MLflow logging |
| `services/fraud_scorer.py` | Regex red-flag detection + XGBoost risk scoring |
| `api/routes.py` | FastAPI endpoint handlers — thin orchestration layer |
| `main.py` | App factory + CORS + lifespan (VectorStore init) |
| `frontend/app.py` | Streamlit UI — Chat, Upload, Library, Dashboard |

## Key Design Decisions

- **Dedup**: SHA-256 of raw PDF bytes → 16-char hex ID stored in metadata.
  Before any embedding, ChromaDB is queried for existing `doc_id`.
- **Chunking strategy**: Financial separators ordered coarse→fine to keep
  table rows and numbered lists together. Chunk size 1000 / overlap 200.
- **Hallucination guard**: System prompt explicitly forbids using information
  outside the CONTEXT block; instructs model to decline if context insufficient.
- **Graceful degradation**: RAGAS failures → (0.5, 0.5). XGBoost not found →
  heuristic (0.15 × flag_count). MLflow down → silently skip.
- **Shared state**: VectorStoreService instantiated once in FastAPI lifespan,
  injected via `request.app.state` — avoids multiple ChromaDB connections.

## Commit History (v2)

1. chore: wipe v1 files for fresh v2 rebuild
2. feat: settings via pydantic-settings with env file support
3. feat: pydantic request/response schemas for all API endpoints
4. feat: PDF ingestion with SHA-256 dedup and financial-tuned chunking
5. feat: ChromaDB wrapper with dedup, similarity search, doc management
6. feat: RAG chain with grounding-enforced system prompt and source citations
7. feat: RAGAS faithfulness + answer_relevancy eval with MLflow logging
8. feat: fraud scorer with regex red-flags and XGBoost classifier layer
9. feat: FastAPI routes - upload, query, document library, health
10. feat: FastAPI app with lifespan vectorstore init and CORS middleware
11. feat: backend requirements, Dockerfile and .env.example
12. feat: Streamlit UI with chat, upload, library and dashboard pages
13. feat: docker-compose for full stack and comprehensive README
14. feat: batch RAG evaluation notebook with metrics visualisation
