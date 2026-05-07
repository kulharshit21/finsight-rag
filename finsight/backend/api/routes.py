from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from core.config import settings
from models.schemas import (
    DocumentInfo,
    ListDocumentsResponse,
    QueryRequest,
    QueryResponse,
    Source,
    UploadResponse,
)
from services.evaluator import evaluate_response
from services.fraud_scorer import FraudScorer
from services.rag_chain import RAGChain
from services.vectorstore import VectorStoreService


router = APIRouter()

# Fraud scorer is stateful (holds model in memory), initialise once at import time
_fraud_scorer = FraudScorer()

ALLOWED_CONTENT_TYPES = {"application/pdf", "text/plain"}
MAX_FILE_MB = 50


def _get_vs(request: Request) -> VectorStoreService:
    """Dependency: pull the VectorStoreService from app.state (set in lifespan)."""
    return request.app.state.vectorstore


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    vs: VectorStoreService = Depends(_get_vs),
):
    """
    Upload a PDF or TXT financial document.
    Duplicate uploads (same file content) are detected via SHA-256 and skipped.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Use PDF or TXT.")

    data = await file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_MB} MB limit.")

    result = vs.ingest(data, file.filename)
    return UploadResponse(**result, ingested_at=datetime.utcnow())


# ── Query ──────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    req: QueryRequest,
    vs: VectorStoreService = Depends(_get_vs),
):
    """
    Ask a natural-language question about the uploaded documents.
    Returns a grounded answer with source citations, RAGAS evaluation scores,
    and a fraud risk assessment — all in one response.
    """
    chain = RAGChain(vectorstore=vs)
    answer, sources, docs_scores, latency_ms = chain.query(
        question=req.question,
        doc_ids=req.doc_ids,
    )

    # RAGAS evaluation (runs against OpenAI — skipped if no docs retrieved)
    faithfulness_score, answer_relevancy_score = 0.5, 0.5
    if settings.run_evaluation and docs_scores:
        faithfulness_score, answer_relevancy_score = evaluate_response(
            req.question, answer, docs_scores
        )

    # Fraud risk analysis
    fraud_risk_score, fraud_flags = _fraud_scorer.score(answer, docs_scores)

    return QueryResponse(
        answer=answer,
        sources=sources,
        faithfulness_score=round(faithfulness_score, 3),
        answer_relevancy_score=round(answer_relevancy_score, 3),
        fraud_risk_score=fraud_risk_score,
        fraud_flags=fraud_flags,
        latency_ms=latency_ms,
    )


# ── Document library ───────────────────────────────────────────────────────────

@router.get("/documents", response_model=ListDocumentsResponse)
async def list_documents(vs: VectorStoreService = Depends(_get_vs)):
    """Return all ingested documents with their chunk counts."""
    docs = vs.list_documents()
    return ListDocumentsResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, vs: VectorStoreService = Depends(_get_vs)):
    """Permanently remove a document and all its chunks from the vector store."""
    deleted = vs.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(404, f"Document '{doc_id}' not found.")
    return {"doc_id": doc_id, "chunks_deleted": deleted}


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}
