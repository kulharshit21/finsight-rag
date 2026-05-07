from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from datetime import datetime
from typing import List, Optional
import json

from models.schemas import (
    QueryRequest, QueryResponse, UploadResponse,
    ListDocumentsResponse, DocumentInfo, Source,
)
from services.vectorstore import VectorStoreService
from services.rag_chain import RAGChain
from services.evaluator import evaluate_response
from services.fraud_scorer import FraudScorer
from core.config import settings


router = APIRouter()
fraud_scorer = FraudScorer()

ALLOWED_TYPES = {"application/pdf", "text/plain"}
MAX_FILE_SIZE_MB = 50


def get_vectorstore(request: Request) -> VectorStoreService:
    return request.app.state.vectorstore


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    vs: VectorStoreService = Depends(get_vectorstore),
):
    """Upload and ingest a financial document (PDF or TXT)."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Only PDF and TXT supported. Got: {file.content_type}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    result = vs.ingest(file_bytes, file.filename)
    return UploadResponse(
        **result,
        ingested_at=datetime.utcnow(),
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    req: QueryRequest,
    vs: VectorStoreService = Depends(get_vectorstore),
):
    """Query uploaded financial documents using RAG."""
    chain = RAGChain(vectorstore=vs)
    answer, sources, docs_scores, latency_ms = chain.query(
        question=req.question,
        doc_ids=req.doc_ids,
    )

    # Evaluation
    faithfulness_score, answer_relevancy_score = 0.5, 0.5
    if settings.run_evaluation and docs_scores:
        faithfulness_score, answer_relevancy_score = evaluate_response(
            req.question, answer, docs_scores
        )

    # Fraud scoring
    fraud_risk_score, fraud_flags = fraud_scorer.score(answer, docs_scores)

    return QueryResponse(
        answer=answer,
        sources=sources,
        faithfulness_score=round(faithfulness_score, 3),
        answer_relevancy_score=round(answer_relevancy_score, 3),
        fraud_risk_score=fraud_risk_score,
        fraud_flags=fraud_flags,
        latency_ms=latency_ms,
    )


@router.get("/documents", response_model=ListDocumentsResponse)
async def list_documents(vs: VectorStoreService = Depends(get_vectorstore)):
    """List all ingested documents."""
    docs = vs.list_documents()
    return ListDocumentsResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    vs: VectorStoreService = Depends(get_vectorstore),
):
    """Remove a document and all its chunks from the vector store."""
    deleted = vs.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(404, f"Document {doc_id} not found.")
    return {"doc_id": doc_id, "chunks_deleted": deleted}


@router.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}
