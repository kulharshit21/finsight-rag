"""
ChromaDB vector store wrapper using Mistral embeddings (mistral-embed).
"""

from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

from core.config import settings
from services.ingestion import load_and_split


class VectorStoreService:
    def __init__(self):
        self.embeddings = MistralAIEmbeddings(
            model=settings.embedding_model,
            mistral_api_key=settings.mistral_api_key,
        )
        self.db = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=settings.chroma_persist_dir,
        )

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest(self, file_bytes: bytes, filename: str) -> dict:
        chunks, doc_id = load_and_split(file_bytes, filename)

        existing = self.db.get(where={"doc_id": doc_id})
        if existing["ids"]:
            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunks_created": len(existing["ids"]),
                "status": "already_exists",
            }

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "status": "ingested",
        }

    # ── Retrieval ──────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        k = k or settings.top_k_retrieval
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None

        results = self.db.similarity_search_with_relevance_scores(
            query, k=k, filter=where
        )
        return [(doc, score) for doc, score in results
                if score >= settings.min_relevance_score]

    # ── Document management ────────────────────────────────────────────────

    def list_documents(self) -> List[dict]:
        all_meta = self.db.get()["metadatas"]
        seen: dict = {}
        for m in all_meta:
            if not m:
                continue
            did = m.get("doc_id")
            if did not in seen:
                seen[did] = {
                    "doc_id": did,
                    "filename": m.get("filename", "unknown"),
                    "chunk_count": 0,
                    "ingested_at": m.get("ingested_at", ""),
                }
            seen[did]["chunk_count"] += 1
        return list(seen.values())

    def delete_document(self, doc_id: str) -> int:
        existing = self.db.get(where={"doc_id": doc_id})
        if not existing["ids"]:
            return 0
        self.db.delete(ids=existing["ids"])
        return len(existing["ids"])
