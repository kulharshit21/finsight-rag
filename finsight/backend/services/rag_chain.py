"""
RAG chain — retrieval → context formatting → Mistral LLM generation.

Uses mistral-large-latest for deterministic, grounded financial analysis.
Temperature is set to 0 — no creativity, pure grounding.
"""

import time
from typing import List, Optional, Tuple

from langchain_mistralai import ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

from core.config import settings
from services.vectorstore import VectorStoreService
from models.schemas import Source


SYSTEM_PROMPT = """You are FinSight, an expert financial analyst AI assistant.
You answer questions strictly based on the provided financial document excerpts.

Rules you MUST follow:
1. Only use information from the CONTEXT below. Never hallucinate figures.
2. If the context does not contain enough information, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
3. Always cite the page number and document name when referencing a specific figure or fact.
4. Format monetary values exactly as they appear in the document (₹, $, crore, million, etc.).
5. If you detect contradictory information across documents, flag it explicitly.
6. Be concise and structured — use bullet points for lists of figures.

CONTEXT:
{context}

QUESTION: {question}

Answer (with citations):"""


def _format_context(docs_scores: List[Tuple[Document, float]]) -> Tuple[str, List[Source]]:
    parts: List[str] = []
    sources: List[Source] = []

    for doc, score in docs_scores:
        meta = doc.metadata
        page = meta.get("page", 0) + 1
        filename = meta.get("filename", "unknown")
        excerpt = doc.page_content[:200].replace("\n", " ")

        parts.append(
            f"[Source: {filename}, Page {page}, Relevance: {score:.2f}]\n{doc.page_content}"
        )
        sources.append(Source(page=page, filename=filename, excerpt=excerpt))

    return "\n\n---\n\n".join(parts), sources


class RAGChain:
    def __init__(self, vectorstore: VectorStoreService):
        self.vectorstore = vectorstore
        self.llm = ChatMistralAI(
            model=settings.llm_model,
            mistral_api_key=settings.mistral_api_key,
            temperature=0,
            max_tokens=8192,   # mistral-large-latest supports up to 128k context
        )
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.parser = StrOutputParser()

    def query(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[Source], List[Tuple], int]:
        """Returns (answer, sources, raw_docs_scores, latency_ms)."""
        t0 = time.time()

        docs_scores = self.vectorstore.similarity_search(question, doc_ids=doc_ids)

        if not docs_scores:
            msg = "No relevant content found in the uploaded documents for this question."
            return msg, [], [], int((time.time() - t0) * 1000)

        context, sources = _format_context(docs_scores)

        chain = self.prompt | self.llm | self.parser
        answer = chain.invoke({"context": context, "question": question})

        latency_ms = int((time.time() - t0) * 1000)
        return answer, sources, docs_scores, latency_ms
