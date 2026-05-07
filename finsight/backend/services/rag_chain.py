"""
RAG chain.

Flow:
  user question
    → similarity_search (ChromaDB)
    → build context string from top-k chunks
    → financial-tuned system prompt
    → LLM (GPT-4o-mini)
    → structured answer with source citations
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from typing import List, Tuple, Optional
import time

from core.config import settings
from services.vectorstore import VectorStoreService
from models.schemas import QueryResponse, Source


SYSTEM_PROMPT = """You are FinSight, an expert financial analyst AI assistant.
You answer questions strictly based on the provided financial document excerpts.

Rules you MUST follow:
1. Only use information from the CONTEXT below. Never hallucinate figures.
2. If the context does not contain enough information, say: "The provided documents do not contain sufficient information to answer this question."
3. Always cite the page number and document when referencing a specific figure or fact.
4. Format monetary values clearly (₹, $, crore, million, etc. as they appear in the document).
5. If you detect contradictory information across documents, flag it explicitly.
6. Be concise and structured. Use bullet points for lists of figures.

CONTEXT:
{context}

QUESTION: {question}

Answer (with citations):"""


def format_context(docs_scores: List[Tuple]) -> Tuple[str, List[Source]]:
    """Format retrieved chunks into context string and source objects."""
    context_parts = []
    sources = []

    for doc, score in docs_scores:
        meta = doc.metadata
        page = meta.get("page", 0) + 1  # 0-indexed → 1-indexed
        filename = meta.get("filename", "unknown")
        excerpt = doc.page_content[:200].replace("\n", " ")

        context_parts.append(
            f"[Source: {filename}, Page {page}, Relevance: {score:.2f}]\n{doc.page_content}"
        )
        sources.append(Source(page=page, filename=filename, excerpt=excerpt))

    return "\n\n---\n\n".join(context_parts), sources


class RAGChain:
    def __init__(self, vectorstore: VectorStoreService):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0,          # deterministic for financial data
            max_tokens=1500,
        )
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.output_parser = StrOutputParser()

    def query(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[Source], List[Tuple], int]:
        """Returns (answer, sources, raw_docs_scores, latency_ms)."""
        start = time.time()

        # Step 1: Retrieve
        docs_scores = self.vectorstore.similarity_search(question, doc_ids=doc_ids)
        if not docs_scores:
            return (
                "No relevant content found in the uploaded documents for this question.",
                [],
                [],
                int((time.time() - start) * 1000),
            )

        # Step 2: Format context
        context, sources = format_context(docs_scores)

        # Step 3: LLM call
        chain = self.prompt | self.llm | self.output_parser
        answer = chain.invoke({"context": context, "question": question})

        latency_ms = int((time.time() - start) * 1000)
        return answer, sources, docs_scores, latency_ms
