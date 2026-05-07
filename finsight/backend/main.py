from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from services.vectorstore import VectorStoreService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialise the vector store once and attach it to app.state so all
    # request handlers can share the same ChromaDB connection.
    app.state.vectorstore = VectorStoreService()
    yield
    # Cleanup (ChromaDB handles persistence internally)


app = FastAPI(
    title="FinSight API",
    description="Production-grade RAG chatbot for financial document analysis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
