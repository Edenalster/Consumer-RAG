from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv(".env")

from consumer_rag.rag import ask
from consumer_rag.retrieval import build_index
from consumer_rag.reranking import Reranker


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    number: int
    title: str
    url: str
    chunk_id: str
    distance: float | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    col, embedder, n_chunks = build_index(
        model_key="e5-small",
        max_chars=800,
        with_header=True,
    )

    app.state.col = col
    app.state.embedder = embedder
    app.state.n_chunks = n_chunks
    app.state.reranker = Reranker()

    yield


app = FastAPI(
    title="Consumer-RAG API",
    description="Hebrew consumer-rights RAG assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "indexed_chunks": app.state.n_chunks,
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        return ask(
            question=request.question,
            col=app.state.col,
            embedder=app.state.embedder,
            k=request.k,
            strategy="fusion_llm_boosted",
            reranker=app.state.reranker,
        )
    except Exception:
        import logging
        logging.getLogger(__name__).exception(
            "RAG request failed"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer",
        )
