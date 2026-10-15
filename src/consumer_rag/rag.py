import re

from .retrieval import search
from .generation import generate_answer
from .reranking import Reranker, fused_search
from .context_expansion import expand_context
from .hybrid_retrieval import hybrid_search

def extract_citations(answer: str, num_chunks: int) -> set[int]:
    """Extract valid source numbers from an LLM-generated answer."""

    citation_groups = re.findall(
        r"\[([^\[\]]*מקור\s+\d+[^\[\]]*)\]",
        answer,
    )

    cited_numbers = set()

    for group in citation_groups:
        numbers = re.findall(r"מקור\s+(\d+)", group)
        cited_numbers.update(int(number) for number in numbers)

    return {
        number
        for number in cited_numbers
        if 1 <= number <= num_chunks
    }


def ask(
    question: str,
    col,
    embedder,
    k: int = 5,
    strategy: str = "dense",
    reranker: Reranker | None = None,
    bm25=None,
) -> dict:

    # Step 1: Retrieve relevant chunks
    if strategy == "dense":
        chunks = search(
            col,
            embedder,
            question,
            k=k,
        )

    elif strategy in ("fusion", "fusion_expanded"):
        if reranker is None:
            raise ValueError(
                f"{strategy} strategy requires a reranker"
            )

        chunks = fused_search(
            col,
            embedder,
            reranker,
            question,
            k=k,
            candidate_k=20,
        )

        # Expand context only for fusion_expanded
        if strategy == "fusion_expanded":
            chunks = expand_context(
                col=col,
                embedder=embedder,
                query=question,
                base_results=chunks,
                extra_k=5,
            )

    elif strategy == "hybrid":
        if reranker is None:
            raise ValueError(
                "hybrid strategy requires a reranker"
            )

        if bm25 is None:
            raise ValueError(
                "hybrid strategy requires a BM25 retriever"
            )

        chunks = hybrid_search(
            question=question,
            col=col,
            embedder=embedder,
            bm25=bm25,
            reranker=reranker,
            dense_k=5,
            bm25_k=5,
            final_k=k,
        )

    else:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy}"
        )

    # Step 2: Generate answer
    answer = generate_answer(
        question=question,
        chunks=chunks,
    )

    # Step 3: Extract citations
    cited_numbers = extract_citations(
        answer,
        len(chunks),
    )

    # Step 4: Return only cited sources
    sources = [
        {
            "number": i,
            "title": chunk["meta"]["title"],
            "url": chunk["meta"]["url"],
            "chunk_id": chunk["id"],
            "distance": chunk.get("distance"),
        }
        for i, chunk in enumerate(chunks, start=1)
        if i in cited_numbers
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }