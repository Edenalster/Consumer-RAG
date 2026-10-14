import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple Hebrew-friendly tokenization."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [
            {
                **self.chunks[i],
                "bm25_score": float(scores[i])
            }
            for i in ranked_indices[:k]
        ]


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    dense_weight: float = 0.8,
    sparse_weight: float = 0.2,
) -> list[dict]:
    """Combine ranked results using weighted Reciprocal Rank Fusion."""

    scores = {}
    documents = {}

    for results, weight in [
        (dense_results, dense_weight),
        (sparse_results, sparse_weight),
    ]:
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + weight / (k + rank)
            )

            documents[chunk_id] = result

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    return [
        {
            **documents[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        for chunk_id in ranked_ids
    ]

def hybrid_search(
    col,
    embedder,
    bm25_retriever: BM25Retriever,
    query: str,
    k: int = 5,
    candidate_k: int = 20
) -> list[dict]:
    """Retrieve with dense and BM25, then combine rankings using RRF."""

    from .retrieval import search

    dense_results = search(
        col,
        embedder,
        query,
        k=candidate_k
    )

    sparse_results = bm25_retriever.search(
        query,
        k=candidate_k
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results
    )

    return fused_results[:k]