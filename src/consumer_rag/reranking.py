from sentence_transformers import CrossEncoder

from .retrieval import search
from .hybrid import reciprocal_rank_fusion


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
    ) -> list[dict]:
        if not results:
            return []

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, results),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        return [
            {**result, "rerank_score": float(score)}
            for score, result in ranked
        ]


def fused_search(
    col,
    embedder,
    reranker: Reranker,
    query: str,
    k: int = 5,
    candidate_k: int = 20,
) -> list[dict]:
    dense_results = search(
        col,
        embedder,
        query,
        k=candidate_k,
    )

    reranked_results = reranker.rerank(
        query,
        dense_results,
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        reranked_results,
        dense_weight=0.4,
        sparse_weight=0.6,
    )

    return fused_results[:k]