from .retrieval import search


def hybrid_search(
    question,
    col,
    embedder,
    bm25,
    reranker,
    dense_k=5,
    bm25_k=5,
    final_k=5,
):
    # Retrieve semantic candidates
    dense_results = search(
        col,
        embedder,
        question,
        k=dense_k,
    )

    # Retrieve lexical candidates
    bm25_results = bm25.search(
        question,
        k=bm25_k,
    )

    # Merge candidates and remove duplicates
    seen = set()
    candidates = []

    for result in dense_results + bm25_results:
        if result["id"] not in seen:
            seen.add(result["id"])
            candidates.append(result)

    # Rerank all unique candidates
    reranked = reranker.rerank(
        question,
        candidates,
    )

    return reranked[:final_k]