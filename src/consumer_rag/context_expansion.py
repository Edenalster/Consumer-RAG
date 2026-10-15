def expand_context(
    col,
    embedder,
    query: str,
    base_results: list[dict],
    extra_k: int = 5,
) -> list[dict]:
    """
    Expand retrieved context with additional chunks
    from the highest-ranked document.
    """

    if not base_results:
        return []

    # Identify the document of the top-ranked chunk
    top_doc_id = base_results[0]["meta"]["doc_id"]

    # Retrieve additional chunks from the same document
    res = col.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=extra_k,
        where={"doc_id": top_doc_id},
        include=["documents", "metadatas", "distances"],
    )

    extra_results = [
        {
            "id": chunk_id,
            "text": text,
            "meta": meta,
            "distance": distance,
        }
        for chunk_id, text, meta, distance in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ]

    # Merge while preserving order and removing duplicates
    seen = set()
    expanded_results = []

    for result in base_results + extra_results:
        if result["id"] not in seen:
            seen.add(result["id"])
            expanded_results.append(result)

    return expanded_results