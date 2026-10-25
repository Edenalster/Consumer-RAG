def expand_context(
    col,
    embedder,
    query: str,
    base_results: list[dict],
    extra_k: int = 3,
    search_k: int = 10,
) -> list[dict]:
    """
    Expand context with additional unique chunks
    from the highest-ranked document.

    search_k: Number of candidates to retrieve.
    extra_k: Maximum number of new chunks to add.
    """

    if not base_results or extra_k <= 0:
        return list(base_results)

    top_doc_id = base_results[0]["meta"]["doc_id"]

    # Search more candidates than we intend to add.
    res = col.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=search_k,
        where={"doc_id": top_doc_id},
        include=["documents", "metadatas", "distances"],
    )

    seen = {chunk["id"] for chunk in base_results}
    expanded_results = list(base_results)

    added = 0

    for chunk_id, chunk_text, meta, distance in zip(
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
    ):
        if chunk_id in seen:
            continue

        expanded_results.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "meta": meta,
                "distance": distance,
            }
        )

        seen.add(chunk_id)
        added += 1

        if added >= extra_k:
            break

    return expanded_results