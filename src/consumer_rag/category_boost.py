from __future__ import annotations


CATEGORY_TO_DOC_ID = {
    "עסקה ברוכלות": "cancel-doorstep",
    "עסקה מתמשכת": "cancel-ongoing",
    "עסקת מכר מרחוק": "cancel-online",
    "ביטול עסקה בחנות": "cancel-in-store",
}


def boost_by_category(
    chunks: list[dict],
    category: str,
    boost: float = 0.05,
) -> list[dict]:
    """
    Reorder ranked chunks by giving a small bonus to documents
    matching the predicted legal category.

    The original list is not modified.
    """

    if boost < 0:
        raise ValueError("boost must be non-negative")

    preferred_doc_id = CATEGORY_TO_DOC_ID.get(category)

    if preferred_doc_id is None or boost == 0:
        return list(chunks)

    ranked = []

    for rank, chunk in enumerate(chunks, start=1):
        # Rank-based score: higher rank gets a higher score.
        base_score = 1.0 / rank

        doc_id = chunk.get("meta", {}).get("doc_id")

        category_bonus = (
            boost if doc_id == preferred_doc_id else 0.0
        )

        boosted_score = base_score + category_bonus

        ranked.append(
            (boosted_score, rank, chunk)
        )

    ranked.sort(
        key=lambda item: (-item[0], item[1])
    )

    return [
        chunk
        for _, _, chunk in ranked
    ]
def boost_by_category_linear(
    chunks: list[dict],
    category: str,
    boost: float = 3.0,
) -> list[dict]:
    """
    Reorder chunks using a linear rank score and
    a category-specific bonus.

    The original list is not modified.
    """

    if boost < 0:
        raise ValueError("boost must be non-negative")

    preferred_doc_id = CATEGORY_TO_DOC_ID.get(category)

    if preferred_doc_id is None or boost == 0:
        return list(chunks)

    total_chunks = len(chunks)
    ranked = []

    for rank, chunk in enumerate(chunks, start=1):
        base_score = total_chunks + 1 - rank

        doc_id = chunk.get("meta", {}).get("doc_id")

        category_bonus = (
            boost if doc_id == preferred_doc_id else 0.0
        )

        boosted_score = base_score + category_bonus

        ranked.append(
            (boosted_score, rank, chunk)
        )

    ranked.sort(
        key=lambda item: (-item[0], item[1])
    )

    return [
        chunk
        for _, _, chunk in ranked
    ]