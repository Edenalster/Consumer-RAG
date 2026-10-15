from src.consumer_rag.rag import extract_citations


def test_single_citation():
    answer = "אפשר לבטל את העסקה בתוך יומיים [מקור 1]."

    assert extract_citations(answer, 5) == {1}


def test_multiple_citations():
    answer = "דמי הביטול מוגבלים בחוק [מקור 1, מקור 2, מקור 5]."

    assert extract_citations(answer, 5) == {1, 2, 5}


def test_mixed_citations():
    answer = """
    דמי הביטול מוגבלים בחוק [מקור 1, מקור 2].
    קיימים חריגים [מקור 3].
    """

    assert extract_citations(answer, 5) == {1, 2, 3}


def test_invalid_citations():
    answer = "מידע כללי [מקור 0] [מקור 8] [מקור 2]."

    assert extract_citations(answer, 5) == {2}


def test_no_citations():
    answer = "לא נמצא מידע רלוונטי במקורות."

    assert extract_citations(answer, 5) == set()

from unittest.mock import patch
from src.consumer_rag.rag import ask


@patch("src.consumer_rag.rag.generate_answer")
@patch("src.consumer_rag.rag.search")
def test_ask_returns_only_cited_sources(mock_search, mock_generate):
    mock_search.return_value = [
        {
            "id": f"chunk-{i}",
            "text": f"Text {i}",
            "meta": {
                "title": f"Document {i}",
                "url": f"https://example.com/{i}",
            },
            "distance": 0.1,
        }
        for i in range(1, 6)
    ]

    mock_generate.return_value = (
        "דמי הביטול מוגבלים בחוק [מקור 1, מקור 3]."
    )

    result = ask(
        question="מה גובה דמי הביטול?",
        col=None,
        embedder=None,
        k=5,
    )

    assert len(result["sources"]) == 2

    assert [source["number"] for source in result["sources"]] == [1, 3]

    assert [source["chunk_id"] for source in result["sources"]] == [
        "chunk-1",
        "chunk-3",
    ]

    mock_generate.assert_called_once()

from unittest.mock import patch
import pytest

from src.consumer_rag.rag import ask


def test_ask_uses_dense_strategy():
    with (
        patch("src.consumer_rag.rag.search", return_value=[]) as mock_search,
        patch(
            "src.consumer_rag.rag.generate_answer",
            return_value="לא נמצא מידע רלוונטי.",
        ),
    ):
        result = ask(
            question="שאלת בדיקה",
            col=None,
            embedder=None,
            strategy="dense",
        )

    mock_search.assert_called_once()
    assert result["sources"] == []


def test_ask_uses_fusion_strategy():
    with (
        patch("src.consumer_rag.rag.fused_search", return_value=[]) as mock_fusion,
        patch(
            "src.consumer_rag.rag.generate_answer",
            return_value="לא נמצא מידע רלוונטי.",
        ),
    ):
        result = ask(
            question="שאלת בדיקה",
            col=None,
            embedder=None,
            strategy="fusion",
            reranker=object(),
        )

    mock_fusion.assert_called_once()
    assert result["sources"] == []


def test_fusion_requires_reranker():
    with pytest.raises(ValueError, match="requires a reranker"):
        ask(
            question="שאלת בדיקה",
            col=None,
            embedder=None,
            strategy="fusion",
        )

def test_ask_uses_fusion_expanded_strategy():
    from unittest.mock import patch

    from src.consumer_rag.rag import ask

    base_chunks = [
        {
            "id": "doc-a::001",
            "text": "Original text",
            "meta": {
                "title": "Document A",
                "url": "https://example.com/a",
            },
            "distance": 0.1,
        }
    ]

    expanded_chunks = base_chunks + [
        {
            "id": "doc-a::002",
            "text": "Additional context",
            "meta": {
                "title": "Document A",
                "url": "https://example.com/a",
            },
            "distance": 0.2,
        }
    ]

    with (
        patch(
            "src.consumer_rag.rag.fused_search",
            return_value=base_chunks,
        ) as mock_fusion,
        patch(
            "src.consumer_rag.rag.expand_context",
            return_value=expanded_chunks,
        ) as mock_expand,
        patch(
            "src.consumer_rag.rag.generate_answer",
            return_value="תשובה המבוססת על ההקשר הנוסף [מקור 2].",
        ) as mock_generate,
    ):
        result = ask(
            question="שאלת בדיקה",
            col=None,
            embedder=None,
            strategy="fusion_expanded",
            reranker=object(),
        )

    # Fusion retrieval was executed
    mock_fusion.assert_called_once()

    # Context expansion was executed
    mock_expand.assert_called_once()

    # Expansion received the original retrieval results
    assert mock_expand.call_args.kwargs["base_results"] == base_chunks

    # Gemini received the expanded context
    assert mock_generate.call_args.kwargs["chunks"] == expanded_chunks

    # Only the cited source is returned
    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_id"] == "doc-a::002"
    assert result["sources"][0]["number"] == 2