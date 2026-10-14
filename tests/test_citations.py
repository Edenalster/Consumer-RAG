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