from unittest.mock import Mock

from src.consumer_rag.reranking import Reranker


def test_rerank_sorts_by_score():
    reranker = Reranker.__new__(Reranker)

    reranker.model = Mock()
    reranker.model.predict.return_value = [0.2, 0.9, 0.5]

    results = [
        {"id": "chunk-1", "text": "First chunk"},
        {"id": "chunk-2", "text": "Second chunk"},
        {"id": "chunk-3", "text": "Third chunk"},
    ]

    ranked = reranker.rerank("test question", results)

    assert [item["id"] for item in ranked] == [
        "chunk-2",
        "chunk-3",
        "chunk-1",
    ]

    assert ranked[0]["rerank_score"] == 0.9


def test_rerank_empty_results():
    reranker = Reranker.__new__(Reranker)

    reranker.model = Mock()

    ranked = reranker.rerank("test question", [])

    assert ranked == []
    reranker.model.predict.assert_not_called()