from unittest.mock import MagicMock

from src.consumer_rag.context_expansion import expand_context


def make_chunk(chunk_id, doc_id):
    return {
        "id": chunk_id,
        "text": f"Text for {chunk_id}",
        "meta": {
            "doc_id": doc_id,
            "title": "Test document",
        },
        "distance": 0.1,
    }


def test_empty_results():
    col = MagicMock()
    embedder = MagicMock()

    result = expand_context(
        col,
        embedder,
        "test question",
        [],
    )

    assert result == []
    col.query.assert_not_called()


def test_expands_top_document():
    col = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]

    base_results = [
        make_chunk("doc-a::001", "doc-a"),
        make_chunk("doc-b::001", "doc-b"),
    ]

    col.query.return_value = {
        "ids": [["doc-a::002"]],
        "documents": [["Additional text"]],
        "metadatas": [[{"doc_id": "doc-a"}]],
        "distances": [[0.2]],
    }

    result = expand_context(
        col,
        embedder,
        "test question",
        base_results,
        extra_k=3,
    )

    assert len(result) == 3
    assert result[-1]["id"] == "doc-a::002"

    assert col.query.call_args.kwargs["where"] == {
        "doc_id": "doc-a"
    }


def test_removes_duplicates():
    col = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]

    base_results = [
        make_chunk("doc-a::001", "doc-a"),
    ]

    col.query.return_value = {
        "ids": [["doc-a::001", "doc-a::002"]],
        "documents": [["Original text", "Additional text"]],
        "metadatas": [[
            {"doc_id": "doc-a"},
            {"doc_id": "doc-a"},
        ]],
        "distances": [[0.1, 0.2]],
    }

    result = expand_context(
        col,
        embedder,
        "test question",
        base_results,
    )

    assert [chunk["id"] for chunk in result] == [
        "doc-a::001",
        "doc-a::002",
    ]


def test_preserves_original_order():
    col = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]

    base_results = [
        make_chunk("doc-a::003", "doc-a"),
        make_chunk("doc-b::001", "doc-b"),
        make_chunk("doc-a::001", "doc-a"),
    ]

    col.query.return_value = {
        "ids": [["doc-a::002"]],
        "documents": [["Additional text"]],
        "metadatas": [[{"doc_id": "doc-a"}]],
        "distances": [[0.2]],
    }

    result = expand_context(
        col,
        embedder,
        "test question",
        base_results,
    )

    assert [chunk["id"] for chunk in result] == [
        "doc-a::003",
        "doc-b::001",
        "doc-a::001",
        "doc-a::002",
    ]