
from unittest.mock import Mock, patch

import pytest

from src.consumer_rag.rag import ask


MODULE = "src.consumer_rag.rag"


def make_chunk(chunk_id):
    return {
        "id": chunk_id,
        "text": f"Content for {chunk_id}",
        "meta": {
            "title": "Test document",
            "url": "https://example.com",
            "doc_id": "cancel-doorstep",
        },
        "distance": 0.1,
    }


@patch(f"{MODULE}.generate_answer")
@patch(f"{MODULE}.boost_by_category_linear")
@patch(f"{MODULE}.reciprocal_rank_fusion")
@patch(f"{MODULE}.search")
@patch(f"{MODULE}.classify_legal_query")
def test_boosted_pipeline(
    mock_classify,
    mock_search,
    mock_fusion,
    mock_boost,
    mock_generate,
):
    question = "קניתי מוצר בבית, אפשר לבטל?"
    category = "עסקה ברוכלות"
    expanded = f"{question} {category}"

    dense_results = [make_chunk("dense-1")]
    reranked_results = [make_chunk("reranked-1")]
    fused_results = [make_chunk(f"fused-{i}") for i in range(5)]
    boosted_results = [make_chunk(f"boosted-{i}") for i in range(5)]

    reranker = Mock()
    reranker.rerank.return_value = reranked_results

    mock_classify.return_value = category
    mock_search.return_value = dense_results
    mock_fusion.return_value = fused_results
    mock_boost.return_value = boosted_results
    mock_generate.return_value = "תשובה [מקור 1]"

    result = ask(
        question=question,
        col="fake_col",
        embedder="fake_embedder",
        k=3,
        strategy="fusion_llm_boosted",
        reranker=reranker,
    )

    mock_classify.assert_called_once_with(question)

    mock_search.assert_called_once_with(
        "fake_col",
        "fake_embedder",
        expanded,
        k=20,
    )

    reranker.rerank.assert_called_once_with(
        expanded,
        dense_results,
    )

    mock_fusion.assert_called_once_with(
        dense_results,
        reranked_results,
        dense_weight=0.4,
        sparse_weight=0.6,
    )

    mock_boost.assert_called_once_with(
        fused_results,
        category,
        boost=10,
    )

    mock_generate.assert_called_once_with(
        question=question,
        chunks=boosted_results[:3],
    )

    assert result["answer"] == "תשובה [מקור 1]"
    assert result["sources"][0]["chunk_id"] == "boosted-0"
    assert len(result["sources"]) == 1


@patch(f"{MODULE}.generate_answer")
@patch(f"{MODULE}.boost_by_category_linear")
@patch(f"{MODULE}.reciprocal_rank_fusion")
@patch(f"{MODULE}.search")
@patch(f"{MODULE}.classify_legal_query")
def test_unknown_category_does_not_expand_question(
    mock_classify,
    mock_search,
    mock_fusion,
    mock_boost,
    mock_generate,
):
    question = "אפשר להחזיר חולצה?"
    chunks = [make_chunk("chunk-1")]

    reranker = Mock()
    reranker.rerank.return_value = chunks

    mock_classify.return_value = "לא ידוע"
    mock_search.return_value = chunks
    mock_fusion.return_value = chunks
    mock_boost.return_value = chunks
    mock_generate.return_value = "תשובה [מקור 1]"

    ask(
        question=question,
        col="fake_col",
        embedder="fake_embedder",
        strategy="fusion_llm_boosted",
        reranker=reranker,
    )

    mock_search.assert_called_once_with(
        "fake_col",
        "fake_embedder",
        question,
        k=20,
    )

    reranker.rerank.assert_called_once_with(
        question,
        chunks,
    )


def test_boosted_requires_reranker():
    with pytest.raises(ValueError, match="requires a reranker"):
        ask(
            question="אפשר לבטל עסקה?",
            col="fake_col",
            embedder="fake_embedder",
            strategy="fusion_llm_boosted",
            reranker=None,
        )
