from unittest.mock import MagicMock, patch

from src.consumer_rag.rag import ask


def test_fusion_llm_expanded_pipeline():

    question = "נציג מכירות הגיע אליי הביתה ומכר לי מוצר"
    expanded_question = question + " עסקה ברוכלות"

    dense_results = [
        {
            "id": "cancel-doorstep::001",
            "text": "מידע על ביטול עסקה ברוכלות",
            "meta": {
                "title": "ביטול עסקה ברוכלות",
                "url": "https://example.com/doorstep",
            },
            "distance": 0.1,
        }
    ]

    reranked_results = dense_results.copy()
    fused_results = dense_results.copy()

    reranker = MagicMock()
    reranker.rerank.return_value = reranked_results

    with (
        patch(
            "src.consumer_rag.rag.expand_query_with_llm",
            return_value=expanded_question,
        ) as mock_expand,

        patch(
            "src.consumer_rag.rag.search",
            return_value=dense_results,
        ) as mock_search,

        patch(
            "src.consumer_rag.rag.reciprocal_rank_fusion",
            return_value=fused_results,
        ) as mock_fusion,

        patch(
            "src.consumer_rag.rag.generate_answer",
            return_value="ניתן לבטל את העסקה בתנאים מסוימים [מקור 1].",
        ) as mock_generate,
    ):

        result = ask(
            question=question,
            col=None,
            embedder=None,
            strategy="fusion_llm_expanded",
            reranker=reranker,
            k=5,
        )

    # 1. Query expansion received the original question
    mock_expand.assert_called_once_with(question)

    # 2. Dense retrieval received the expanded question
    mock_search.assert_called_once_with(
        None,
        None,
        expanded_question,
        k=20,
    )

    # 3. Reranker received the original question
    reranker.rerank.assert_called_once_with(
        question,
        dense_results,
    )

    # 4. Fusion received both result lists
    mock_fusion.assert_called_once_with(
        dense_results,
        reranked_results,
        dense_weight=0.4,
        sparse_weight=0.6,
    )

    # 5. Generation received the original question
    mock_generate.assert_called_once_with(
        question=question,
        chunks=fused_results,
    )

    # 6. The cited source was returned
    assert result["question"] == question
    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_id"] == "cancel-doorstep::001"