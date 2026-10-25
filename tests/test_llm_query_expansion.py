from unittest.mock import patch

from src.consumer_rag.legal_classifier import expand_query_with_llm


def test_expands_query_with_legal_category():
    """
    Should append the legal category when classification succeeds.
    """

    question = "נציג מכירות הגיע אליי הביתה ומכר לי מוצר."

    with patch(
        "src.consumer_rag.legal_classifier.classify_legal_query",
        return_value="עסקה ברוכלות",
    ):
        result = expand_query_with_llm(question)

    assert result == f"{question} עסקה ברוכלות"


def test_unknown_category_does_not_expand():
    """
    Should preserve the original question when category is unknown.
    """

    question = "קניתי מוצר ואני רוצה לבטל את העסקה."

    with patch(
        "src.consumer_rag.legal_classifier.classify_legal_query",
        return_value="לא ידוע",
    ):
        result = expand_query_with_llm(question)

    assert result == question


def test_existing_category_is_not_duplicated():
    """
    Should not append a category already present in the question.
    """

    question = "מה התנאים לביטול עסקה ברוכלות?"

    with patch(
        "src.consumer_rag.legal_classifier.classify_legal_query",
        return_value="עסקה ברוכלות",
    ):
        result = expand_query_with_llm(question)

    assert result == question