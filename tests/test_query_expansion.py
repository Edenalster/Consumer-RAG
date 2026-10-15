import pytest

from src.consumer_rag.query_expansion import expand_query


@pytest.mark.parametrize(
    "question, expected_term",
    [
        (
            "מישהו הגיע אליי הביתה ומכר לי מוצר",
            "עסקה ברוכלות",
        ),
        (
            "סוכן הגיע לבית והציע לי שירות",
            "עסקה ברוכלות",
        ),
        (
            "מוכר הגיע לבית ורכשתי ממנו מוצר",
            "עסקה ברוכלות",
        ),
    ],
)
def test_adds_legal_term(question, expected_term):
    result = expand_query(question)

    assert expected_term in result


def test_does_not_duplicate_existing_term():
    question = "מהי עסקה ברוכלות?"

    result = expand_query(question)

    assert result == question


def test_does_not_expand_unrelated_question():
    question = "הזמנתי מוצר באינטרנט. אפשר להחזיר אותו?"

    result = expand_query(question)

    assert result == question


def test_preserves_original_question():
    question = "סוכן הגיע לבית ומכר לי מוצר"

    result = expand_query(question)

    assert result.startswith(question)


def test_expansion_is_idempotent():
    question = "מישהו הגיע אליי הביתה ומכר לי מוצר"

    once = expand_query(question)
    twice = expand_query(once)

    assert once == twice


@pytest.mark.parametrize(
    "question",
    [
        "אני משלם כל חודש על שירות ורוצה להפסיק אותו",
        "יש לי מנוי חודשי ואני רוצה לבטל",
        "מחייבים אותי כל חודש למרות שביקשתי להתנתק",
    ],
)
def test_expands_ongoing_transaction(question):
    result = expand_query(question)

    assert "עסקה מתמשכת" in result
    assert result.startswith(question)


def test_does_not_duplicate_ongoing_transaction():
    question = "איך מבטלים עסקה מתמשכת?"

    result = expand_query(question)

    assert result == question


def test_ongoing_expansion_is_idempotent():
    question = "יש לי מנוי חודשי ואני רוצה לבטל"

    once = expand_query(question)
    twice = expand_query(once)

    assert once == twice


def test_does_not_add_ongoing_term_to_online_purchase():
    question = "הזמנתי מוצר באינטרנט ורוצה להחזיר אותו"

    result = expand_query(question)

    assert "עסקה מתמשכת" not in result
