import re


LEGAL_TERMS = {
    "עסקה ברוכלות": [
        "הגיע אליי הביתה",
        "הגיע אלי הביתה",
        "הגיע אליי לבית",
        "מוכר הגיע לבית",
        "סוכן הגיע לבית",
        "מכירה בבית",
        "מכירה מדלת לדלת",
    ],
    "עסקה מתמשכת": [
        "משלם כל חודש",
        "משלמת כל חודש",
        "מחייבים אותי כל חודש",
        "מנוי חודשי",
        "מתחדש כל חודש",
        "חיוב חודשי",
    ],
}


def expand_query(question: str) -> str:
    """
    Add relevant legal terminology to a user's question
    using predefined phrase matching.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        question.strip().lower(),
    )

    additions = []

    for legal_term, phrases in LEGAL_TERMS.items():
        if legal_term in normalized:
            continue

        if any(
            phrase in normalized
            for phrase in phrases
        ):
            additions.append(legal_term)

    if not additions:
        return question

    return question + " " + " ".join(additions)
