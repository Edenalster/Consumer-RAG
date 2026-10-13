from src.consumer_rag.evaluate import first_hit_rank, summarize, validate


def test_first_hit_rank_requires_right_doc_and_phrase():
    results = [
        {"meta": {"doc_id": "a"}, "text": "בתוך יומיים"},    # right phrase, wrong page
        {"meta": {"doc_id": "b"}, "text": "משהו אחר"},       # right page, missing phrase
        {"meta": {"doc_id": "b"}, "text": "בתוך   יומיים"},  # hit (whitespace is normalised)
    ]
    assert first_hit_rank(results, "b", "בתוך יומיים") == 3
    assert first_hit_rank(results, "c", "בתוך יומיים") is None


def test_summarize():
    s = summarize([1, 2, None, 5])
    assert s["recall@1"] == 0.25 and s["recall@3"] == 0.5 and s["recall@5"] == 0.75
    assert round(s["mrr"], 3) == round((1 + 0.5 + 0 + 0.2) / 4, 3)


def test_validate_catches_bad_phrase_and_unknown_doc():
    chunks = {"a": ["כותרת\nבתוך   יומיים מהקנייה"]}
    questions = [
        {"q": "x", "doc": "a", "phrase": "בתוך יומיים", "type": "direct"},   # fine
        {"q": "x", "doc": "a", "phrase": "ביטוי שלא קיים", "type": "direct"},
        {"q": "x", "doc": "zzz", "phrase": "בתוך", "type": "direct"},
    ]
    assert len(validate(questions, chunks)) == 2
