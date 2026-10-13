from src.consumer_rag.chunking import chunk_doc, clean_title, split_sections


def test_clean_title():
    assert clean_title("עסקה צרכנית שלא ניתן לבטל (זכות) – כל-זכות") == "עסקה צרכנית שלא ניתן לבטל"


def test_split_sections():
    text = "פתיחה\n## כותרת א\nשורה 1\n## כותרת ב\nשורה 2"
    assert split_sections(text) == [("", "פתיחה"), ("כותרת א", "שורה 1"), ("כותרת ב", "שורה 2")]


def test_chunks_respect_max_chars_and_keep_header():
    doc = {"id": "d", "title": "כותרת הדף (זכות) – כל-זכות", "url": "u", "retrieved": "2026-01-01",
           "text": "## סעיף\n" + "\n".join(["משפט ארוך מאוד " * 5] * 6)}
    chunks = chunk_doc(doc, max_chars=200)
    assert len(chunks) > 1
    assert all(c["text"].startswith("כותרת הדף | סעיף") for c in chunks)
    assert all(len(c["text"].split("\n", 1)[1]) <= 200 for c in chunks)


def test_tiny_and_navigation_chunks_are_dropped():
    doc = {"id": "d", "title": "דף", "url": "u", "retrieved": "2026-01-01",
           "text": "## תוכן\n" + "משפט תוכן מספיק ארוך כדי להישאר באינדקס. " * 3
                   + "\n## ארגוני סיוע\nרשימת ארגונים ארוכה מאוד שלא צריכה להיכנס לאינדקס בכלל כי היא ניווט"
                   + "\n## קצר\nקצר"}
    chunks = chunk_doc(doc)
    assert [c["meta"]["heading"] for c in chunks] == ["תוכן"]
