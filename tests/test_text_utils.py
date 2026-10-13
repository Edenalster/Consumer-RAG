from src.consumer_rag.text_utils import clean_text


def test_footer_is_cut():
    text = 'תוכן הדף\n\nשורה שנייה\n\nאוחזר מתוך "https://example.org"\nתפריט'
    assert clean_text(text) == "תוכן הדף\n\nשורה שנייה"


def test_whitespace_is_collapsed():
    assert clean_text("שלום    עולם\n\n\n\nשוב") == "שלום עולם\n\nשוב"