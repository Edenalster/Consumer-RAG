from src.consumer_rag.text_utils import clean_text


def test_footer_is_cut():
    text = 'תוכן הדף\n\nשורה שנייה\n\nאוחזר מתוך "https://example.org"\nתפריט'
    assert clean_text(text) == "תוכן הדף\n\nשורה שנייה"


def test_whitespace_is_collapsed():
    assert clean_text("שלום    עולם\n\n\n\nשוב") == "שלום עולם\n\nשוב"

def test_promo_lines_are_removed():
    text = "פסקה ראשונה\nבחרו בכל זכות כאתר מועדף בגוגל\n?\nבלחיצה תועברו למסך של Google בו תוכלו\nפסקה שנייה"
    assert clean_text(text) == "פסקה ראשונה\nפסקה שנייה"