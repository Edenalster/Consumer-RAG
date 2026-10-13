import re
import unicodedata

# Everything after these markers is the site's footer, not content
FOOTER_MARKERS = ("אוחזר מתוך", "האם העמוד סייע לכם")
# Site promo text that appears in the middle of pages
PROMO_PHRASES = ("בחרו בכל זכות כאתר מועדף", "בלחיצה תועברו למסך של Google")
DROP_EXACT = {"?", "תוכן עניינים"}


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    lines = [
        ln for ln in text.splitlines()
        if not any(p in ln for p in PROMO_PHRASES) and ln.strip() not in DROP_EXACT
    ]
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()