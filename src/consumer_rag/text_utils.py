import re
import unicodedata

# Everything after these markers is the site's footer, not content
FOOTER_MARKERS = ("אוחזר מתוך", "האם העמוד סייע לכם")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for marker in FOOTER_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()