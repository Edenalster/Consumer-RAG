import json
import re
from pathlib import Path

TITLE_SUFFIX = re.compile(r"\s*[–-]\s*כל-זכות\s*$")
ZCHUT_TAG = re.compile(r"\s*\(זכות\)\s*")


def clean_title(title: str) -> str:
    return ZCHUT_TAG.sub(" ", TITLE_SUFFIX.sub("", title)).strip()


def load_docs(raw_dir) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(raw_dir).glob("*.json"))]


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split on '## ' headings -> [(heading, body)]. Text before the first heading gets heading ''."""
    parts = re.split(r"(?m)^## ", text)
    sections = []
    if parts[0].strip():
        sections.append(("", parts[0].strip()))
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        sections.append((heading.strip(), body.strip()))
    return sections


# Sections that only list links, phone numbers or credits
NAV_HEADINGS = {"ארגוני סיוע", "גורמי ממשל", "מוקדים ממשלתיים", "גורמים מסייעים",
                "תודות", "ראו גם", "הרחבות ופרסומים"}


def chunk_doc(doc: dict, max_chars: int = 800, with_header: bool = True,
              min_chars: int = 60, drop_nav: bool = True) -> list[dict]:
    title = clean_title(doc.get("title", "")) or doc["id"]
    chunks: list[dict] = []
    for heading, body in split_sections(doc["text"]):
        if drop_nav and heading in NAV_HEADINGS:
            continue
        paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
        pieces, buf = [], ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 1 > max_chars:
                pieces.append(buf)
                buf = para
            else:
                buf = f"{buf}\n{para}" if buf else para
        if buf:
            pieces.append(buf)
        for piece in pieces:
            if len(piece) < min_chars:
                continue                      # too short to be a useful answer
            header = f"{title} | {heading}" if heading else title
            chunks.append({
                "id": f"{doc['id']}::{len(chunks):03d}",
                "text": f"{header}\n{piece}" if with_header else piece,
                "meta": {"doc_id": doc["id"], "url": doc["url"], "title": title,
                         "heading": heading, "retrieved": doc["retrieved"]},
            })
    return chunks