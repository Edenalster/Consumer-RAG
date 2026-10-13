import json
import time
from datetime import date
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .text_utils import clean_text

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "consumer-rag-portfolio/0.1 (personal learning project)"
CONTENT_SELECTORS = ["#mw-content-text", "main", "article", "body"]


def check_robots(url: str) -> tuple[bool, str]:
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        r = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=20)
    except requests.RequestException as e:
        return False, f"could not read robots.txt ({type(e).__name__})"
    if r.status_code in (401, 403):
        return False, f"robots.txt returned {r.status_code}"
    if r.status_code >= 500:
        return False, f"robots.txt server error {r.status_code}, try again later"
    if r.status_code >= 400:
        return True, "no robots.txt (no restrictions)"
    rp = robotparser.RobotFileParser()
    rp.parse(r.text.splitlines())
    ok = rp.can_fetch(USER_AGENT, url)
    return ok, "allowed" if ok else "disallowed by robots.txt"


def extract_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "form"]):
        tag.decompose()
    node = next((n for sel in CONTENT_SELECTORS if (n := soup.select_one(sel))), soup)
    return title, clean_text(node.get_text("\n"))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    for s in sources:
        ok, reason = check_robots(s["url"])
        if not ok:
            print(f"SKIP {s['id']}: {reason}")
            continue
        r = requests.get(s["url"], headers={"User-Agent": USER_AGENT}, timeout=20)
        if r.status_code != 200:
            print(f"FAILED {r.status_code}: {s['id']}")
            continue
        r.encoding = "utf-8"
        title, text = extract_text(r.text)
        doc = {
            "id": s["id"], "url": s["url"], "title": title,
            "source": s["source"], "license": s["license"],
            "retrieved": date.today().isoformat(), "text": text,
        }
        (RAW_DIR / f"{s['id']}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"OK {s['id']}: {len(text)} characters")
        time.sleep(2)   # be polite: one request every 2 seconds


if __name__ == "__main__":
    main()