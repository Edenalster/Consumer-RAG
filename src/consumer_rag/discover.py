import hashlib
import json
import sys
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .fetch import ROOT, USER_AGENT, check_robots


def main(hub_url: str, keywords: list[str]) -> None:
    ok, reason = check_robots(hub_url)
    if not ok:
        print(f"Stopped: {reason}")
        return
    r = requests.get(hub_url, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.select("#mw-content-text a[href]") or soup.select("a[href]")

    seen: dict[str, str] = {}
    for a in links:
        href = urljoin(hub_url, a["href"]).split("#")[0]
        path = unquote(urlparse(href).path)
        if not path.startswith("/he/") or ":" in path or path == urlparse(hub_url).path:
            continue
        title = a.get_text(strip=True)
        if keywords and not any(k in title or k in path for k in keywords):
            continue
        seen.setdefault(unquote(href), title)

    existing = {unquote(s["url"]) for s in json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))}
    fresh = {u: t for u, t in seen.items() if u not in existing}
    print(f"{len(fresh)} new candidates. Delete the lines you do not want, then paste into sources.json:\n")
    for url, title in fresh.items():
        entry = {"id": "kz-" + hashlib.sha1(url.encode()).hexdigest()[:6], "title": title,
                 "source": "Kol-Zchut", "license": "CC BY-NC-SA 2.5 IL", "url": url}
        print(json.dumps(entry, ensure_ascii=False) + ",")


if __name__ == "__main__":
    hub = sys.argv[1] if len(sys.argv) > 1 else "https://www.kolzchut.org.il/he/צרכנות"
    kws = [k for k in (sys.argv[2].split(",") if len(sys.argv) > 2 else []) if k]
    main(hub, kws)