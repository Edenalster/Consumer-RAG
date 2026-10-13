import json
import sys
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .fetch import USER_AGENT, check_robots


def main(hub_url: str) -> None:
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
        seen.setdefault(href, a.get_text(strip=True))

    print(f"{len(seen)} candidates. Delete the lines you do not want, then paste into sources.json:\n")
    for i, (url, title) in enumerate(seen.items(), 1):
        entry = {"id": f"kz-{i:02d}", "title": title, "source": "Kol-Zchut",
                 "license": "CC BY-NC-SA 2.5 IL", "url": unquote(url)}
        print(json.dumps(entry, ensure_ascii=False) + ",")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "https://www.kolzchut.org.il/he/צרכנות")