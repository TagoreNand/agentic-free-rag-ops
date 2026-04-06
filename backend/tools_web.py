from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


def http_get_text(url: str, timeout_s: int = 20) -> str:
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "agentic-rag-ops/0.1"})
    resp.raise_for_status()
    return resp.text


def extract_main_text_from_html(html: str, max_chars: int = 25000) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate-ish sections
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    # Try common content containers
    candidates = []
    for sel in ["article", "main", "[role=main]", "div[class*=content]", "div[class*=article]"]:
        nodes = soup.select(sel)
        for n in nodes:
            txt = n.get_text(" ", strip=True)
            if len(txt) > 200:
                candidates.append(txt)
    if not candidates:
        candidates.append(soup.get_text(" ", strip=True))

    # Pick the longest candidate
    text = max(candidates, key=len)
    text = " ".join(text.split())
    if title and title not in text[:200]:
        text = f"{title}. {text}"
    return text[:max_chars]


def wikipedia_search(
    query: str,
    endpoint: str = "https://en.wikipedia.org/w/api.php",
    limit: int = 5,
    timeout_s: int = 20,
) -> List[Dict[str, str]]:
    """
    Uses MediaWiki search API (no key required).
    Returns [{title, page_url}, ...]
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": str(limit),
    }
    resp = requests.get(endpoint, params=params, timeout=timeout_s, headers={"User-Agent": "agentic-rag-ops/0.1"})
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("query", {}).get("search", []) or []
    out: List[Dict[str, str]] = []
    for h in hits[:limit]:
        title = h.get("title")
        if not title:
            continue
        out.append(
            {
                "title": title,
                "page_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            }
        )
    return out


def wikipedia_page_summary(page_title: str, summary_endpoint: str = "https://en.wikipedia.org/api/rest_v1/page/summary") -> str:
    url = f"{summary_endpoint}/{page_title}"
    resp = requests.get(url, timeout=20, headers={"User-Agent": "agentic-rag-ops/0.1"})
    resp.raise_for_status()
    data = resp.json()
    extract = data.get("extract") or ""
    return " ".join(extract.split()).strip()


def fetch_url(url: str) -> str:
    html = http_get_text(url)
    return extract_main_text_from_html(html)

