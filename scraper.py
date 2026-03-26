"""
ODN Systems Website Scraper (Improved Version)
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse
from collections import deque
from typing import Optional, List, Dict

BASE_URL = "https://odnsystems.com"
OUTPUT_FILE = "knowledge_base.json"
MAX_PAGES = 20  # limit crawl for safety

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

SKIP_PATTERNS = [
    "wp-admin", "wp-login", "wp-json", "feed", "xmlrpc",
    "?s=", "#", "mailto:", "tel:", ".jpg", ".png", ".pdf",
    ".gif", ".svg", ".css", ".js"
]

session = requests.Session()
session.headers.update(HEADERS)


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != "odnsystems.com":
        return False
    for pattern in SKIP_PATTERNS:
        if pattern in url:
            return False
    return url.startswith(BASE_URL) or url.startswith("/")


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and len(line) > 2]
    return " ".join(lines)


def fetch_html(url: str) -> Optional[str]:
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                print(f"  Status {response.status_code} for {url}")
        except Exception as e:
            print(f"  Retry {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def scrape_page(url: str) -> Optional[Dict]:
    html = fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "noscript", "iframe",
                     "header", "footer"]):
        tag.decompose()

    # Title
    title = ""
    if soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # Meta description
    meta_desc = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta:
        meta_desc = meta.get("content", "")

    # Main content
    main = soup.find("main") or soup.body
    raw_text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    content = clean_text(raw_text)

    # Headings
    headings = [h.get_text(strip=True) for h in soup.find_all(["h2", "h3"])]

    # Internal links
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        if is_valid_url(href) and href not in links:
            links.append(href)

    return {
        "url": url,
        "title": title,
        "meta_description": meta_desc,
        "headings": headings[:15],
        "content": content[:5000],
        "internal_links": links
    }


def crawl_website() -> List[Dict]:
    visited = set()
    queue = deque([BASE_URL])
    pages = []

    print(f"Starting crawl of {BASE_URL}...\n")

    while queue and len(pages) < MAX_PAGES:
        url = queue.popleft().rstrip("/")

        if url in visited:
            continue
        visited.add(url)

        print(f"  Scraping: {url}")
        page_data = scrape_page(url)

        if page_data and page_data["content"]:
            pages.append(page_data)
            print(f"    ✓ {page_data['title']}")

            for link in page_data["internal_links"]:
                link = link.rstrip("/")
                if link not in visited and link not in queue:
                    queue.append(link)
        else:
            print("    ✗ Skipped")

        time.sleep(1)

    return pages


def build_knowledge_base(pages: List[Dict]) -> Dict:
    knowledge = {
        "company": "ODN Systems",
        "website": BASE_URL,
        "pages": [],
        "full_text": "",
        "total_pages": len(pages)
    }

    all_text = []

    for page in pages:
        entry = {
            "url": page["url"],
            "title": page["title"],
            "content": page["content"]
        }
        knowledge["pages"].append(entry)
        all_text.append(f"{page['title']}\n{page['content']}")

    knowledge["full_text"] = "\n\n".join(all_text)

    return knowledge


def main():
    pages = crawl_website()

    if not pages:
        print("\n❌ No pages scraped.")
        return

    kb = build_knowledge_base(pages)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to {OUTPUT_FILE}")
    print(f"Pages scraped: {len(pages)}")


if __name__ == "__main__":
    main()