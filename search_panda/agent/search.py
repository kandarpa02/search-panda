import trafilatura
from ddgs import DDGS

_SEARCH_RESULTS = []


def search(query, max_results=5):
    global _SEARCH_RESULTS

    _SEARCH_RESULTS = []

    with DDGS() as ddgs:
        response = ddgs.text(query, max_results=max_results)

        for i, item in enumerate(response):
            _SEARCH_RESULTS.append(
                {
                    "index": i,
                    "title": item.get("title", ""),
                    "url": item.get("href") or item.get("url", ""),
                    "snippet": item.get("body", "")
                }
            )

    return _SEARCH_RESULTS


def read_result(index):
    global _SEARCH_RESULTS

    if index < 0 or index >= len(_SEARCH_RESULTS):
        return {
            "error": "Invalid search result index."
        }

    result = _SEARCH_RESULTS[index]

    if not result.get("url"):
        return {
            "error": "No URL available for this result."
        }

    downloaded = trafilatura.fetch_url(result["url"])

    if downloaded is None:
        return {
            "error": "Failed to fetch webpage."
        }

    text = trafilatura.extract(
        downloaded,
        include_links=False,
        include_images=False
    )

    if text is None:
        return {
            "error": "Failed to extract webpage."
        }

    return {
        "title": result["title"],
        "url": result["url"],
        "content": text
    }