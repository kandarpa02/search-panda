import trafilatura
from ddgs import DDGS
from agents import function_tool

RESULTS = []


@function_tool
def search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return search results."""
    global RESULTS
    RESULTS = []

    with DDGS() as ddgs:
        response = ddgs.text(
            query,
            max_results=max_results,
        )

        for i, item in enumerate(response):
            RESULTS.append({
                "index": i,
                "title": item.get("title", ""),
                "url": item.get("href") or item.get("url", ""),
                "snippet": item.get("body", ""),
            })

    return RESULTS


@function_tool
def read_result(index: int) -> dict:
    """Read and extract the contents of a search result by index."""
    global RESULTS

    if index < 0 or index >= len(RESULTS):
        return {
            "error": "Invalid search result index."
        }

    result = RESULTS[index]

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



