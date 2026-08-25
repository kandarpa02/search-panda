"""Query planner — the brain of the agentic search loop.

The planner runs *before* any web request. It classifies intent,
decides whether the web is needed at all, decomposes complex queries
into focused sub-queries, and hints at high-value domains to search.

Deliberately LLM-free: fast heuristics keep latency near-zero and
ensure the agent stays snappy even on small local models.
"""

from __future__ import annotations

import re
from typing import Literal

from .data_representation import QueryPlan

# ---------------------------------------------------------------------------
# Time-sensitivity & Sports/Event signals
# ---------------------------------------------------------------------------

_TIME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(today|tonight|right now|currently|at the moment)\b",
        r"\b(this (week|month|year|morning|afternoon|evening))\b",
        r"\b(latest|recent|newest|just|breaking|live)\b",
        r"\b(yesterday|last (week|month|night))\b",
        r"\b(202[4-9]|203\d)\b",  # years 2024-2039
        r"\b(q[1-4] 20\d{2})\b",  # quarterly references
        r"\b(price|stock|rate|score|weather|forecast)\b",
        r"\b(fifa|world cup|olympics|super bowl|championship|tournament|uefa|premier league|nfl|nba)\b",
        r"\b(winner|champion|results|standings|who won|who is leading)\b",
    ]
]

# ---------------------------------------------------------------------------
# Intent keyword buckets
# ---------------------------------------------------------------------------

_CODE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(code|program|script|function|class|method|implement|debug|error|exception|traceback)\b",
        r"\b(python|javascript|typescript|rust|go|java|kotlin|swift|cpp|c\+\+|bash|sql)\b",
        r"\b(library|package|module|import|install|pip|npm|cargo|brew)\b",
        r"\b(api|endpoint|rest|graphql|webhook|http|json|yaml|xml)\b",
        r"\b(git|docker|kubernetes|terraform|ci|cd|deploy)\b",
        r"\b(how (do|to|can) (i|you|we) .*(code|build|write|create|implement))\b",
    ]
]

_MATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(calculate|compute|solve|evaluate|simplify|differentiate|integrate)\b",
        r"^\s*\d+\s*[\+\-\*/\^]\s*\d+\s*\??$",  # arithmetic expression
        r"\b(equation|formula|proof|theorem|derivative|integral|matrix|vector)\b",
        r"^(what is|what's)\s+\d+\s*[\+\-\*/\^]",  # "what is 12 * 7"
        r"\b(square root|factorial|logarithm|sine|cosine|tangent)\b",
    ]
]

_NEWS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(news|headlines|announcement|press release|update|report)\b",
        r"\b(who won|who is leading|what happened|what's happening|winner|champion|championship)\b",
        r"\b(election|war|conflict|summit|treaty|policy|law|regulation)\b",
        r"\b(president|prime minister|ceo|government|parliament|senate|congress)\b",
        r"\b(stock market|gdp|inflation|interest rate|unemployment)\b",
        r"\b(fifa|world cup|olympics|super bowl|tournament|uefa|nfl|nba)\b",
    ]
]

_OPINION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(best|worst|should i|recommend|opinion|think|feel|prefer|better|compare)\b",
        r"\b(pros|cons|advantages|disadvantages|trade.?off)\b",
        r"\b(vs\.?|versus|or|alternative)\b",
    ]
]

# ---------------------------------------------------------------------------
# No-search signals (strictly conversational greetings / self / trivial)
# ---------------------------------------------------------------------------

_NO_SEARCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(what is|what's)\s+\d+[\d\s\+\-\*/\^\.]*\??$",  # pure arithmetic
        r"^\s*(hello|hi|hey|howdy|good (morning|afternoon|evening))\s*[!\.]*$",
        r"^\s*(thank(s| you)|goodbye|bye|see you)\s*[!\.]*$",
        r"^\s*(who are you|what are you|tell me about yourself)\s*\??$",
    ]
]

# ---------------------------------------------------------------------------
# Site routing — domain hints per intent
# ---------------------------------------------------------------------------

_SITE_HINTS: dict[str, list[str]] = {
    "code": [
        "stackoverflow.com",
        "github.com",
        "docs.python.org",
        "developer.mozilla.org",
        "realpython.com",
    ],
    "news": [
        "reuters.com",
        "bbc.com",
        "apnews.com",
        "bloomberg.com",
        "techcrunch.com",
    ],
    "factual": [
        "wikipedia.org",
        "britannica.com",
    ],
    "math": [
        "wolframalpha.com",
        "math.stackexchange.com",
        "khanacademy.org",
    ],
    "opinion": [],
    "conversational": [],
}

# ---------------------------------------------------------------------------
# Query decomposition & cleanup heuristics
# ---------------------------------------------------------------------------

_MULTI_PART_SPLITTERS = re.compile(
    r"\band\b|\balso\b|\bas well as\b|\bmoreover\b|\bfurthermore\b|\bplus\b",
    re.IGNORECASE,
)

_SEARCH_PREFIX_CLEANER = re.compile(
    r"^(search(\s+the\s+web)?(\s+for|\s+and|\s+to)?|google|look\s+up|find(\s+out)?|tell\s+me(\s+about)?|can\s+you\s+(search(\s+for)?|find(\s+out)?|tell\s+me(\s+about)?)|about)\s+",
    re.IGNORECASE,
)


def clean_query_text(query: str) -> str:
    """Strip conversational search meta-commands like 'search and tell me who won'."""
    q = query.strip()
    # Remove leading command phrases iteratively
    for _ in range(2):
        cleaned = _SEARCH_PREFIX_CLEANER.sub("", q).strip()
        if cleaned and len(cleaned) >= 3:
            q = cleaned
        else:
            break
    return q


def _split_query(query: str) -> list[str]:
    """Return focused sub-queries for compound questions."""
    clean = clean_query_text(query)
    parts = _MULTI_PART_SPLITTERS.split(clean)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 8]
    if len(parts) > 1:
        return parts[:3]  # cap at 3 parallel searches to stay polite
    return [clean or query]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def needs_web_search(query: str, web_mode: str = "on") -> bool:
    """Return True when the query should search the web.

    web_mode:
      - "off": never search
      - "on": search by default (search engine behavior) unless pure greeting/math
      - "auto": search only when heuristics flag it
    """
    mode = (web_mode or "on").lower()
    if mode == "off":
        return False

    q = query.strip()

    # Explicit no-search checks (greetings, simple arithmetic, self-intro)
    for pat in _NO_SEARCH_PATTERNS:
        if pat.search(q):
            return False

    # Short pure arithmetic
    for pat in _MATH_PATTERNS:
        if pat.search(q) and any(op in q for op in ["+", "-", "*", "/", "^"]):
            return False

    # Conversational 1-word greetings
    if len(q.split()) <= 1 and q.lower() in {"hi", "hello", "hey", "yo", "sup"}:
        return False

    if mode == "on":
        # Search-first: any real question or query searches the web
        return True

    # "auto" mode: check if time-sensitive or news or factual
    if is_time_sensitive(q):
        return True

    for pat in _NEWS_PATTERNS:
        if pat.search(q):
            return True

    # Pure definitions like "explain recursion" can be answered from weights in auto mode
    if re.search(r"\b(explain|define)\b.*\b(recursion|concept)\b", q, re.IGNORECASE):
        return False

    return True


def classify_intent(
    query: str,
) -> Literal["factual", "news", "code", "math", "opinion", "conversational"]:
    """Classify query intent using keyword heuristics."""
    q = query.strip()

    if len(q.split()) <= 2 and not any(
        pat.search(q) for pat in _CODE_PATTERNS + _NEWS_PATTERNS + _MATH_PATTERNS
    ):
        return "conversational"

    for pat in _MATH_PATTERNS:
        if pat.search(q):
            return "math"

    for pat in _NEWS_PATTERNS:
        if pat.search(q):
            return "news"

    for pat in _CODE_PATTERNS:
        if pat.search(q):
            return "code"

    for pat in _OPINION_PATTERNS:
        if pat.search(q):
            return "opinion"

    return "factual"


def is_time_sensitive(query: str) -> bool:
    """Return True when the query asks about recent or live information."""
    return any(pat.search(query) for pat in _TIME_PATTERNS)


def rewrite_query(query: str, intent: str, time_sensitive: bool) -> str:
    """Return an improved search query for DuckDuckGo."""
    q = clean_query_text(query).strip()

    if intent == "news" or time_sensitive:
        import datetime

        year = datetime.date.today().year
        # If year not mentioned at all and query doesn't specify a future/past year
        if not re.search(r"\b20\d{2}\b", q):
            q = f"{q} {year}"

    elif intent == "code":
        if not re.search(r"\b(how (to|do|can)|tutorial|example|fix|error)\b", q, re.IGNORECASE):
            q = f"how to {q}"

    return q


def plan(query: str, web_mode: str = "on") -> QueryPlan:
    """Build a QueryPlan for the given user query."""
    intent = classify_intent(query)
    time_sensitive = is_time_sensitive(query)
    web_needed = needs_web_search(query, web_mode=web_mode)

    if intent == "math" and any(op in query for op in ["+", "-", "*", "/", "^"]):
        web_needed = False

    if intent == "conversational" and len(query.split()) <= 2:
        web_needed = False

    sub_queries: list[str] = []
    if web_needed:
        raw_parts = _split_query(query)
        sub_queries = [
            rewrite_query(part, intent, time_sensitive) for part in raw_parts
        ]

    return QueryPlan(
        intent=intent,
        needs_search=web_needed,
        sub_queries=sub_queries,
        site_hints=_SITE_HINTS.get(intent, []),
        time_sensitive=time_sensitive,
    )
