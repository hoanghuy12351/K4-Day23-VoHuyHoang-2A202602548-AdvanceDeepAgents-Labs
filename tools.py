"""Host-side source tools for the Deep Research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
from xml.etree import ElementTree
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
RETRY_STATUSES = {429, 500, 502, 503, 504}
_arxiv_lock = threading.Lock()
_last_arxiv_call = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Retry temporary failures with capped backoff and optional Retry-After."""
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError("attempts must be positive; base and cap must be non-negative")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0.0, float(exc.retry_after)))
            else:
                delay = min(cap, base * 2**attempt + random.uniform(0, base))
            time.sleep(delay)


def _request(method, url, **kwargs):
    """Convert only temporary HTTP and transport failures into retryable errors."""
    try:
        response = httpx.request(method, url, timeout=30, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(str(exc)) from exc
    if response.status_code in RETRY_STATUSES:
        header = response.headers.get("Retry-After", "")
        try:
            retry_after = float(header)
        except ValueError:
            retry_after = None
        raise RetryableError(f"HTTP {response.status_code}", retry_after=retry_after)
    response.raise_for_status()
    return response


def _error(exc, secret=""):
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
        message = message.replace(quote(secret, safe=""), "[REDACTED]")
    return f"ERROR: {message}"


def _clean(text, limit=600):
    return " ".join(str(text or "").split())[:limit]


def _is_rate_limited(metadata, content):
    phrases = ("rate limit exceeded", "rate_limit_exceeded", "quota exceeded",
               "too many requests", "rate limited")
    if any(phrase in content.lower() for phrase in phrases):
        return True
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            name = re.sub(r"[^a-z]", "", key.lower())
            if ("ratelimit" in name or "quota" in name) and value is True:
                return True
            if _is_rate_limited(value, ""):
                return True
    elif isinstance(metadata, (list, tuple)):
        return any(_is_rate_limited(item, "") for item in metadata)
    elif isinstance(metadata, str):
        return any(phrase in metadata.lower() for phrase in phrases)
    return False


def _exa_call(name, arguments):
    key = os.getenv("EXA_API_KEY", "").strip()
    params = {"exaApiKey": key} if key else None
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}

    def call():
        response = _request("POST", EXA_URL, params=params, json=payload,
                            headers={"Accept": "application/json, text/event-stream",
                                     "Content-Type": "application/json"})
        events = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    events.append(json.loads(data))
        if not events:
            events = [response.json()]
        message = next((event for event in reversed(events)
                        if isinstance(event, dict) and ("result" in event or "error" in event)), events[-1])
        if not isinstance(message, dict):
            raise ValueError("invalid Exa MCP response")
        if message.get("error"):
            raise ValueError(f"Exa MCP error: {message['error']}")
        result = message.get("result", {})
        if not isinstance(result, dict):
            raise ValueError("invalid Exa MCP result")
        content = result.get("content", [])
        text = "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
        if _is_rate_limited(result.get("_meta", {}), text):
            raise RetryableError("Exa rate limit")
        if result.get("isError"):
            raise ValueError(f"Exa tool error: {text[:300]}")
        return text.strip()

    try:
        # Anonymous Exa access can be rate limited for long periods. Return
        # sooner so the researcher can try another source family.
        result = with_retry(call, attempts=6 if key else 3, cap=60 if key else 10)
        if key:
            result = result.replace(key, "[REDACTED]").replace(quote(key, safe=""), "[REDACTED]")
        return result
    except Exception as exc:
        return _error(exc, key)


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _last_arxiv_call
    try:
        terms = [term for term in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query, re.UNICODE)
                 if term.upper() not in {"AND", "OR", "NOT"}]
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{term}" for term in terms),
                  "sortBy": "submittedDate", "sortOrder": "descending",
                  "max_results": min(30, max(1, int(max_results))), "start": 0}

        def search():
            global _last_arxiv_call
            with _arxiv_lock:
                time.sleep(max(0.0, 3.0 - (time.monotonic() - _last_arxiv_call)))
                _last_arxiv_call = time.monotonic()
            return _request("GET", ARXIV_URL, params=params)

        response = with_retry(search, attempts=6, cap=60)
        root = ElementTree.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            raw_url = entry.findtext("atom:id", default="", namespaces=ns)
            paper_id = re.sub(r"v\d+$", "", raw_url.rsplit("/abs/", 1)[-1].strip())
            if not paper_id:
                continue
            records.append({
                "id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                "published": entry.findtext("atom:published", default="", namespaces=ns)[:10],
                "title": _clean(entry.findtext("atom:title", default="", namespaces=ns)),
                "summary": _clean(entry.findtext("atom:summary", default="", namespaces=ns)),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


def _paper_record(item):
    paper = item.get("paper", item) if isinstance(item, dict) else None
    if not isinstance(paper, dict) or not paper.get("id"):
        return None
    paper_id = str(paper["id"])
    try:
        upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
    except (TypeError, ValueError):
        upvotes = 0
    try:
        stars = int(paper.get("githubStars") or item.get("githubStars") or 0)
    except (TypeError, ValueError):
        stars = 0
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
        "title": _clean(paper.get("title") or item.get("title")),
        "summary": _clean(paper.get("ai_summary") or item.get("ai_summary") or
                          paper.get("summary") or item.get("summary")),
        "upvotes": upvotes,
        "github": paper.get("githubRepo") or item.get("githubRepo") or "",
        "stars": stars,
    }


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": min(100, max(1, int(limit)))}
        if date:
            params["date"] = date
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        records = [record for item in response.json() if (record := _paper_record(item))]
        if keyword:
            needle = keyword.casefold()
            records = [record for record in records
                       if needle in (record["title"] + " " + record["summary"]).casefold()]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not query.strip():
            return "NO RESULTS"
        params = {"q": query.strip(), "limit": min(50, max(1, int(limit)))}
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params))
        records = [record for item in response.json() if (record := _paper_record(item))]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not query.strip():
            return "NO RESULTS"
        text = _exa_call("web_search_exa", {"query": query.strip(),
                         "objective": objective.strip() or f"Find reliable sources about {query.strip()}",
                         "numResults": min(10, max(1, int(num_results)))})
        return text[:18000] if text and not text.startswith("ERROR:") else (text or "NO RESULTS")
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY", "").strip())


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        if not url.startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        text = _exa_call("web_fetch_exa", {"urls": [url]})
        return text[:12000] if text and not text.startswith("ERROR:") else (text or "NO RESULTS")
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY", "").strip())


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
