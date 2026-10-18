"""Exa search client with a disk cache, charged against the same budget ledger."""
import time

import httpx

from .cache import request_key
from .llm import require_env

EXA = "https://api.exa.ai/search"


class Exa:
    def __init__(self, cache, ledger, cost_estimate_usd, tracer=None):
        self.cache, self.ledger, self.cost_estimate, self.tracer = cache, ledger, cost_estimate_usd, tracer

    def search(self, query, num_results=8, published_before=None):
        """Return a list of {url, title, published, highlights}."""
        request = {"query": query, "type": "auto", "numResults": num_results,
                   "contents": {"highlights": {"maxCharacters": 1200}}}
        if published_before:
            request["endPublishedDate"] = f"{published_before}T00:00:00Z"
        key = request_key(request)
        cached = bool(entry := self.cache.get(key))
        if cached:
            self.ledger.charge_cached(self.cost_estimate, entry["cost_usd"])
        else:
            self.ledger.reserve(self.cost_estimate)
            cost = 0.0
            try:
                response = self._post(request)
                cost = (response.get("costDollars") or {}).get("total", self.cost_estimate)
            finally:
                self.ledger.settle(self.cost_estimate, cost)
            entry = self.cache.put(key, {"request": request, "response": response, "cost_usd": cost,
                                         "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if self.tracer:
            self.tracer.log({"kind": "exa", "stage": "search", "model": "exa", "request": request,
                             "response": entry["response"], "cost_usd": entry["cost_usd"],
                             "cache": "hit" if cached else "miss", "new_cost_usd": 0.0 if cached else entry["cost_usd"]})
        return [{"url": r["url"], "title": r.get("title") or "", "published": r.get("publishedDate"),
                 "highlights": r.get("highlights") or []} for r in entry["response"].get("results", [])]

    def _post(self, request, attempts=3):
        headers = {"x-api-key": require_env("EXA_API_KEY")}
        error = None
        for attempt in range(attempts):
            try:
                r = httpx.post(EXA, headers=headers, json=request, timeout=120)
                if r.status_code == 200:
                    return r.json()
                error = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 402, 403):
                    break
            except httpx.HTTPError as e:
                error = f"{type(e).__name__}: {e}"
            time.sleep(3 * 2 ** attempt)
        raise RuntimeError(f"exa: {error}")
