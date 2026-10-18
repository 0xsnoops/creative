"""OpenRouter chat client with a disk cache and a hard budget cap."""
import json
import os
import threading
import time
from pathlib import Path

import httpx

from .cache import JsonlCache, request_key

OPENROUTER = "https://openrouter.ai/api/v1"


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Estimated spend. Every uncached call reserves its worst-case cost up front, so
    concurrent calls cannot push the total past the cap."""

    def __init__(self, cap_usd, spent_usd):
        self.cap, self.spent, self.reserved = cap_usd, spent_usd, 0.0
        self.spent_this_run = 0.0
        self.lock = threading.Lock()

    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + amount > self.cap:
                raise BudgetExceeded(f"budget ${self.cap:.2f} reached (spent ${self.spent:.4f}, "
                                     f"in flight ${self.reserved:.4f}, next call up to ${amount:.4f})")
            self.reserved += amount

    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved
            self.spent += actual
            self.spent_this_run += actual


def load_model_catalog(cache_dir):
    """OpenRouter's /models listing, saved on first use so pricing stays fixed across reruns."""
    path = Path(cache_dir) / "openrouter_models.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(f"{OPENROUTER}/models", timeout=60)
        response.raise_for_status()
        path.write_text(response.text)
    return {m["id"]: m for m in json.loads(path.read_text())["data"]}


class LLM:
    def __init__(self, cache_dir, ledger):
        self.cache = JsonlCache(Path(cache_dir) / "llm.jsonl")
        self.catalog = load_model_catalog(cache_dir)
        self.ledger = ledger

    def check_models(self, models):
        missing = [m for m in models if m not in self.catalog]
        if missing:
            raise ValueError(f"not on OpenRouter /models: {missing}")

    def chat(self, model, messages, temperature=1.0, max_tokens=8000, sample_idx=0):
        """Return the cache entry {request, response, cost_usd, cached}. sample_idx only
        distinguishes repeated samples of the same request in the cache key."""
        info = self.catalog[model]
        max_tokens = min(max_tokens, info["top_provider"].get("max_completion_tokens") or max_tokens)
        request = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "sample_idx": sample_idx}
        key = request_key(request)
        if hit := self.cache.get(key):
            return {**hit, "cached": True}

        price_in, price_out = float(info["pricing"]["prompt"]), float(info["pricing"]["completion"])
        prompt_tokens_estimate = sum(len(m["content"]) for m in messages) / 3
        worst_case = prompt_tokens_estimate * price_in + max_tokens * price_out
        self.ledger.reserve(worst_case)
        cost = 0.0
        try:
            body = {k: v for k, v in request.items() if k != "sample_idx"}
            response = self._post(body)
            usage = response.get("usage") or {}
            cost = usage.get("cost")
            if cost is None:
                cost = (usage.get("prompt_tokens", 0) * price_in
                        + usage.get("completion_tokens", 0) * price_out)
        finally:
            self.ledger.settle(worst_case, cost)
        entry = self.cache.put(key, {"request": request, "response": response, "cost_usd": cost,
                                     "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return {**entry, "cached": False}

    def _post(self, body, attempts=4):
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        error = None
        for attempt in range(attempts):
            try:
                r = httpx.post(f"{OPENROUTER}/chat/completions", headers=headers,
                               json={**body, "usage": {"include": True}}, timeout=900)
                data = r.json()
                if r.status_code == 200 and data.get("choices"):
                    return data
                error = f"HTTP {r.status_code}: {json.dumps(data.get('error', data))[:500]}"
                if r.status_code in (400, 401, 402, 403, 404):
                    break  # retrying will not help
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                error = f"{type(e).__name__}: {e}"
            time.sleep(5 * 2 ** attempt)
        raise RuntimeError(f"{body['model']}: {error}")


def message_text(entry):
    return entry["response"]["choices"][0]["message"].get("content") or ""


def finish_reason(entry):
    return entry["response"]["choices"][0].get("finish_reason")
