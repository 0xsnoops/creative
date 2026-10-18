"""OpenRouter chat client with a disk cache and a hard budget cap."""
import json
import os
import re
import threading
import time
from pathlib import Path

import httpx

from .cache import request_key

OPENROUTER = "https://openrouter.ai/api/v1"


def require_env(name):
    if not os.environ.get(name):
        raise RuntimeError(f"{name} is not set; add it to .env")
    return os.environ[name]


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Estimated spend, checked against two caps:
    - global_cap: total money ever sent to the APIs (everything already in the cache plus new
      calls). Cache hits are already counted in the cache total, so they never count again.
    - run_cap: what this run (one tool call, one benchmark operator) may spend, measured from
      zero. With bill_cached, cache hits count toward it at their recorded cost, so a rerun of a
      budgeted run stops where the original did.
    Every new call reserves its worst-case cost up front, so concurrent calls cannot overshoot."""

    def __init__(self, global_cap_usd, global_spent_usd, run_cap_usd=None, bill_cached=False):
        self.global_cap, self.global_spent = global_cap_usd, global_spent_usd
        self.run_cap, self.bill_cached = run_cap_usd, bill_cached
        self.reserved = 0.0
        self.spent_this_run = 0.0  # this run's spend, including billed cache hits
        self.new_spend = 0.0       # money actually sent to the APIs in this run
        self.lock = threading.Lock()

    @property
    def total_spent(self):
        return self.global_spent + self.new_spend

    def _check(self, amount, new_money):
        if new_money and self.total_spent + self.reserved + amount > self.global_cap:
            raise BudgetExceeded(f"global budget ${self.global_cap:.2f} reached (spent ${self.total_spent:.4f}, "
                                 f"in flight ${self.reserved:.4f}, next call up to ${amount:.4f})")
        if self.run_cap is not None and self.spent_this_run + self.reserved + amount > self.run_cap:
            raise BudgetExceeded(f"run budget ${self.run_cap:.2f} reached (spent ${self.spent_this_run:.4f}, "
                                 f"in flight ${self.reserved:.4f}, next call up to ${amount:.4f})")

    def reserve(self, amount):
        with self.lock:
            self._check(amount, new_money=True)
            self.reserved += amount

    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved
            self.new_spend += actual
            self.spent_this_run += actual

    def charge_cached(self, worst_case, cost):
        """Bill a cache hit to this run only; it is already part of the global total."""
        if self.bill_cached:
            with self.lock:
                self._check(worst_case, new_money=False)
                self.spent_this_run += cost


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
    def __init__(self, cache, catalog, ledger):
        self.cache, self.catalog, self.ledger = cache, catalog, ledger

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
        price_in, price_out = float(info["pricing"]["prompt"]), float(info["pricing"]["completion"])
        prompt_tokens_estimate = sum(len(m["content"]) for m in messages) / 3
        worst_case = prompt_tokens_estimate * price_in + max_tokens * price_out
        if hit := self.cache.get(key):
            self.ledger.charge_cached(worst_case, hit["cost_usd"])
            return {**hit, "cached": True}

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
        headers = {"Authorization": f"Bearer {require_env('OPENROUTER_API_KEY')}"}
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


def parse_json(text):
    """The JSON object in a model reply, with or without a ```json fence."""
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
    return json.loads(candidate)


def ask_json(llm, model, prompt, max_tokens=16000, attempts=3, key=None, temperature=0.0):
    """Ask for a JSON object; a reply that does not parse is re-asked under a new cache key.
    key separates otherwise identical requests (e.g. benchmark repeats) in the cache."""
    for attempt in range(attempts):
        entry = llm.chat(model, [{"role": "user", "content": prompt}], temperature=temperature,
                         max_tokens=max_tokens, sample_idx=attempt if key is None else f"{key}:{attempt}")
        try:
            return parse_json(message_text(entry))
        except ValueError:
            continue
    raise ValueError(f"{model} did not return valid JSON after {attempts} attempts")
