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
ANTHROPIC = "https://api.anthropic.com/v1/messages"
# DIVERGE_PROVIDER=anthropic sends every call to the Anthropic API with ANTHROPIC_API_KEY and never
# to OpenRouter. Model slugs stay "anthropic/claude-..." so cached replies and prices still apply.


def provider():
    return os.environ.get("DIVERGE_PROVIDER", "openrouter").strip().lower()


ANTHROPIC_DEFAULTS = {"samplers": ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"],
                      "implementer": "anthropic/claude-sonnet-5.5", "extractor": "anthropic/claude-haiku-5.5",
                      "labeler": "anthropic/claude-opus-5.5"}


def anthropic_only_models(models):
    """The configured models with every non-Anthropic role replaced by a Claude model."""
    out = dict(models)
    for role, default in ANTHROPIC_DEFAULTS.items():
        value = models.get(role)
        if isinstance(value, list):
            kept = [m for m in value if m.startswith("anthropic/")]
            out[role] = kept or default
        elif value is None or not str(value).startswith("anthropic/"):
            out[role] = default
    return out


def anthropic_model_id(slug):
    """'anthropic/claude-haiku-5.5' -> 'claude-haiku-5-5'; None for a non-Anthropic slug."""
    if not slug.startswith("anthropic/"):
        return None
    return slug.split("/", 1)[1].replace(".", "-")
COMPLETION_MARGIN = 1.25
MIN_SHRUNK_TOKENS = 4000  # implementer replies use ~2k tokens; smaller budgets are refused as before


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
        self.recorded = 0.0        # original cost of every response used, cached or not
        self.overruns = 0          # calls whose settled cost exceeded their reservation
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

    def headroom(self):
        """What a new call may still reserve under both caps (inf when uncapped)."""
        with self.lock:
            room = self.global_cap - self.total_spent - self.reserved
            if self.run_cap is not None:
                room = min(room, self.run_cap - self.spent_this_run - self.reserved)
            return room

    def reserve(self, amount):
        with self.lock:
            self._check(amount, new_money=True)
            self.reserved += amount

    def settle(self, reserved, actual):
        with self.lock:
            self.reserved -= reserved
            if actual > reserved:
                self.overruns += 1
            self.new_spend += actual
            self.spent_this_run += actual
            self.recorded += actual

    def charge_cached(self, worst_case, cost):
        """Bill a cache hit to this run only; it is already part of the global total."""
        with self.lock:
            if self.bill_cached:
                self._check(worst_case, new_money=False)
                self.spent_this_run += cost
            self.recorded += cost


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
    def __init__(self, cache, catalog, ledger, tracer=None):
        self.cache, self.catalog, self.ledger, self.tracer = cache, catalog, ledger, tracer

    def _trace(self, request, entry, cached, latency):
        if not self.tracer:
            return
        response = entry.get("response") or {}
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        self.tracer.log({"kind": "llm", "stage": str(request["sample_idx"]), "model": request["model"],
                         "provider": response.get("provider"), "request_id": response.get("id"),
                         "messages": request["messages"],
                         "params_sent": {k: request[k] for k in ("temperature", "max_tokens")},
                         "content": message.get("content"),
                         "reasoning": message.get("reasoning") or message.get("reasoning_content"),
                         "finish_reason": choice.get("finish_reason"), "usage": usage,
                         "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                         "cost_usd": entry.get("cost_usd"), "cache": "hit" if cached else "miss",
                         "new_cost_usd": 0.0 if cached else entry.get("cost_usd"), "latency_s": latency})

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
        # Some providers return more completion tokens than max_tokens (seen in the model review:
        # 12,634 against 10,000), so the completion side carries a 25% margin.
        worst_case = prompt_tokens_estimate * price_in + max_tokens * price_out * COMPLETION_MARGIN
        if hit := self.cache.get(key):
            self.ledger.charge_cached(worst_case, hit["cost_usd"])
            self._trace(request, hit, True, 0.0)
            return {**hit, "cached": True}
        # A full-size reservation that no longer fits the budget used to end the run early (bench
        # operators stopped at about a third of their cap). Shrink max_tokens to what the budget can
        # still pay for, never below MIN_SHRUNK_TOKENS; the reservation stays a hard upper bound.
        room = self.ledger.headroom()
        if worst_case > room:
            fit = int((room - prompt_tokens_estimate * price_in) / (price_out * COMPLETION_MARGIN)) if price_out else 0
            if fit >= MIN_SHRUNK_TOKENS:
                max_tokens = fit
                request = {**request, "max_tokens": max_tokens}
                key = request_key(request)
                worst_case = prompt_tokens_estimate * price_in + max_tokens * price_out * COMPLETION_MARGIN
                if hit := self.cache.get(key):
                    self.ledger.charge_cached(worst_case, hit["cost_usd"])
                    self._trace(request, hit, True, 0.0)
                    return {**hit, "cached": True}
        started = time.time()

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
        self._trace(request, entry, False, round(time.time() - started, 2))
        return {**entry, "cached": False}

    def _post(self, body, attempts=4):
        if provider() == "anthropic":
            return self._post_anthropic(body, attempts)
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


def _anthropic_post(self, body, attempts=4):
    """The same call through the Anthropic Messages API, returned in the OpenRouter response shape
    (choices[0].message.content, finish_reason, usage with cost) so the rest of DIVERGE is unchanged."""
    model = anthropic_model_id(body["model"])
    if model is None:
        raise RuntimeError(f"{body['model']}: DIVERGE_PROVIDER=anthropic only serves anthropic/* models "
                           "(OpenRouter is disabled); configure a Claude model instead")
    system = "\n\n".join(m["content"] for m in body["messages"] if m["role"] == "system")
    request = {"model": model, "max_tokens": body["max_tokens"], "temperature": min(body.get("temperature", 1.0), 1.0),
               "messages": [m for m in body["messages"] if m["role"] != "system"]}
    if system:
        request["system"] = system
    headers = {"x-api-key": require_env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    info = self.catalog.get(body["model"], {}).get("pricing", {})
    error = None
    for attempt in range(attempts):
        try:
            r = httpx.post(ANTHROPIC, headers=headers, json=request, timeout=900)
            data = r.json()
            if r.status_code == 200 and data.get("content") is not None:
                text = "".join(c.get("text", "") for c in data["content"] if c.get("type") == "text")
                usage = data.get("usage") or {}
                tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                cost = tokens_in * float(info.get("prompt", 0)) + tokens_out * float(info.get("completion", 0))
                return {"choices": [{"message": {"content": text},
                                     "finish_reason": "length" if data.get("stop_reason") == "max_tokens" else "stop"}],
                        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "cost": cost},
                        "provider": "anthropic", "model": data.get("model")}
            error = f"HTTP {r.status_code}: {json.dumps(data.get('error', data))[:500]}"
            if r.status_code in (400, 401, 403, 404):
                break
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            error = f"{type(e).__name__}: {e}"
        time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"{body['model']} (anthropic): {error}")


LLM._post_anthropic = _anthropic_post


def message_text(entry):
    return entry["response"]["choices"][0]["message"].get("content") or ""


def finish_reason(entry):
    return entry["response"]["choices"][0].get("finish_reason")


def parse_json(text):
    """The JSON object in a model reply, with or without a ```json fence."""
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"):text.rfind("}") + 1]
    return json.loads(candidate)


def ask_json(llm, model, prompt, max_tokens=16000, attempts=3, key=None, temperature=0.0, required=()):
    """Ask for a JSON object with the required top-level keys; a reply that does not parse or
    lacks a key is re-asked under a new cache key. key separates otherwise identical requests
    (e.g. benchmark repeats) in the cache."""
    for attempt in range(attempts):
        entry = llm.chat(model, [{"role": "user", "content": prompt}], temperature=temperature,
                         max_tokens=max_tokens, sample_idx=attempt if key is None else f"{key}:{attempt}")
        try:
            reply = parse_json(message_text(entry))
        except ValueError:
            continue
        if isinstance(reply, dict) and all(k in reply for k in required):
            return reply
    raise ValueError(f"{model} did not return a JSON object with keys {list(required)} after {attempts} attempts")
