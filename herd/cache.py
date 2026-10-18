"""Append-only JSONL cache keyed by a hash of the request."""
import hashlib
import json
import threading
from pathlib import Path


def request_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


class JsonlCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.entries = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, entry):
        entry = {"key": key, **entry}
        with self.lock:
            self.entries[key] = entry
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry

    def total_cost(self):
        return sum(e.get("cost_usd", 0.0) for e in self.entries.values())
