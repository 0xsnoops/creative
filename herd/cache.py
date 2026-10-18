"""Append-only JSONL cache keyed by a hash of the request. Several processes (a CLI run and
the MCP server) may share a file: writes take an exclusive lock, and a damaged line is
skipped on load rather than breaking every later run."""
import fcntl
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
        self.skipped_lines = 0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    self.entries[entry["key"]] = entry
                except (ValueError, KeyError, TypeError):
                    self.skipped_lines += 1

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, entry):
        entry = {"key": key, **entry}
        with self.lock:
            self.entries[key] = entry
            with self.path.open("a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # other processes append to the same file
                f.write(json.dumps(entry) + "\n")
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        return entry

    def total_cost(self):
        return sum(e.get("cost_usd", 0.0) for e in self.entries.values())
