# Cache eviction policy

Write a Python **cache eviction policy**.

A cache holds at most `capacity` keys. Requests for keys arrive one at a time. If
the key is in the cache it is a hit; otherwise it is a miss and the key is
inserted, first evicting a key chosen by your policy if the cache is full. Your
policy only sees requests as they happen; it never sees future requests.

Your policy is scored by its average miss ratio (misses / requests) over several
synthetic request traces and cache sizes; lower is better. The traces mix
skewed, Zipf-like key popularity with periodic sequential scans and with a large
share of keys that are requested only once. Cache sizes are 1% and 10% of the
number of unique keys in a trace; traces have roughly 50,000 requests.

## Interface

Define a class:

```python
class Policy:
    def __init__(self, capacity: int): ...
    def on_hit(self, key: int) -> None: ...   # key was requested and is in the cache
    def on_miss(self, key: int) -> None: ...  # key was requested, missed, and has just been inserted
    def evict(self) -> int: ...               # cache is full: return the key to remove
```

On a miss with a full cache, the simulator calls `evict()` first, removes the key
it returns (which must currently be in the cache), then inserts the new key and
calls `on_miss(key)`. Keys are integers. A fresh `Policy` is created for every
trace and cache size.

You may use the Python 3.11 standard library and numpy. The full evaluation (about
310,000 requests in total) must finish within 60 seconds.

Reply with your complete solution in a single Python code block.
