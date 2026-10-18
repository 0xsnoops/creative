from collections import deque


class Policy:
    """First in, first out: hits do not change the eviction order."""

    def __init__(self, capacity):
        self.queue = deque()

    def on_hit(self, key):
        pass

    def on_miss(self, key):
        self.queue.append(key)

    def evict(self):
        return self.queue.popleft()
