from collections import OrderedDict


class Policy:
    """Least recently used."""

    def __init__(self, capacity):
        self.order = OrderedDict()

    def on_hit(self, key):
        self.order.move_to_end(key)

    def on_miss(self, key):
        self.order[key] = None

    def evict(self):
        return self.order.popitem(last=False)[0]
