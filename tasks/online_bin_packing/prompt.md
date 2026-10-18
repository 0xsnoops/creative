# Online bin packing heuristic

Write a Python heuristic for **online bin packing**.

Items arrive one at a time and must each be placed in a bin immediately, without
knowing future items. Every bin has capacity 100. Item sizes are integers in
[1, 100], drawn independently from a Weibull distribution (shape k=3, scale 45),
rounded and clipped to [1, 100]. Each test instance has 5,000 items.

Your goal is to use as few bins as possible. Your heuristic is scored by the
average percentage of excess bins over the lower bound `ceil(sum(items) / 100)`
across several instances; lower is better.

## Interface

Define a function:

```python
def priority(item: int, capacities: list[int]) -> list[float]:
    ...
```

- `item` is the size of the arriving item.
- `capacities` holds the remaining capacity of every currently open bin that can
  fit the item, in the order the bins were opened. The last entry is always 100
  and represents opening a new, empty bin.
- Return one score per entry of `capacities`. The item goes into the bin with the
  highest score (ties go to the earliest bin).

The function is called once per item. Module-level state (e.g. statistics of
past items) is allowed; your module is freshly loaded for each instance. You may
use the Python 3.11 standard library and numpy. The full evaluation (25,000 calls) must
finish within 60 seconds.

Reply with your complete solution in a single Python code block.
