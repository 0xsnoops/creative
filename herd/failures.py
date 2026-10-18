"""Failure sets: which dev cases a program does worst on, so two programs can be compared by
where they fail rather than how they read. Used by experiment E1 (detector validity) and by
anything that needs "do these two attempts fail on the same cases?".

Detail shapes (from the task evaluators' dev results):
- bin packing: {"instances": [{"seed", "excess_pct", ...}]}, ids "seed:N";
- cache eviction: {"runs": [{"trace", "cache_size", "miss_ratio", ...}]}, ids "trace@size".
Higher score = worse in both."""
import math


def _cases(detail):
    """[(id, score)] for a dev detail; [] for a missing or unrecognised one. Entries without a
    finite numeric score are skipped."""
    if not isinstance(detail, dict):
        return []
    if isinstance(detail.get("instances"), list):
        rows, ident, score = detail["instances"], lambda r: f"seed:{r['seed']}", "excess_pct"
    elif isinstance(detail.get("runs"), list):
        rows, ident, score = detail["runs"], lambda r: f"{r['trace']}@{r['cache_size']}", "miss_ratio"
    else:
        return []
    out = []
    for r in rows:
        try:
            value = float(r[score])
            case = ident(r)
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append((case, value))
    return out


def failure_set(detail, k=2):
    """The k worst cases of a dev detail (largest excess_pct / miss_ratio) as a frozenset of
    ids; ties are broken by id (smaller id first). Empty for missing or invalid detail."""
    if not isinstance(k, int) or k <= 0:
        return frozenset()
    worst = sorted(_cases(detail), key=lambda c: (-c[1], c[0]))
    return frozenset(case for case, _ in worst[:k])


def relative_detail(detail, baseline_detail):
    """The detail with each case's score replaced by its gap to the baseline on the same case
    (score - baseline score); cases the baseline lacks are dropped. Feed it to failure_set for
    failure sets relative to a baseline."""
    base = dict(_cases(baseline_detail))
    if not isinstance(detail, dict):
        return None
    key, score = ("instances", "excess_pct") if isinstance(detail.get("instances"), list) else ("runs", "miss_ratio")
    if not isinstance(detail.get(key), list):
        return None
    ids = dict(_cases(detail))
    rows = []
    for r in detail[key]:
        case = f"seed:{r.get('seed')}" if key == "instances" else f"{r.get('trace')}@{r.get('cache_size')}"
        if case in ids and case in base:
            rows.append({**r, score: ids[case] - base[case]})
    return {key: rows}


def jaccard(a, b):
    """|a & b| / |a | b|; 1.0 when both are empty."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def auc(scores, labels):
    """Mann-Whitney AUC: the chance a random positive scores above a random negative, ties
    counted as 0.5. None when either class is empty."""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):  # average ranks over runs of equal scores
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        mid = (i + 1 + j) / 2
        rank_sum += mid * sum(1 for _, y in pairs[i:j] if y)
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
