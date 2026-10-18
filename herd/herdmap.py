"""HERD MAP and BLIND SPOTS."""
import math
from collections import Counter

from .label import EMERGENT, UNLABELED


def shares(families):
    counted = [f for f in families if f != UNLABELED]
    counts = Counter(counted)
    return counts, {f: c / len(counted) for f, c in counts.items()} if counted else {}


def build_herd_map(records, labels, families):
    """Share of samples per family, overall and per model. Unlabeled samples are excluded from
    the shares and reported separately."""
    model_of = {r["sample_id"]: r["model"] for r in records}
    counts, overall = shares([l["family"] for l in labels.values()])
    rows = [{"family": f["family"], "count": counts.get(f["family"], 0),
             "share": round(overall.get(f["family"], 0.0), 4), "sourced": bool(f["source_urls"])}
            for f in families]
    rows.sort(key=lambda r: (-r["share"], r["family"]))
    per_model, per_model_counts = {}, {}
    for model in dict.fromkeys(model_of.values()):
        c, s = shares([l["family"] for sid, l in labels.items() if model_of[sid] == model])
        per_model[model] = {f: round(v, 4) for f, v in sorted(s.items(), key=lambda kv: -kv[1])}
        per_model_counts[model] = {"n": sum(c.values()), "counts": dict(c)}
    top3 = [r["family"] for r in rows[:3] if r["count"]]
    return {"n_samples": len(records), "n_labeled": sum(counts.values()),
            "n_unlabeled": sum(l["family"] == UNLABELED for l in labels.values()),
            "concentration_top3": round(sum(r["share"] for r in rows[:3]), 4),
            "top3": top3, "families": rows, "per_model": per_model, "per_model_counts": per_model_counts}


def share_upper_bound(k, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound on the true share after seeing k of n samples
    (assumes independent draws, which samples from several models only approximate)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):  # bisect for p with P(X <= k | n, p) = 1 - confidence
        p = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
        lo, hi = (p, hi) if cdf > 1 - confidence else (lo, p)
    return round(hi, 4)


def find_blind_spots(herd_map, families, max_share):
    """Sourced families whose observed share is below max_share. With few samples that only
    means "rarely or never observed", so each entry carries the raw count, the per-model counts
    and an upper bound on the true share, instead of a bare share."""
    by_name = {f["family"]: f for f in families}
    n = herd_map["n_labeled"]
    spots = []
    for r in herd_map["families"]:
        if r["sourced"] and not r["family"].startswith(EMERGENT) and r["share"] < max_share:
            per_model = {m: c["counts"].get(r["family"], 0) for m, c in herd_map["per_model_counts"].items()}
            sampled = [m for m, c in herd_map["per_model_counts"].items() if c["n"] > 0]
            spots.append({**by_name[r["family"]], "share": r["share"], "count": r["count"],
                          "observed": f"{r['count']} of {n} labeled samples",
                          "per_model_counts": per_model,
                          "models_missing": sum(per_model[m] == 0 for m in sampled),
                          "share_upper_95": share_upper_bound(r["count"], n)})
    # Overlooked by more models first, then the tighter bound on the pooled share.
    return sorted(spots, key=lambda b: (-b["models_missing"], b["share_upper_95"], b["family"]))
