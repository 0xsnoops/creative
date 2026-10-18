"""HERD MAP and BLIND SPOTS."""
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
    per_model = {}
    for model in dict.fromkeys(model_of.values()):
        _, s = shares([l["family"] for sid, l in labels.items() if model_of[sid] == model])
        per_model[model] = {f: round(v, 4) for f, v in sorted(s.items(), key=lambda kv: -kv[1])}
    top3 = [r["family"] for r in rows[:3] if r["count"]]
    return {"n_samples": len(records), "n_labeled": sum(counts.values()),
            "n_unlabeled": sum(l["family"] == UNLABELED for l in labels.values()),
            "concentration_top3": round(sum(r["share"] for r in rows[:3]), 4),
            "top3": top3, "families": rows, "per_model": per_model}


def find_blind_spots(herd_map, families, max_share):
    """Families with a literature source that fewer than max_share of samples used."""
    by_name = {f["family"]: f for f in families}
    return [{**by_name[r["family"]], "share": r["share"], "count": r["count"]}
            for r in herd_map["families"]
            if r["sourced"] and not r["family"].startswith(EMERGENT) and r["share"] < max_share]
