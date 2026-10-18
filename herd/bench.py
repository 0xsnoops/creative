"""BENCHMARK: every operator at the same budget, repeated on different seeds.

Candidates from all operators are labeled together by the same labeler against the same
taxonomy, so family counts are comparable. Labeling and the herd map are measurement and
setup costs: they are reported but not billed to any operator."""
import json
import statistics

from .label import UNLABELED, label_samples
from .operators import run_operator
from .pipeline import Herd, ensure_map, read_json, write_json
from .task import Task

# Reference palette (dataviz skill): one categorical hue, neutral ink and surface.
SURFACE, INK, INK_2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None, "sd": None, "n": 0}
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def candidate_id(c):
    return f"{c['operator']}-s{c['seed']}-{c['idx']:03d}"


def label_candidates(herd, task, runs, out):
    """Label every valid candidate by mechanism, with one shared taxonomy and labeler."""
    records = [{"sample_id": candidate_id(c), "model": f"{c['operator']}/s{c['seed']}", "idx": c["idx"],
                "code_path": c["code_path"], "text_path": c["code_path"]}
               for run in runs for c in run["candidates"] if c["valid"]]
    taxonomy = read_json(out / "taxonomy.json")
    labels, emergent = label_samples(task, records, taxonomy, herd.llm, herd.cfg, out, herd.progress)
    return labels, emergent


def run_metrics(run, labels, best_baseline):
    candidates = run["candidates"]
    valid = [c for c in candidates if c["valid"]]
    family_best = {}
    for c in valid:
        family = labels.get(candidate_id(c), {}).get("family", UNLABELED)
        if family != UNLABELED:
            family_best[family] = min(family_best.get(family, c["score"]), c["score"])
    useful = sorted(f for f, s in family_best.items() if s < best_baseline)
    spent = run["spent_usd"]
    return {"operator": run["operator"], "seed": run["seed"], "n_candidates": len(candidates),
            "valid_rate": len(valid) / len(candidates) if candidates else None,
            "distinct_valid_families": len(family_best), "useful_discoveries": len(useful),
            "useful_families": useful, "spent_usd": spent,
            "useful_discovery_rate": len(useful) / spent if spent > 0 else None,
            "best_score": min((c["score"] for c in valid), default=None), "notes": run["notes"]}


def aggregate(per_run, operators):
    table = {}
    for op in operators:
        rows = [r for r in per_run if r["operator"] == op]
        table[op] = {k: spread([r[k] for r in rows]) for k in
                     ("valid_rate", "distinct_valid_families", "useful_discoveries",
                      "useful_discovery_rate", "best_score", "spent_usd", "n_candidates")}
        table[op]["useful_families_any_repeat"] = sorted({f for r in rows for f in r["useful_families"]})
    return table


def compare(table, a, b, metric="useful_discovery_rate"):
    """Plain-language comparison that only claims a win when the repeat ranges do not overlap."""
    x, y = table[a][metric], table[b][metric]
    if x["mean"] is None or y["mean"] is None:
        return f"{a} vs {b}: not comparable (no data for one of them)."
    if x["min"] > y["max"]:
        return f"{a} beats {b} ({x['mean']:.2f} vs {y['mean']:.2f} discoveries/$; ranges do not overlap)."
    if x["max"] < y["min"]:
        return f"{a} loses to {b} ({x['mean']:.2f} vs {y['mean']:.2f} discoveries/$; ranges do not overlap)."
    lead = "ahead of" if x["mean"] > y["mean"] else "behind" if x["mean"] < y["mean"] else "level with"
    return (f"{a} is {lead} {b} on average ({x['mean']:.2f} vs {y['mean']:.2f} discoveries/$), "
            f"but the repeat ranges overlap, so this is not a clear result.")


def verdict(table, baseline_name, best_baseline):
    ops = list(table)
    lines = []
    if all((table[o]["useful_discoveries"]["max"] or 0) == 0 for o in ops):
        lines.append(f"No operator produced a valid family that beats {baseline_name} ({best_baseline:.4f}) "
                     f"at this budget, so the benchmark does not separate the operators on useful discoveries.")
    else:
        ranked = sorted(ops, key=lambda o: table[o]["useful_discovery_rate"]["mean"] or 0, reverse=True)
        lines.append(f"Highest useful-discovery rate: {ranked[0]} "
                     f"({table[ranked[0]]['useful_discovery_rate']['mean']:.2f} per $). "
                     + compare(table, ranked[0], ranked[1]))
    for op in ("blind_spot", "transplant"):
        if op in table and "verbalized" in table:
            lines.append(compare(table, op, "verbalized"))
    scored = [o for o in ops if table[o]["best_score"]["mean"] is not None]
    if scored:
        best = min(scored, key=lambda o: table[o]["best_score"]["mean"])
        s = table[best]["best_score"]
        lines.append(f"Best score: {best} (mean {s['mean']:.4f}, range {s['min']:.4f}-{s['max']:.4f}); "
                     f"best baseline {baseline_name} {best_baseline:.4f}.")
    return " ".join(lines)


def chart(table, baselines, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = list(table)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def errors(metric):
        mean = [table[o][metric]["mean"] for o in ops]
        lo = [(m - table[o][metric]["min"]) if m is not None else 0 for o, m in zip(ops, mean)]
        hi = [(table[o][metric]["max"] - m) if m is not None else 0 for o, m in zip(ops, mean)]
        return mean, [lo, hi]

    mean, err = errors("useful_discovery_rate")
    left.bar(ops, [m or 0 for m in mean], width=0.45, color=BAR, yerr=err, ecolor=INK_2, capsize=4,
             error_kw={"linewidth": 1.2})
    left.set_title("Useful discoveries per dollar (higher is better)", color=INK, fontsize=10, loc="left")
    left.set_ylim(bottom=0)

    mean, err = errors("best_score")
    xs = [i for i, m in enumerate(mean) if m is not None]
    right.errorbar(xs, [mean[i] for i in xs], yerr=[[err[0][i] for i in xs], [err[1][i] for i in xs]],
                   fmt="o", color=BAR, ecolor=INK_2, markersize=8, capsize=4, linewidth=1.2)
    for i, m in enumerate(mean):
        if m is None:
            right.annotate("no valid code", (i, 0.02), xycoords=("data", "axes fraction"),
                           color=INK_2, fontsize=8, ha="center")
    for name, score in baselines.items():
        right.axhline(score, color=INK_2, linestyle="--", linewidth=1)
        right.annotate(f"{name} {score:.3f}", (len(ops) - 0.5, score), color=INK_2, fontsize=8,
                       ha="right", va="bottom")
    right.set_xticks(range(len(ops)), ops)
    right.set_xlim(-0.5, len(ops) - 0.5)
    right.set_title("Best score per repeat, mean and range (lower is better)", color=INK, fontsize=10, loc="left")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fmt(s, digits=3):
    if s["mean"] is None:
        return "-"
    return f"{s['mean']:.{digits}f} [{s['min']:.{digits}f}, {s['max']:.{digits}f}]"


def report(task, table, per_run, baselines, cfg, verdict_text, setup):
    budget, repeats = cfg["bench"]["budget_usd_per_operator"], cfg["bench"]["repeats"]
    lines = [f"# Operator benchmark: {task.name}", "",
             f"Each operator: ${budget:.2f} per repeat, {repeats} repeats (seeds 0-{repeats - 1}), "
             f"implementer `{cfg['models']['implementer']}`, labeler `{cfg['models']['labeler']}`. "
             f"Cells are mean [min, max] across repeats.", "",
             "| operator | candidates | valid_rate | distinct_valid_families | useful_discoveries "
             "| useful_discovery_rate (/$) | best_score | spent $ |",
             "|---|---|---|---|---|---|---|---|"]
    for op, t in table.items():
        lines.append(f"| {op} | {fmt(t['n_candidates'], 1)} | {fmt(t['valid_rate'])} | "
                     f"{fmt(t['distinct_valid_families'], 1)} | {fmt(t['useful_discoveries'], 1)} | "
                     f"{fmt(t['useful_discovery_rate'], 2)} | {fmt(t['best_score'], 4)} | {fmt(t['spent_usd'], 3)} |")
    lines += ["", "Reference baselines: " + ", ".join(f"{k} {v:.4f}" for k, v in baselines.items()), "",
              "## Verdict", "", verdict_text, "", "## Notes", "",
              "- useful_discoveries: distinct valid families (by labeled mechanism) whose best candidate "
              "beats the best reference baseline.",
              "- Spend counts cached responses at their original cost, so reruns reproduce the table.",
              f"- Not billed to any operator: the herd map that blind_spot draws on "
              f"(${setup['map_spend']:.3f} spent on it in this run; 0 means it was already built) and "
              f"labeling the candidates (${setup['label_spend']:.3f}).",
              "- Family counts depend on the labeler's granularity; all operators share the same labeler "
              "and taxonomy, so the counts are comparable with each other but not absolute."]
    n = cfg["sampling"]["n_per_model"]
    if n < cfg["reality_check"]["min_samples_per_model"]:
        lines.append(f"- Caution: the herd map behind blind_spot used only {n} samples per model, so "
                     f"its blind-spot list is noisy (any unused literature family qualifies).")
    for r in per_run:
        if r["notes"].get("error"):
            lines.append(f"- {r['operator']}, seed {r['seed']}: stopped early after an error: {r['notes']['error']}")
    for r in per_run:
        if r["operator"] == "hot":
            lines.append(f"- hot, seed {r['seed']}: temperature {r['notes'].get('temperature')} "
                         f"(calibration valid counts {r['notes'].get('calibration_valid_of_n')}).")
    lines += ["", "![benchmark](bench.png)"]
    return "\n".join(lines) + "\n"


def run_bench(task_name, budget_usd=None, repeats=None, operators=None, progress=print):
    task = Task.resolve(task_name)
    if not task.evaluator:
        raise SystemExit("the benchmark needs a registered task with an evaluator")
    herd = Herd(progress=progress)  # setup and measurement: global cap only
    cfg = herd.cfg
    bench_cfg = cfg["bench"]
    bench_cfg["budget_usd_per_operator"] = budget_usd or bench_cfg["budget_usd_per_operator"]
    bench_cfg["repeats"] = repeats or bench_cfg["repeats"]
    operators = operators or bench_cfg["operators"]

    study = ensure_map(herd, task)
    map_spend = herd.ledger.new_spend
    # Operators run one after another; if the global cap could run out partway, the last ones
    # would be underfunded and look worse. Refuse to start instead.
    needed = len(operators) * bench_cfg["repeats"] * bench_cfg["budget_usd_per_operator"]
    remaining = herd.ledger.global_cap - herd.ledger.total_spent
    if needed > remaining:
        raise SystemExit(f"the benchmark may spend up to ${needed:.2f} ({len(operators)} operators x "
                         f"{bench_cfg['repeats']} repeats x ${bench_cfg['budget_usd_per_operator']:.2f}) but only "
                         f"${remaining:.2f} is left under the global cap; raise budget_usd in config.yaml "
                         f"or lower --budget/--repeats")
    runs = [run_operator(op, task, bench_cfg["budget_usd_per_operator"], seed, study, progress)
            for seed in range(bench_cfg["repeats"]) for op in operators]

    out = herd.out_dir(task)
    labels, emergent = label_candidates(herd, task, runs, out)
    label_spend = herd.ledger.new_spend - map_spend

    baselines = {name: r["score"] for name, r in task.baselines().items()}
    baseline_name = min(baselines, key=baselines.get)
    per_run = [run_metrics(run, labels, baselines[baseline_name]) for run in runs]
    table = aggregate(per_run, operators)
    verdict_text = verdict(table, baseline_name, baselines[baseline_name])

    bench_dir = out / "bench"
    write_json(bench_dir / "labels.json", {"labels": labels, "emergent_families": emergent})
    write_json(bench_dir / "bench.json", {"config": bench_cfg, "baselines": baselines, "per_run": per_run,
                                          "table": table, "verdict": verdict_text})
    chart(table, baselines, bench_dir / "bench.png",
          f"{task.name}: operators at ${bench_cfg['budget_usd_per_operator']:.2f} each, "
          f"{bench_cfg['repeats']} repeats")
    (bench_dir / "bench.md").write_text(report(task, table, per_run, baselines, cfg, verdict_text,
                                               {"map_spend": map_spend, "label_spend": label_spend}))
    return table, verdict_text, bench_dir
