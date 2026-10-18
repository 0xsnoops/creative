"""BENCHMARK: every operator at the same budget, repeated on different seeds.

- Candidates from all operators are labeled together by one labeler against one taxonomy,
  so family counts are comparable.
- Scores: each family's candidate is selected on the dev split; reported scores and
  "useful" decisions use that candidate's held-out score, compared with held-out baselines.
- Cost: "cached" counts only what the operator itself spent. "Cold-start" adds the setup an
  operator depends on (the herd map for blind_spot, the literature taxonomy for
  literature_all), as if every repeat started from an empty cache. Labeling the candidates
  is measurement and billed to no operator.
- The repeat "seeds" vary generation, not the evaluation data, and three repeats are a
  small sample: comparisons describe what was observed, not a general guarantee."""
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from .label import UNLABELED, label_samples
from .operators import run_operator
from .llm import BudgetExceeded
from .pipeline import Herd, effective_n, ensure_map, read_json, write_json
from .reality import Evaluator
from .task import Task

# Reference palette (dataviz skill): categorical slots 1-2, neutral ink and surface.
SURFACE, INK, INK_2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"

# Setup each operator relies on, as stages of the herd map build (see ensure_map).
MAP = ("sampling", "literature", "labeling")
SETUP_STAGES = {"blind_spot": MAP, "blind_spot_contrast": MAP, "blind_spot_contrast_random": MAP,
                "blind_spot_feedback": MAP, "blind_spot_feedback_sham": MAP,
                "literature_all": ("literature",), "literature_random": ("literature",)}
# Operators whose families come from the herd map's blind spots (or are sized by them).
NEEDS_MAP = {"blind_spot", "blind_spot_contrast", "blind_spot_contrast_random", "blind_spot_feedback",
             "blind_spot_feedback_sham", "literature_random"}


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None, "sd": None, "n": 0}
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def candidate_id(c):
    return f"{c['operator']}-s{c['seed']}-{c['idx']:03d}"


def setup_cost(operator, build_cost):
    """Cold-start setup cost for an operator, or None if the map predates cost tracking."""
    stages = SETUP_STAGES.get(operator, ())
    if stages and not build_cost:
        return None
    return sum(build_cost.get(s, 0.0) for s in stages)


def label_candidates(herd, task, runs, out):
    """Label every valid candidate by mechanism, with one shared taxonomy and labeler."""
    records = [{"sample_id": candidate_id(c), "model": f"{c['operator']}/s{c['seed']}", "idx": c["idx"],
                "code_path": c["code_path"], "text_path": c["code_path"]}
               for run in runs for c in run["candidates"] if c["valid"]]
    taxonomy = read_json(out / "taxonomy.json")
    return label_samples(task, records, taxonomy, herd.llm, herd.cfg, out, herd.progress)


def run_metrics(run, labels, evaluator, out, best_baseline, setup, taxonomy_names=()):
    """Per-run metrics. Selection (each family's pick, the overall pick) uses dev scores; the
    reported scores are the picks' held-out scores. All comparisons follow the evaluator's
    direction."""
    candidates = run["candidates"]
    valid = [c for c in candidates if c["valid"]]
    family_of = {candidate_id(c): labels.get(candidate_id(c), {}).get("family", UNLABELED) for c in valid}
    picks = {}  # family -> its best candidate on dev
    for c in valid:
        family = family_of[candidate_id(c)]
        if family != UNLABELED and (family not in picks or evaluator.beats(c["score"], picks[family]["score"])):
            picks[family] = c
    best_dev = lambda cs: next((c for c in cs if c["score"] == evaluator.best([x["score"] for x in cs])), None) \
        if cs else None
    overall = best_dev(valid)
    # Purity: does a candidate implement the taxonomy family it was asked for?
    intended = [c for c in valid if c.get("intended_family") in taxonomy_names]
    pure = [c for c in intended if family_of[candidate_id(c)] == c["intended_family"]]
    pure_pick = best_dev(pure)
    to_score = {**{candidate_id(c): c for c in picks.values()},
                **({candidate_id(overall): overall} if overall else {}),
                **({candidate_id(pure_pick): pure_pick} if pure_pick else {})}
    with ThreadPoolExecutor(8) as pool:
        heldout = dict(zip(to_score, pool.map(
            lambda c: evaluator.final(out / c["code_path"], c["score"])[0], to_score.values())))
    family_heldout = {f: heldout[candidate_id(c)] for f, c in picks.items()}
    useful = sorted(f for f, s in family_heldout.items() if s is not None and evaluator.beats(s, best_baseline))
    spent = run["spent_usd"]
    cold = spent + setup if setup is not None else None
    return {"operator": run["operator"], "seed": run["seed"], "n_candidates": len(candidates),
            "valid_rate": len(valid) / len(candidates) if candidates else None,
            "distinct_valid_families": len(picks), "useful_discoveries": len(useful),
            "useful_families": useful, "family_heldout_scores": family_heldout,
            "spent_usd": spent, "spent_cold_usd": cold,
            "useful_discovery_rate": len(useful) / spent if spent > 0 else None,
            "useful_discovery_rate_cold": len(useful) / cold if cold else None,
            "dev_best_score": overall["score"] if overall else None,
            "best_score": heldout.get(candidate_id(overall)) if overall else None,
            "purity": len(pure) / len(intended) if intended else None,
            "pure_best_score": heldout.get(candidate_id(pure_pick)) if pure_pick else None,
            "heldout_by_candidate": heldout,
            "selected": {**{candidate_id(c): f"family_pick:{f}" for f, c in picks.items()},
                         **({candidate_id(overall): "overall_pick"} if overall else {})},
            "notes": run["notes"]}


METRICS = ("valid_rate", "distinct_valid_families", "useful_discoveries", "useful_discovery_rate",
           "useful_discovery_rate_cold", "best_score", "dev_best_score", "spent_usd", "spent_cold_usd",
           "n_candidates", "purity", "pure_best_score")


def aggregate(per_run, operators):
    table = {}
    for op in operators:
        rows = [r for r in per_run if r["operator"] == op]
        table[op] = {k: spread([r[k] for r in rows]) for k in METRICS}
        table[op]["useful_families_any_repeat"] = sorted({f for r in rows for f in r["useful_families"]})
    return table


def compare(table, a, b, metric="useful_discovery_rate", higher_is_better=True, unit="discoveries/$"):
    """(outcome, text): +1 if a did better than b in every repeat, -1 if worse in every repeat,
    0 if the ranges overlap. Non-overlap across 3 repeats is a small-sample observation."""
    x, y = table[a][metric], table[b][metric]
    if x["mean"] is None or y["mean"] is None:
        return 0, f"{a} vs {b} ({metric}): not comparable, no data for one of them."
    better = (lambda p, q: p > q) if higher_is_better else (lambda p, q: p < q)
    worst_x, best_x = (x["min"], x["max"]) if higher_is_better else (x["max"], x["min"])
    worst_y, best_y = (y["min"], y["max"]) if higher_is_better else (y["max"], y["min"])
    vs = f"{x['mean']:.3g} vs {y['mean']:.3g} {unit}"
    if better(worst_x, best_y):
        return 1, f"{a} did better than {b} in every repeat ({vs}; {x['n']} repeats)."
    if better(worst_y, best_x):
        return -1, f"{a} did worse than {b} in every repeat ({vs}; {x['n']} repeats)."
    return 0, f"{a} vs {b}: ranges across repeats overlap ({vs}), no clear difference."


def verdict(table, baseline_name, best_baseline):
    ops, lines = list(table), []
    if {"blind_spot", "literature_all"} <= set(ops):
        checks = [("best_score", False, "held-out score"),
                  ("useful_discovery_rate_cold", True, "discoveries/$ cold-start"),
                  ("useful_discovery_rate", True, "discoveries/$ cached")]
        compared = [compare(table, "blind_spot", "literature_all", m, h, u) for m, h, u in checks]
        results = [c[0] for c in compared]
        if results[0] == 1 and results[1] >= 0:
            conclusion = "supports the thesis: knowing where models concentrate chose better experiments"
        elif results[0] == -1 or results[1] == -1:
            conclusion = ("does not support the thesis: implementing every literature family did as well "
                          "or better without the herd map")
        else:
            conclusion = "is inconclusive at this budget"
        lines.append(f"Ablation, blind_spot vs literature_all (the same work without the herd map) {conclusion}. "
                     + " ".join(c[1] for c in compared))
    if all((table[o]["useful_discoveries"]["max"] or 0) == 0 for o in ops):
        lines.append(f"No operator produced a family whose held-out score beats {baseline_name} "
                     f"({best_baseline:.4f}) at this budget, so the operators are not separated on useful discoveries.")
    else:
        ranked = sorted(ops, key=lambda o: table[o]["useful_discovery_rate"]["mean"] or 0, reverse=True)
        line = (f"Highest useful-discovery rate (cached): {ranked[0]} "
                f"({table[ranked[0]]['useful_discovery_rate']['mean']:.2f} per $).")
        if len(ranked) > 1:
            line += " " + compare(table, ranked[0], ranked[1])[1]
        lines.append(line)
    for op in ("blind_spot", "transplant"):
        if op in table and "verbalized" in table:
            lines.append(compare(table, op, "verbalized")[1])
    scored = [o for o in ops if table[o]["best_score"]["mean"] is not None]
    if scored:
        best = min(scored, key=lambda o: table[o]["best_score"]["mean"])
        s = table[best]["best_score"]
        lines.append(f"Best held-out score: {best} (mean {s['mean']:.4f}, range {s['min']:.4f}-{s['max']:.4f}); "
                     f"best held-out baseline {baseline_name} {best_baseline:.4f}.")
    return " ".join(lines)


def chart(table, baselines, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = list(table)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
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

    xs = range(len(ops))
    for offset, metric, color, label in ((-0.17, "useful_discovery_rate", BLUE, "cached"),
                                         (0.17, "useful_discovery_rate_cold", ORANGE, "cold-start")):
        mean, err = errors(metric)
        left.bar([x + offset for x in xs], [m or 0 for m in mean], width=0.32, color=color, label=label,
                 yerr=err, ecolor=INK_2, capsize=3, error_kw={"linewidth": 1.1})
    left.set_xticks(list(xs), ops, rotation=15)
    left.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    left.set_title("Useful discoveries per dollar (higher is better)", color=INK, fontsize=10, loc="left")
    left.set_ylim(bottom=0)

    mean, err = errors("best_score")
    shown = [i for i, m in enumerate(mean) if m is not None]
    right.errorbar(shown, [mean[i] for i in shown], yerr=[[err[0][i] for i in shown], [err[1][i] for i in shown]],
                   fmt="o", color=BLUE, ecolor=INK_2, markersize=8, capsize=4, linewidth=1.2)
    for i, m in enumerate(mean):
        if m is None:
            right.annotate("no valid code", (i, 0.02), xycoords=("data", "axes fraction"),
                           color=INK_2, fontsize=8, ha="center")
    for name, score in baselines.items():
        right.axhline(score, color=INK_2, linestyle="--", linewidth=1)
        right.annotate(f"{name} {score:.3f}", (len(ops) - 0.5, score), color=INK_2, fontsize=8,
                       ha="right", va="bottom")
    right.set_xticks(list(xs), ops, rotation=15)
    right.set_xlim(-0.5, len(ops) - 0.5)
    right.set_title("Best held-out score, mean and range (lower is better)", color=INK, fontsize=10, loc="left")
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
             f"Each operator: ${budget:.2f} per repeat, {repeats} repeats (generation seeds 0-{repeats - 1}), "
             f"implementer `{cfg['models']['implementer']}`, labeler `{cfg['models']['labeler']}`. "
             f"Cells are mean [min, max] across repeats.", "",
             "| operator | candidates | valid_rate | distinct_valid_families | useful_discoveries "
             "| discoveries/$ cached | discoveries/$ cold-start | best held-out score | spent $ | cold-start $ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for op, t in table.items():
        lines.append(f"| {op} | {fmt(t['n_candidates'], 1)} | {fmt(t['valid_rate'])} | "
                     f"{fmt(t['distinct_valid_families'], 1)} | {fmt(t['useful_discoveries'], 1)} | "
                     f"{fmt(t['useful_discovery_rate'], 2)} | {fmt(t['useful_discovery_rate_cold'], 2)} | "
                     f"{fmt(t['best_score'], 4)} | {fmt(t['spent_usd'], 3)} | {fmt(t['spent_cold_usd'], 3)} |")
    lines += ["", "Held-out reference baselines: " + ", ".join(f"{k} {v:.4f}" for k, v in baselines.items()),
              "", "## Verdict", "", verdict_text, "", "## Notes", "",
              "- Selection uses the dev instances only. Each family's dev-selected candidate is then scored once "
              "on held-out instances; every score and \"useful\" decision above uses held-out numbers.",
              "- useful_discoveries: distinct valid families (by labeled mechanism) whose held-out score beats "
              "the best held-out reference baseline.",
              "- Cached cost counts only the operator's own calls. Cold-start adds the setup it relies on, as if "
              f"each repeat started empty: blind_spot + full herd map (${setup['blind_spot'] or 0:.3f}), "
              f"literature_all + literature search and taxonomy (${setup['literature_all'] or 0:.3f}).",
              f"- Not billed to any operator: labeling the candidates (${setup['label_spend']:.3f} new spend "
              f"this run). New spend on the herd map this run: ${setup['map_spend']:.3f} (0 means it was cached).",
              "- Generation seeds vary the candidates, not the evaluation data; 3 repeats are a small sample.",
              "- Family counts depend on the labeler's granularity; all operators share the same labeler "
              "and taxonomy, so the counts are comparable with each other but not absolute."]
    if setup["map_n_per_model"] < cfg["reality_check"]["min_samples_per_model"]:
        lines.append(f"- Caution: the herd map behind blind_spot used only {setup['map_n_per_model']} samples per "
                     f"model, so its blind spots are families not observed in a small sample.")
    for r in per_run:
        if r["notes"].get("error"):
            lines.append(f"- {r['operator']}, seed {r['seed']}: stopped early after an error: {r['notes']['error']}")
    for r in per_run:
        if r["operator"] == "hot":
            lines.append(f"- hot, seed {r['seed']}: temperature {r['notes'].get('temperature')} "
                         f"(calibration valid counts {r['notes'].get('calibration_valid_of_n')}).")
    lines += ["", "![benchmark](bench.png)"]
    return "\n".join(lines) + "\n"


def labeling_estimate(herd, operators, repeats):
    """Worst-case cost of labeling every candidate the benchmark could produce."""
    cfg = herd.cfg
    info = herd.llm.catalog[cfg["models"]["labeler"]]
    price_in, price_out = float(info["pricing"]["prompt"]), float(info["pricing"]["completion"])
    batch = cfg["labeling"]["batch_size"]
    n_batches = -(-len(operators) * repeats * cfg["bench"]["max_candidates"] // batch)
    per_batch = (batch * 1500 + 3000) * price_in + 3000 * price_out  # code + taxonomy in, labels out
    return n_batches * per_batch


def candidate_records(runs, per_run, labels, out):
    """One trace row per generated candidate: prompt, code, scores, labeled family, selection."""
    rows = []
    for run, metrics in zip(runs, per_run):
        for c in run["candidates"]:
            cid = candidate_id(c) if "idx" in c else None
            code = (out / c["code_path"]).read_text() if c.get("code_path") else None
            rows.append({"candidate_id": cid, "operator": c["operator"], "seed": c["seed"],
                         "intended_family": c.get("intended_family"),
                         "labeled_family": labels.get(cid, {}).get("family") if cid else None,
                         "valid": c["valid"], "dev_score": c["score"], "error": c.get("error"),
                         "heldout_score": metrics["heldout_by_candidate"].get(cid),
                         "selected": metrics["selected"].get(cid),
                         "meta": {k: c[k] for k in ("temperature", "contrast_kind", "contrast_families",
                                                    "feedback_round", "feedback_from", "frame", "adhd_round")
                                  if k in c},
                         "prompt": c.get("prompt"), "code": code, "dev_detail": c.get("dev_detail")})
    return rows


def run_bench(task_name, budget_usd=None, repeats=None, operators=None, progress=print, allow_small_map=False):
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
    min_n = cfg["reality_check"]["min_samples_per_model"]
    if NEEDS_MAP & set(operators) and effective_n(study) < min_n and not allow_small_map:
        raise SystemExit(f"the herd map has only {effective_n(study)} labeled samples per model for its weakest "
                         f"model (need {min_n}); blind spots from it are noise. Build a bigger map "
                         f"(python -m herd run {task.name} --n {min_n} --no-reality) or pass --allow-small-map")
    # Operators run one after another; if the global cap could run out partway, the last ones
    # would be underfunded and look worse. Refuse to start instead.
    labeling = labeling_estimate(herd, operators, bench_cfg["repeats"])
    needed = len(operators) * bench_cfg["repeats"] * bench_cfg["budget_usd_per_operator"] + labeling
    remaining = herd.ledger.global_cap - herd.ledger.total_spent
    if needed > remaining:
        raise SystemExit(f"the benchmark may spend up to ${needed:.2f} ({len(operators)} operators x "
                         f"{bench_cfg['repeats']} repeats x ${bench_cfg['budget_usd_per_operator']:.2f}, plus up to "
                         f"${labeling:.2f} labeling) but only ${remaining:.2f} is left under the global cap; "
                         f"raise budget_usd in config.yaml or lower --budget/--repeats")
    runs = [run_operator(op, task, bench_cfg["budget_usd_per_operator"], seed, study, progress)
            for seed in range(bench_cfg["repeats"]) for op in operators]

    out = herd.out_dir(task)
    bench_dir = out / "bench"
    write_json(bench_dir / "runs.json", runs)  # keep generation results even if labeling stops
    try:
        labels, emergent = label_candidates(herd, task, runs, out)
        labeling_status = {"status": "ok"}
    except BudgetExceeded as e:
        labels, emergent = {}, []
        labeling_status = {"status": "stopped", "detail": str(e)}
    label_spend = herd.ledger.new_spend - map_spend

    evaluator = Evaluator(task)
    baselines = {name: r["score"] for name, r in task.baselines("heldout").items()}
    baseline_name = evaluator.best(baselines.values())
    baseline_name = next(n for n, v in baselines.items() if v == baseline_name)
    build_cost = study["herd_map"].get("build_cost_usd") or {}
    taxonomy_names = {f["family"] for f in study["families"]}
    per_run = [run_metrics(run, labels, evaluator, out, baselines[baseline_name],
                           setup_cost(run["operator"], build_cost), taxonomy_names) for run in runs]
    table = aggregate(per_run, operators)
    verdict_text = verdict(table, baseline_name, baselines[baseline_name])

    write_json(bench_dir / "labels.json", {"labels": labels, "emergent_families": emergent})
    rows = candidate_records(runs, per_run, labels, out)
    with (bench_dir / "candidates.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    if herd.trace_dir:
        with (herd.trace_dir / "candidates.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps({"task": task.name, **row}) + "\n")
    write_json(bench_dir / "bench.json", {"config": bench_cfg, "heldout_baselines": baselines,
                                          "herd_map_build_cost_usd": build_cost, "labeling": labeling_status,
                                          "map_effective_n": effective_n(study),
                                          "per_run": [{k: v for k, v in r.items() if k != "heldout_by_candidate"}
                                                      for r in per_run],
                                          "table": table, "verdict": verdict_text})
    chart(table, baselines, bench_dir / "bench.png",
          f"{task.name}: operators at ${bench_cfg['budget_usd_per_operator']:.2f} each, "
          f"{bench_cfg['repeats']} repeats")
    setup = {"blind_spot": setup_cost("blind_spot", build_cost),
             "literature_all": setup_cost("literature_all", build_cost),
             "map_spend": map_spend, "label_spend": label_spend,
             "map_n_per_model": effective_n(study), "labeling": labeling_status}
    (bench_dir / "bench.md").write_text(report(task, table, per_run, baselines, cfg, verdict_text, setup))
    return table, verdict_text, bench_dir
