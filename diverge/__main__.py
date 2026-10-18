"""python -m diverge run <task> [--n N] [--no-reality]
python -m diverge bench <task> [--budget USD] [--repeats R] [--operators a,b]
python -m diverge spend  |  python -m diverge serve
python -m diverge demo <fixture>  |  python -m diverge preflight"""
import argparse
import json

from .pipeline import Herd, effective_n, ensure_family_scores, ensure_map
from .reality import Evaluator
from .sample import summarize
from .task import Task


def run(task_name, n, reality):
    herd = Herd()
    task = Task.resolve(task_name)
    study = ensure_map(herd, task, n)
    out = herd.out_dir(task)
    print(summarize([json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]))
    m = study["herd_map"]
    print(f"\nconcentration (top 3): {m['concentration_top3']:.0%}  top3: {m['top3']}")
    for r in m["families"]:
        print(f"  {r['share']:6.1%}  {r['count']:4d}  {r['family']}")
    print("blind spots (sourced families rarely or never observed):")
    for b in study["blind_spots"]:
        print(f"  {b['observed']}, true share <= {b['share_upper_95']:.1%} (95%)  {b['family']}")

    min_n = herd.cfg["reality_check"]["min_samples_per_model"]
    if reality and task.evaluator and effective_n(study) < min_n:
        print(f"\nreality check skipped: the weakest model has {effective_n(study)} labeled samples, so the "
              f"blind-spot list is noise; rerun with --n {min_n} or more")
    elif reality and task.evaluator:
        # Top 3 consensus families and every blind spot, all through the same prompt template.
        by_name = {f["family"]: f for f in study["families"]}
        families = [by_name[f] for f in m["top3"]] + study["blind_spots"]
        scores = ensure_family_scores(herd, task, study, families, Evaluator(task),
                                      herd.cfg["reality_check"]["k"])
        print("\nreality check (dev best / dev median / held-out score of the dev-selected best / valid of k):")
        fmt = lambda v: f"{v:.4f}" if v is not None else "-"
        for f in families:
            s = scores[f["family"]]
            print(f"  {s['kind']:10} {fmt(s['best_score']):>8} {fmt(s['median_score']):>8} "
                  f"{fmt(s['final_score']):>8} {s['n_valid']}/{s['k']}  {f['family']}")
        dev, heldout = task.baselines(), task.baselines("heldout")
        for name in dev:
            print(f"  {'baseline':10} {fmt(dev[name]['score']):>8} {'':>8} {fmt(heldout[name]['score']):>8}  {name}")
    print(f"\nspend this run ${herd.ledger.new_spend:.4f}; total ${herd.ledger.total_spent:.4f} "
          f"of ${herd.ledger.global_cap:.2f}")


def escape_cmd(args):
    from pathlib import Path
    from .reality import Evaluator
    from .server import escape_for_agent
    spec = Path(args.task)
    task = Task.resolve(spec.read_text() if spec.is_file() else args.task)
    if not args.evaluator and not task.evaluator:
        raise SystemExit("this task has no built-in evaluator; pass --evaluator")
    history = sorted(Path(args.history).glob("*.py")) if args.history else []
    herd = Herd(budget_usd=args.budget)
    evaluator = Evaluator(task, args.evaluator or None, not args.higher_is_better)
    r = escape_for_agent(herd, task, Path(args.candidate).read_text(), [p.read_text() for p in history], evaluator)
    ev = r["evaluation"]
    print(f"Current score: {ev['before']}")
    print(f"{len(history) + 1} attempts. Strategy similarity: {r['strategy_similarity']}  "
          f"(code similarity: {r['code_similarity']})")
    print(f"Loop detected: {r['loop_detected']}. Action: {r['action']}")
    for reason in r["reasons"]:
        print(f"  - {reason}")
    if r.get("herd_prevalence") is not None:
        print(f"Consensus prevalence of this strategy profile: {r['herd_prevalence']}")
    if r.get("herd_stability"):
        print(f"Consensus stability under rewording: {r['herd_stability']['verdict']}")
    print(f"Recommended switch: {r['recommended_switch']['mechanism']}\n  why: {r['recommended_switch']['why']}")
    if r.get("candidate"):
        print(f"Candidate: {r['candidate_path']}  dev score {ev['after']} (improved: {ev['improved']})")
    for ce in r.get("counterexamples", [])[:2]:
        print(f"Counterexample: {ce['case']} -> {ce['explanation']}")
    print(f"spent ${herd.ledger.new_spend:.4f}")
    from . import contract
    from .pipeline import read_json
    out = herd.out_dir(task)
    print()
    print(contract.render(r, read_json(out / "overlook.json"), read_json(out / "paraphrase.json"),
                          read_json(out / "blind_spots.json")))


def fingerprints_cmd(task_name):
    """Fingerprint the consensus map's valid samples and compare behavior clusters with labels."""
    from . import fingerprint
    from .pipeline import read_json, write_json
    herd = Herd()
    task = Task.resolve(task_name)
    if not fingerprint.supported(task):
        raise SystemExit(f"{task.name} has no fingerprint support")
    gate = fingerprint.validation_gate(task)
    print(f"validation gate: {gate}")
    if not gate["passed"]:
        raise SystemExit("validation gate failed; not fingerprinting")
    out = herd.out_dir(task)
    records = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    labels = {r["sample_id"]: r["family"] for r in map(json.loads, (out / "labels.jsonl").read_text().splitlines())}
    paths = {r["sample_id"]: out / r["code_path"] for r in records if r["status"] == "valid"}
    fps = fingerprint.fingerprints(task, paths)
    groups = fingerprint.clusters(fps)
    summary = {"gate": gate, "n_valid_samples": len(paths), **fingerprint.label_agreement(
        groups, {k: labels[k] for k in fps if k in labels and fps[k] is not None}),
        "clusters": [{"size": len(g), "labels": sorted({labels.get(k) for k in g}), "members": g} for g in groups]}
    write_json(out / "fingerprints.json", summary)
    print(f"{summary['n_valid_samples']} valid samples -> {summary['behavior_clusters']} behavior clusters vs "
          f"{summary['labeled_families']} labeled families; labels split across clusters: "
          f"{summary['labels_split_across_clusters']}; clusters with several labels: "
          f"{summary['clusters_with_several_labels']}")
    for c in summary["clusters"][:12]:
        print(f"  {c['size']:3d}  {c['labels']}")


def main():
    parser = argparse.ArgumentParser(prog="diverge", description=(
        "DIVERGE: help AI agents find the answers they keep missing. Different code isn't always different "
        "thinking. Built for agents (MCP: diverge serve); the CLI and offline demo are for humans too."))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="map the consensus and blind spots for a task, then score them (the blind-spot experiment)")
    p.add_argument("task")
    p.add_argument("--n", type=int, help="samples per model (overrides config)")
    p.add_argument("--no-reality", action="store_true", help="stop after the consensus map and blind spots")
    b = sub.add_parser("bench", help="benchmark the generation operators at equal budget")
    b.add_argument("task")
    b.add_argument("--budget", type=float, help="USD per operator per repeat (overrides config)")
    b.add_argument("--repeats", type=int, help="repeats on different seeds (overrides config)")
    b.add_argument("--operators", help="comma-separated subset, e.g. plain,verbalized")
    b.add_argument("--n", type=int, help="samples per model of the consensus map to use (default: at least "
                   "reality_check.min_samples_per_model when blind-spot operators run)")
    b.add_argument("--allow-small-map", action="store_true", help="run blind-spot operators on a smaller map")
    f = sub.add_parser("fingerprints", help="behavioral fingerprints of a task's map samples (bin packing)")
    f.add_argument("task")
    pp = sub.add_parser("paraphrase", help="Step 0: does paraphrasing the prompt move the consensus map?")
    pp.add_argument("task")
    ov = sub.add_parser("overlook", help="Step 1: overlooked inputs, assumptions, grid gaps, minority and latent ideas")
    ov.add_argument("task")
    es = sub.add_parser("escape", help="are my attempts one strategy in different code? if so, propose and test one evidence-backed alternative")
    es.add_argument("--task", required=True, help="registered task name, or a path to a task description file")
    es.add_argument("--candidate", required=True, help="current solution file")
    es.add_argument("--history", help="directory of earlier attempt files (sorted by name = chronological)")
    es.add_argument("--evaluator", default="", help="evaluator command (default: the task's built-in evaluator)")
    es.add_argument("--higher-is-better", action="store_true")
    es.add_argument("--budget", type=float, default=2.0, help="USD this run may spend")
    dm = sub.add_parser("demo", help="replay a recorded escape run offline (no API calls)")
    dm.add_argument("fixture", help="name under fixtures/trajectories and fixtures/runs, e.g. strategy_loop")
    pf = sub.add_parser("preflight", help="check budget, API keys and model slugs before a live run "
                        "(exit code reflects live readiness, also with --offline; the offline demo needs no keys)")
    pf.add_argument("--json", action="store_true", help="print the readiness dict (as diverge_status returns it)")
    pf.add_argument("--offline", action="store_true", help="skip the live model catalog check (no network)")
    sub.add_parser("spend", help="show estimated spend so far")
    sub.add_parser("serve", help="run the MCP server for coding agents on stdio")
    ob = sub.add_parser("onboard", help="install the DIVERGE skill, /diverge command and MCP config into a project")
    ob.add_argument("--client", default="claude", help="coding agent client (only 'claude' is supported)")
    ob.add_argument("--target", default=".", help="project directory to onboard (default: current directory)")
    ob.add_argument("--dry-run", action="store_true", help="write nothing; list the planned actions")
    ob.add_argument("--force", action="store_true", help="overwrite conflicting existing files or MCP server entry")
    ob.add_argument("--json", action="store_true", help="print a structured JSON report")
    args = parser.parse_args()

    if args.command == "run":
        run(args.task, args.n, not args.no_reality)
    elif args.command == "bench":
        from .bench import run_bench
        _, verdict, bench_dir = run_bench(args.task, args.budget, args.repeats,
                                          args.operators.split(",") if args.operators else None,
                                          allow_small_map=args.allow_small_map, n_per_model=args.n)
        print((bench_dir / "bench.md").read_text())
    elif args.command == "escape":
        escape_cmd(args)
    elif args.command == "overlook":
        from .overlook import build
        result = build(Herd(), Task.resolve(args.task))
        print(json.dumps({"inputs": [(r["input"], r["share"], r["overlooked"]) for r in result["inputs"]],
                          "shared_assumptions": [(a["assumption"], a["kind"]) for a in result["assumptions"]["assumptions"]
                                                 if a["shared"]],
                          "new_spend_usd": result["new_spend_usd"]}, indent=2))
    elif args.command == "paraphrase":
        from .paraphrase import run as run_paraphrase
        result = run_paraphrase(Herd(), Task.resolve(args.task))
        print(json.dumps({k: v for k, v in result.items() if not k.endswith("distributions")}, indent=2))
    elif args.command == "fingerprints":
        fingerprints_cmd(args.task)
    elif args.command == "demo":
        from . import demo
        raise SystemExit(demo.run(args.fixture))
    elif args.command == "preflight":
        from . import demo
        if args.json:
            r = demo.readiness(live=not args.offline)
            print(json.dumps(r, indent=2))
            raise SystemExit(0 if r["ready"] else 1)
        raise SystemExit(demo.preflight(live=not args.offline))
    elif args.command == "onboard":
        from . import onboard
        raise SystemExit(onboard.main(args))
    elif args.command == "spend":
        ledger = Herd().ledger
        print(f"estimated spend: ${ledger.total_spent:.4f} of ${ledger.global_cap:.2f}")
    else:
        from .server import serve
        serve()


if __name__ == "__main__":
    main()
