"""python -m herd run <task> [--n N] [--no-reality]
python -m herd bench <task> [--budget USD] [--repeats R] [--operators a,b]
python -m herd spend  |  python -m herd serve"""
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
        # Top 3 herd families and every blind spot, all through the same prompt template.
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


def fingerprints_cmd(task_name):
    """Fingerprint the herd map's valid samples and compare behavior clusters with labels."""
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
    parser = argparse.ArgumentParser(prog="herd")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="run the blind-spot experiment on a task")
    p.add_argument("task")
    p.add_argument("--n", type=int, help="samples per model (overrides config)")
    p.add_argument("--no-reality", action="store_true", help="stop after the herd map and blind spots")
    b = sub.add_parser("bench", help="benchmark the generation operators at equal budget")
    b.add_argument("task")
    b.add_argument("--budget", type=float, help="USD per operator per repeat (overrides config)")
    b.add_argument("--repeats", type=int, help="repeats on different seeds (overrides config)")
    b.add_argument("--operators", help="comma-separated subset, e.g. plain,verbalized")
    b.add_argument("--n", type=int, help="samples per model of the herd map to use (default: at least "
                   "reality_check.min_samples_per_model when blind-spot operators run)")
    b.add_argument("--allow-small-map", action="store_true", help="run blind-spot operators on a smaller map")
    f = sub.add_parser("fingerprints", help="behavioral fingerprints of a task's map samples (bin packing)")
    f.add_argument("task")
    pp = sub.add_parser("paraphrase", help="Step 0: does paraphrasing the prompt move the herd map?")
    pp.add_argument("task")
    sub.add_parser("spend", help="show estimated spend so far")
    sub.add_parser("serve", help="run the MCP server on stdio")
    args = parser.parse_args()

    if args.command == "run":
        run(args.task, args.n, not args.no_reality)
    elif args.command == "bench":
        from .bench import run_bench
        _, verdict, bench_dir = run_bench(args.task, args.budget, args.repeats,
                                          args.operators.split(",") if args.operators else None,
                                          allow_small_map=args.allow_small_map, n_per_model=args.n)
        print((bench_dir / "bench.md").read_text())
    elif args.command == "paraphrase":
        from .paraphrase import run as run_paraphrase
        result = run_paraphrase(Herd(), Task.resolve(args.task))
        print(json.dumps({k: v for k, v in result.items() if not k.endswith("distributions")}, indent=2))
    elif args.command == "fingerprints":
        fingerprints_cmd(args.task)
    elif args.command == "spend":
        ledger = Herd().ledger
        print(f"estimated spend: ${ledger.total_spent:.4f} of ${ledger.global_cap:.2f}")
    else:
        from .server import serve
        serve()


if __name__ == "__main__":
    main()
