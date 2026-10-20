"""scripts/bench_reanalysis.py on synthetic benchmark files: equal-K cuts, strongest baseline, cold cost."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bench_reanalysis", ROOT / "scripts" / "bench_reanalysis.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def rows(op, seed, scores):
    return [{"operator": op, "seed": seed, "valid": s is not None, "dev_score": s} for s in scores]


def test_equal_k_strongest_baseline_and_cold_cost(tmp_path):
    cands = (rows("plain", 0, [0.60, 0.58, 0.59, 0.57, 0.56, 0.55]) + rows("blind_spot", 0, [0.59, None, 0.58])
             + rows("plain", 1, [0.61, 0.60]) + rows("blind_spot", 1, [0.57, 0.62, 0.63]))
    (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cands))
    meta = {"per_run": [{"operator": "plain", "spent_usd": 0.5, "spent_cold_usd": 0.5},
                        {"operator": "blind_spot", "spent_usd": 0.3, "spent_cold_usd": 3.3}]}
    (tmp_path / "bench.json").write_text(json.dumps(meta))
    r = br.analyze(*br.load(tmp_path), {"lru": 0.675, "s3fifo": 0.585})
    assert r["k_by_seed"] == {0: 2, 1: 2} and r["strongest_dev_baseline"]["name"] == "s3fifo"
    plain, bs = r["operators"]["plain"], r["operators"]["blind_spot"]
    # seed 0: plain's first 2 valid are 0.60, 0.58 (its later 0.55 does not count at equal K)
    assert plain["best_dev_equal_k"] == (0.58 + 0.60) / 2 and bs["best_dev_equal_k"] == (0.58 + 0.57) / 2
    assert plain["beats_strongest_baseline"] == 4 and bs["beats_strongest_baseline"] == 2
    assert bs["spent_cold_usd"] == 3.3 and bs["beats_per_cold_usd"] == round(2 / 3.3, 3)
    assert "| blind_spot |" in br.markdown(r).splitlines()[4]  # better equal-K best sorts first
