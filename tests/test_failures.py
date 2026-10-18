"""diverge.failures: failure sets, Jaccard and the Mann-Whitney AUC on hand examples."""
import pytest

from diverge.failures import auc, failure_set, jaccard, relative_detail


def bins(*excess):
    return {"instances": [{"seed": i, "excess_pct": e, "bins": 0} for i, e in enumerate(excess)]}


def cache(rows):
    return {"runs": [{"trace": t, "cache_size": s, "miss_ratio": m} for t, s, m in rows]}


def test_failure_set_bin_packing():
    assert failure_set(bins(1.0, 5.0, 3.0, 0.5, 4.0)) == {"seed:1", "seed:4"}
    assert failure_set(bins(1.0, 5.0, 3.0, 0.5, 4.0), k=1) == {"seed:1"}
    assert failure_set(bins(1.0, 5.0, 3.0, 0.5, 4.0), k=3) == {"seed:1", "seed:4", "seed:2"}
    assert len(failure_set(bins(1.0, 2.0), k=5)) == 2


def test_failure_set_cache():
    d = cache([("zipf", 10, 0.4), ("zipf", 100, 0.1), ("scan", 10, 0.9), ("loop", 50, 0.4)])
    assert failure_set(d) == {"scan@10", "loop@50"}  # 0.4 tie: "loop@50" < "zipf@10"
    assert failure_set(d, k=1) == {"scan@10"}


def test_failure_set_ties_broken_by_id():
    assert failure_set(bins(2.0, 2.0, 2.0, 2.0)) == {"seed:0", "seed:1"}
    d = {"instances": [{"seed": 3, "excess_pct": 1.0}, {"seed": 1, "excess_pct": 1.0},
                       {"seed": 2, "excess_pct": 1.0}]}
    assert failure_set(d) == {"seed:1", "seed:2"}


@pytest.mark.parametrize("detail", [None, {}, {"instances": []}, {"runs": []}, "x", {"other": 1},
                                    {"instances": [{"seed": 0}]}, {"instances": [{"seed": 0, "excess_pct": "nan"}]}])
def test_failure_set_empty(detail):
    assert failure_set(detail) == frozenset()


def test_failure_set_k_zero_and_type():
    assert failure_set(bins(1.0, 2.0), k=0) == frozenset()
    assert isinstance(failure_set(bins(1.0, 2.0)), frozenset)


def test_relative_detail():
    cand, base = bins(5.0, 5.0, 5.0), bins(4.9, 1.0, 3.0)
    assert failure_set(relative_detail(cand, base)) == {"seed:1", "seed:2"}
    assert failure_set(relative_detail(None, base)) == frozenset()


def test_jaccard():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(frozenset({"a"}), ["a"]) == 1.0


def test_auc_hand_examples():
    assert auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0          # perfect
    assert auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0          # inverse
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5          # all tied
    assert auc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1]) == pytest.approx(0.875)  # one tied pair
    assert auc([1, 2, 3], [1, 1, 1]) is None and auc([], []) is None
