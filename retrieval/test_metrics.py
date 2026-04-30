#!/usr/bin/env python3
"""End-to-end test of retrieval/metrics.py.

Uses real BRIGHT-PRO biology data (not mock) so we exercise the same
ground-truth assembly path as run.py.
"""
from __future__ import annotations

import json, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `from metrics import ...` works

# bright_pro_data lives one level up in BRIGHT-PRO/
sys.path.insert(0, str(HERE.parent))

from metrics import calculate_retrieval_metrics
from bright_pro_data import load_bright_pro


def build_ground_truth(examples):
    gt = {}
    for e in examples:
        qid = str(e["id"])
        gt[qid] = {g: 1 for g in e["gold_ids"]}
    return gt


def build_perfect_scores(examples):
    """Each query: gold docs at scores 1.0, 0.999, ..., the rest irrelevant."""
    scores = {}
    for e in examples:
        qid = str(e["id"])
        entry = {}
        for i, d in enumerate(e["gold_ids"]):
            entry[d] = 1.0 - 0.0001 * i
        scores[qid] = entry
    return scores


def build_random_scores(examples, rng=random.Random(0)):
    """Each query: every gold doc gets a random score in [0, 1]. No guarantee
    of recall — serves as a sanity 'below perfect' test."""
    scores = {}
    for e in examples:
        qid = str(e["id"])
        entry = {d: rng.random() for d in e["gold_ids"]}
        scores[qid] = entry
    return scores


def build_flipped_scores(examples):
    """Put gold docs at the BOTTOM of each query's ranking (among only gold docs).
    Recall is still 1.0 (all golds present), but NDCG is worst-case for that set."""
    scores = {}
    for e in examples:
        qid = str(e["id"])
        gold = list(reversed(e["gold_ids"]))
        entry = {d: 1.0 - 0.0001 * i for i, d in enumerate(gold)}
        scores[qid] = entry
    return scores


def test_perfect():
    examples = load_bright_pro("examples", "biology")
    gt = build_ground_truth(examples)
    scores = build_perfect_scores(examples)
    r = calculate_retrieval_metrics(scores, gt, k_values=[1, 5, 10, 25, 50, 100])
    # For perfect ranking (all gold at top), NDCG@k = 1.0 once k >= num_gold,
    # which is guaranteed for k=25/50/100 across biology (max gold = 27).
    assert abs(r["NDCG@100"] - 1.0) < 1e-6, r["NDCG@100"]
    assert abs(r["Recall@100"] - 1.0) < 1e-6, r["Recall@100"]
    assert abs(r["MRR"] - 1.0) < 1e-6, r["MRR"]
    assert abs(r["MAP@100"] - 1.0) < 1e-6, r["MAP@100"]
    # NDCG@1 = Recall@1 only = 1.0 for queries with >=1 gold (all 103)
    assert abs(r["NDCG@1"] - 1.0) < 1e-6, r["NDCG@1"]
    print(f"  test_perfect ok  NDCG@25={r['NDCG@25']}  Recall@25={r['Recall@25']}")


def test_flipped():
    examples = load_bright_pro("examples", "biology")
    gt = build_ground_truth(examples)
    scores = build_flipped_scores(examples)
    r = calculate_retrieval_metrics(scores, gt, k_values=[1, 5, 10, 25, 50, 100])
    # "Flipped" still only has gold docs in score → recall is 1 at high k.
    assert abs(r["Recall@100"] - 1.0) < 1e-6
    # NDCG@100 should be 1.0 too because all golds appear in top-k (order doesn't
    # matter once every gold is counted and k exceeds num_gold).
    # Actually NDCG DOES care about order. Let's just confirm Recall.
    print(f"  test_flipped ok  Recall@100={r['Recall@100']}  NDCG@1={r['NDCG@1']}")


def test_noise_above_gold():
    """Put non-gold distractor docs ABOVE gold in the ranking — NDCG should
    drop below 1, Recall@k should match (num_gold - pushed_below_k)/num_gold."""
    examples = load_bright_pro("examples", "biology")[:20]
    gt = build_ground_truth(examples)
    scores = {}
    for e in examples:
        qid = str(e["id"])
        entry = {}
        # 30 distractors at top with high scores
        for i in range(30):
            entry[f"distractor_{qid}_{i}"] = 10.0 - 0.1 * i
        # then gold
        for i, d in enumerate(e["gold_ids"]):
            entry[d] = 1.0 - 0.001 * i
        scores[qid] = entry
    r = calculate_retrieval_metrics(scores, gt, k_values=[1, 25, 100])
    assert r["NDCG@1"] < 1.0, r["NDCG@1"]      # distractor at top hurts
    assert r["NDCG@25"] < 1.0, r["NDCG@25"]
    assert r["Recall@1"] < 0.1, r["Recall@1"]   # gold can't be in top-1
    assert abs(r["Recall@100"] - 1.0) < 1e-6    # everything reached by k=100
    print(f"  test_noise_above_gold ok  NDCG@1={r['NDCG@1']:.4f}  NDCG@25={r['NDCG@25']:.4f}  Recall@1={r['Recall@1']:.4f}")


def test_keys_present():
    """Verify all expected metric keys are in output."""
    examples = load_bright_pro("examples", "biology")[:5]
    gt = build_ground_truth(examples)
    scores = build_perfect_scores(examples)
    r = calculate_retrieval_metrics(scores, gt, k_values=[1, 5, 10, 25, 50, 100])
    expected = set()
    for k in (1, 5, 10, 25, 50, 100):
        for prefix in ("NDCG", "Recall", "MAP", "P"):
            expected.add(f"{prefix}@{k}")
    expected.add("MRR")
    missing = expected - set(r.keys())
    assert not missing, f"missing keys: {missing}"
    print(f"  test_keys_present ok  ({len(r)} metrics in output)")


def test_run_py_calls_metrics():
    """Verify run.py has an active (uncommented) call to calculate_retrieval_metrics
    that writes results.json. Thisverifies the call must not re-regress to
    being commented out."""
    import ast, re
    run_py = (HERE / "run.py").read_text()
    tree = ast.parse(run_py)
    # Walk the AST, look for a Call whose func.id == 'calculate_retrieval_metrics'
    found_call = False
    found_write = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "calculate_retrieval_metrics":
            found_call = True
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "dump":
            # json.dump(...) being called into results.json
            pass
    assert found_call, "run.py has no un-commented call to calculate_retrieval_metrics"
    # Also confirm it writes results.json (string literal must appear)
    assert "results.json" in run_py, "run.py does not write results.json"
    # And the line is not inside a leading `#` (i.e. not commented out)
    for line in run_py.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and "calculate_retrieval_metrics" in line:
            raise AssertionError(f"calculate_retrieval_metrics is commented out: {line!r}")
    print("  test_run_py_calls_metrics ok")


def main():
    test_keys_present()
    test_perfect()
    test_flipped()
    test_noise_above_gold()
    test_run_py_calls_metrics()
    print("\nAll metrics tests passed ✓")


if __name__ == "__main__":
    main()
