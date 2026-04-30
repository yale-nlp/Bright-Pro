#!/usr/bin/env python3
"""End-to-end test of the rewritten retrieval_evaluation.py.

Covers:
  * unit: `build_qrels_and_scores` correctly converts a ranked list into the
    qrels/results dict shape, with descending synthetic scores preserving order
  * integration: fake runs dir + fake examples → metrics match hand-computed
    NDCG@k / Recall@k / MRR
  * CLI: subprocess invocation with --runs-dir writes the expected aggregate
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # so bright_pro_data is importable

import retrieval_evaluation as re_eval  # the rewritten module


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def test_build_qrels_and_scores_basic():
    """Assign descending scores, keep only qids that have gold."""
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td)
        # qid "a": perfect ranking (gold at top)
        (runs / "run_a.json").write_text(json.dumps({
            "query_id": "a",
            "retrieved_documents_id": ["g1", "g2", "n1", "n2"],
        }))
        # qid "b": no gold → should be skipped
        (runs / "run_b.json").write_text(json.dumps({
            "query_id": "b",
            "retrieved_documents_id": ["n3", "n4"],
        }))
        # qid "c": empty ranked list but has gold → should get empty results[qid]
        (runs / "run_c.json").write_text(json.dumps({
            "query_id": "c",
            "retrieved_documents_id": [],
        }))
        gold_full = {"a": {"g1", "g2"}, "c": {"g5"}}
        qrels, results, stats = re_eval.build_qrels_and_scores(str(runs), gold_full)

        assert set(qrels.keys()) == {"a", "c"}, qrels.keys()
        assert set(results.keys()) == {"a", "c"}, results.keys()
        # qid "a": descending scores
        assert results["a"]["g1"] > results["a"]["g2"] > results["a"]["n1"] > results["a"]["n2"]
        # qid "c": empty ranked → empty results
        assert results["c"] == {}
        assert stats["files"] == 3
        assert stats["files_with_qrels"] == 2  # a and c
        assert stats["files_no_qrels"] == 1    # b
        assert stats["files_empty_ranked"] == 1
        print("  test_build_qrels_and_scores_basic ok")


def test_metrics_on_perfect_and_flipped():
    """Synthesize a 3-query task: one perfect, one flipped, one partial.
    Compute expected NDCG@k / Recall@k by hand and compare."""
    # Gold pool:
    #   q1: gold = {g1, g2, g3}    ranking: [g1, g2, g3, n1, n2]  → perfect
    #   q2: gold = {g4, g5}        ranking: [n1, n2, g5, g4]      → partial
    #   q3: gold = {g6}            ranking: [g6]                  → perfect (k >= 1)
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td) / "runs"
        runs.mkdir()
        for qid, ranked in [("q1", ["g1", "g2", "g3", "n1", "n2"]),
                            ("q2", ["n1", "n2", "g5", "g4"]),
                            ("q3", ["g6"])]:
            (runs / f"run_{qid}.json").write_text(json.dumps({
                "query_id": qid,
                "retrieved_documents_id": ranked,
            }))
        gold_full = {"q1": {"g1", "g2", "g3"}, "q2": {"g4", "g5"}, "q3": {"g6"}}
        qrels, results, _ = re_eval.build_qrels_and_scores(str(runs), gold_full)

        from metrics import calculate_retrieval_metrics
        m = calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 25])

        # Expected:
        # Recall@5:
        #   q1 = 3/3 = 1, q2 = 2/2 = 1, q3 = 1/1 = 1  ⇒ avg 1.0
        assert approx(m["Recall@5"], 1.0, 1e-4), f"Recall@5={m['Recall@5']}"
        # Recall@1:
        #   q1: top1 = g1 → rel ⇒ 1/3
        #   q2: top1 = n1 → not rel ⇒ 0/2
        #   q3: top1 = g6 → rel ⇒ 1/1
        expected_r1 = (1 / 3 + 0 + 1) / 3
        assert approx(m["Recall@1"], expected_r1, 1e-4), f"Recall@1={m['Recall@1']} vs {expected_r1}"
        # NDCG@1:
        #   q1: dcg = 1/log2(2) = 1, idcg = 1 ⇒ 1
        #   q2: dcg = 0 ⇒ 0
        #   q3: dcg = 1/log2(2) = 1, idcg = 1 ⇒ 1
        expected_n1 = (1 + 0 + 1) / 3
        assert approx(m["NDCG@1"], expected_n1, 1e-4), f"NDCG@1={m['NDCG@1']} vs {expected_n1}"
        # MRR:
        #   q1: 1st gold at rank 1 → 1/1 = 1
        #   q2: 1st gold at rank 3 → 1/3
        #   q3: 1st gold at rank 1 → 1
        expected_mrr = (1 + 1/3 + 1) / 3
        assert approx(m["MRR"], expected_mrr, 1e-4), f"MRR={m['MRR']} vs {expected_mrr}"
        print(f"  test_metrics_on_perfect_and_flipped ok  "
              f"NDCG@1={m['NDCG@1']:.4f} MRR={m['MRR']:.4f} Recall@5={m['Recall@5']:.4f}")


def test_evaluate_runs_end_to_end():
    """evaluate_runs() must load gold from the REAL local dataset. Build a
    tempdir of run files for a real biology qid subset and check metrics."""
    import bright_pro_data as bpd

    examples = bpd.load_bright_pro("examples", "biology")[:5]
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td)
        for ex in examples:
            qid = str(ex["id"])
            # Perfect ranking: gold docs first, then 5 fake distractors
            ranked = list(ex["gold_ids"]) + [f"n_{qid}_{i}" for i in range(5)]
            (runs / f"run_{qid}.json").write_text(json.dumps({
                "query_id": qid,
                "retrieved_documents_id": ranked,
            }))

        metrics, stats = re_eval.evaluate_runs(str(runs), task="biology",
                                                k_values=[1, 5, 25])
        assert stats["files"] == 5
        assert stats["files_with_qrels"] == 5
        # Perfect ranking → NDCG@25 = Recall@25 = 1.0
        assert approx(metrics["NDCG@25"], 1.0, 1e-4), metrics
        assert approx(metrics["Recall@25"], 1.0, 1e-4), metrics
        assert approx(metrics["MRR"], 1.0, 1e-4), metrics
        print(f"  test_evaluate_runs_end_to_end ok  "
              f"NDCG@25={metrics['NDCG@25']} Recall@25={metrics['Recall@25']}")


def test_cli():
    """Run the script as a subprocess and check save-json output."""
    import bright_pro_data as bpd
    examples = bpd.load_bright_pro("examples", "biology")[:3]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        runs = td / "runs"
        runs.mkdir()
        for ex in examples:
            qid = str(ex["id"])
            ranked = list(ex["gold_ids"]) + [f"distractor_{qid}_{i}" for i in range(3)]
            (runs / f"run_{qid}.json").write_text(json.dumps({
                "query_id": qid,
                "retrieved_documents_id": ranked,
            }))
        save_path = td / "result.json"
        script = HERE / "retrieval_evaluation.py"
        r = subprocess.run(
            ["python3", str(script), "--runs-dir", str(runs), "--task", "biology",
             "--k-values", "1,25", "--save-json", str(save_path)],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, f"stderr:\n{r.stderr}"
        payload = json.loads(save_path.read_text())
        assert "per_task" in payload and "biology" in payload["per_task"]
        assert approx(payload["per_task"]["biology"]["NDCG@25"], 1.0, 1e-4)
        print(f"  test_cli ok  payload: {payload}")


def main():
    test_build_qrels_and_scores_basic()
    test_metrics_on_perfect_and_flipped()
    test_evaluate_runs_end_to_end()
    test_cli()
    print("\nAll retrieval_evaluation tests passed ✓")


if __name__ == "__main__":
    main()
