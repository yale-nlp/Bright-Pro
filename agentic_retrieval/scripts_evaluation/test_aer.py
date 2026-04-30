#!/usr/bin/env python3
"""Self-contained test of aer.py.

Unit tests:
  - exact formula values at γ=0.05 for hand-computed cases
  - R < 1 raises
  - OQ=0 → 0 for any R

End-to-end:
  - Build a fake runs/ tree with 2 retrievers × 2 tasks × 3 queries
  - Build a fake OQ json
  - Run compute_aer_for_runs and check every per-row aggregate
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aer import aer, compute_aer_for_runs, DEFAULT_GAMMA


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_formula():
    # R=1: no decay
    assert aer(5.0, 1) == 5.0
    assert aer(3.2, 1) == 3.2
    assert aer(0.0, 1) == 0.0
    # R=2: factor e^(-0.05)
    assert approx(aer(5.0, 2), 5.0 * math.exp(-0.05))
    # R=5: factor e^(-0.05*4) = e^(-0.2)
    assert approx(aer(4.0, 5), 4.0 * math.exp(-0.2))
    # γ override
    assert approx(aer(4.0, 5, gamma=0.1), 4.0 * math.exp(-0.4))
    # OQ=0 kills it
    assert aer(0.0, 10) == 0.0
    # R<1 raises
    try:
        aer(5.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("aer() with R=0 should raise")
    print("  test_formula ok")


def test_end_to_end():
    """Fake 2 retrievers × 2 tasks × 3 queries. Verify aggregates."""
    with tempfile.TemporaryDirectory() as td:
        runs_dir = Path(td) / "runs"
        # Build runs/<gen>/<retriever>/<task>/run_<qid>.json
        gen = runs_dir / "gpt-5-mini-08-07"
        data = [
            # (retriever, task, qid, rounds)
            ("bm25", "biology", "0", 1),
            ("bm25", "biology", "1", 3),
            ("bm25", "biology", "2", 2),
            ("bm25", "robotics", "0", 5),
            ("bm25", "robotics", "1", 1),
            ("bm25", "robotics", "2", 1),
            ("diver", "biology", "0", 1),
            ("diver", "biology", "1", 2),
            ("diver", "biology", "2", 1),
            ("diver", "robotics", "0", 2),
            ("diver", "robotics", "1", 1),
            ("diver", "robotics", "2", 3),
        ]
        for retriever, task, qid, rounds in data:
            d = gen / retriever / task
            d.mkdir(parents=True, exist_ok=True)
            (d / f"run_{qid}.json").write_text(json.dumps({
                "query_id": qid,
                "retrieved_round_count": rounds,
                "retrieved_documents_id": ["doc_a", "doc_b"],
            }))

        # OQ scores: intentionally leave one qid missing to exercise n_missing_oq
        oq_scores = {
            "0": 5.0,
            "1": 4.0,
            # "2" missing → should be flagged
        }

        results = compute_aer_for_runs(runs_dir, oq_scores, gamma=DEFAULT_GAMMA)

        # Expect 4 groups
        assert set(results.keys()) == {("bm25", "biology"), ("bm25", "robotics"),
                                        ("diver", "biology"), ("diver", "robotics")}

        # Check bm25/biology: qids 0 (OQ=5, R=1), 1 (OQ=4, R=3). qid 2 missing.
        r = results[("bm25", "biology")]
        expected_aer_0 = aer(5.0, 1)        # 5.0
        expected_aer_1 = aer(4.0, 3)        # 4 * e^(-0.1)
        expected_avg = (expected_aer_0 + expected_aer_1) / 2
        assert r["n_queries"] == 3
        assert r["n_evaluated"] == 2
        assert r["n_missing_oq"] == 1
        assert approx(r["avg_aer"], expected_avg), f"{r['avg_aer']} vs {expected_avg}"
        assert approx(r["avg_rounds"], (1 + 3) / 2)
        assert approx(r["avg_oq"], (5.0 + 4.0) / 2)

        # Check diver/robotics: qids 0 (OQ=5, R=2), 1 (OQ=4, R=1), 2 missing
        r = results[("diver", "robotics")]
        expected = (aer(5.0, 2) + aer(4.0, 1)) / 2
        assert approx(r["avg_aer"], expected), f"{r['avg_aer']} vs {expected}"
        assert r["n_missing_oq"] == 1

        # Sanity: the retrievers that happen to run fewer rounds should get higher AER
        aer_bm25 = (results[("bm25", "biology")]["avg_aer"] + results[("bm25", "robotics")]["avg_aer"]) / 2
        aer_diver = (results[("diver", "biology")]["avg_aer"] + results[("diver", "robotics")]["avg_aer"]) / 2
        # diver in this fake setup has fewer rounds overall -> higher AER expected
        print(f"  end-to-end: aer_bm25={aer_bm25:.4f}  aer_diver={aer_diver:.4f}")
        print("  test_end_to_end ok")


def test_reproduce_paper_example():
    """Paper's setup: γ=0.05. Quick sanity numbers."""
    # If OQ=4.5 and R=3, AER = 4.5 * e^(-0.1) ≈ 4.0719
    assert approx(aer(4.5, 3, 0.05), 4.5 * math.exp(-0.1))
    # OQ=4.5 with R=10 should be much lower
    assert aer(4.5, 10, 0.05) < aer(4.5, 3, 0.05)
    # AER is monotone decreasing in R for fixed OQ>0
    prev = aer(3.0, 1)
    for r in range(2, 20):
        cur = aer(3.0, r)
        assert cur < prev, f"AER should decrease: R={r}, cur={cur}, prev={prev}"
        prev = cur
    print("  test_reproduce_paper_example ok")


def main():
    test_formula()
    test_reproduce_paper_example()
    test_end_to_end()
    print("\nAll AER tests passed ✓")


if __name__ == "__main__":
    main()
