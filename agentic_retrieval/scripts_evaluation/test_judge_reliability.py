#!/usr/bin/env python3
"""Reliability / validity check of the judge.

For a small sample of biology queries, we synthesize three tiers of "system
answers" with known expected quality, then judge each multiple times:

  gold       = the reference answer verbatim (should score ~5 / 5)
  truncated  = first ~40% of the reference answer (should score 3-4)
  off_topic  = a generic platitude unrelated to the question (should score ~1)

Two properties we want to observe:
  1) Validity (rank-ordering): mean(gold) > mean(truncated) > mean(off_topic)
     — on both reasoning_completeness and overall_quality.
  2) Reliability (repeatability): average σ across reps should be small
     (target σ < 0.5 on each metric).

Budget: N_queries × 3 tiers × N_reps judge calls. Default 10 × 3 × 3 = 90 calls.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean, pstdev

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # BRIGHT-PRO/ for bright_pro_data

from bright_pro_data import load_bright_pro

OFF_TOPIC_TEMPLATES = [
    "That's an interesting question. There are many factors involved and "
    "people often disagree about the best answer. It's worth thinking "
    "carefully about the topic. [doc_0]",
    "In general, the answer depends on context and specific circumstances. "
    "Experts have debated this for a long time. [doc_0]",
    "The correct answer to this question varies. I would recommend checking "
    "with a domain specialist. [doc_0]",
]


def build_synthetic_answers_jsonl(task: str, n_queries: int, out_path: Path):
    ex = load_bright_pro("examples", task)
    rows = []
    import random
    rng = random.Random(0)
    for ex_row in ex[:n_queries]:
        qid = str(ex_row["id"])
        ref = (ex_row.get("reference_answer") or "").strip()
        if not ref or len(ref.split()) < 40:
            continue
        # gold — full reference answer
        rows.append({"task": task, "qid": qid, "round": None, "retriever": "gold", "answer": ref})
        # truncated — first ~40% of words
        words = ref.split()
        cut = max(20, int(0.4 * len(words)))
        truncated = " ".join(words[:cut]) + " ..."
        rows.append({"task": task, "qid": qid, "round": None, "retriever": "truncated", "answer": truncated})
        # off-topic platitude
        rows.append({"task": task, "qid": qid, "round": None, "retriever": "off_topic",
                     "answer": rng.choice(OFF_TOPIC_TEMPLATES)})

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def analyze(out_jsonl: Path):
    """Group by retriever tier (gold/truncated/off_topic). Report mean ± σ
    for each metric, and the per-group reliability (σ across reps within a
    (qid, tier)). Return True if validity ordering holds."""
    rows = [json.loads(ln) for ln in out_jsonl.read_text().splitlines() if ln.strip()]
    rows = [r for r in rows if r["status"] == "ok"]
    print(f"\n== successful judgments: {len(rows)} ==\n")

    by_tier: dict = {"gold": [], "truncated": [], "off_topic": []}
    by_qid_tier: dict = {}
    for r in rows:
        tier = r.get("retriever") or "unknown"
        by_tier.setdefault(tier, []).append(r)
        by_qid_tier.setdefault((r["qid"], tier), []).append(r)

    print(f"{'tier':12s}  {'n':>4s}  {'RC mean±σ':>12s}  {'OQ mean±σ':>12s}")
    tier_means = {}
    for tier in ["gold", "truncated", "off_topic"]:
        rs = by_tier.get(tier, [])
        if not rs:
            continue
        rc = [x["reasoning_completeness"] for x in rs]
        oq = [x["overall_quality"] for x in rs]
        rc_m, rc_s = mean(rc), pstdev(rc) if len(rc) > 1 else 0.0
        oq_m, oq_s = mean(oq), pstdev(oq) if len(oq) > 1 else 0.0
        tier_means[tier] = (rc_m, oq_m)
        print(f"  {tier:10s}  {len(rs):>4d}  {rc_m:>5.2f}±{rc_s:4.2f}  {oq_m:>5.2f}±{oq_s:4.2f}")

    # Reliability: σ across reps within each (qid, tier)
    print("\n== repeatability (σ within each qid × tier) ==")
    rc_sigmas, oq_sigmas = [], []
    for (qid, tier), rs in by_qid_tier.items():
        if len(rs) <= 1:
            continue
        rc = [x["reasoning_completeness"] for x in rs]
        oq = [x["overall_quality"] for x in rs]
        rc_sigmas.append(pstdev(rc))
        oq_sigmas.append(pstdev(oq))
    if rc_sigmas:
        print(f"  avg σ(reasoning_completeness) = {mean(rc_sigmas):.3f}   "
              f"max σ = {max(rc_sigmas):.2f}   (n groups = {len(rc_sigmas)})")
        print(f"  avg σ(overall_quality)        = {mean(oq_sigmas):.3f}   "
              f"max σ = {max(oq_sigmas):.2f}   (n groups = {len(oq_sigmas)})")

    # Validity ordering
    print("\n== validity (mean rank ordering) ==")
    if {"gold", "truncated", "off_topic"}.issubset(tier_means.keys()):
        g = tier_means["gold"]
        t = tier_means["truncated"]
        o = tier_means["off_topic"]
        rc_ok = g[0] > t[0] > o[0]
        oq_ok = g[1] > t[1] > o[1]
        print(f"  RC: gold({g[0]:.2f}) > truncated({t[0]:.2f}) > off_topic({o[0]:.2f})  {'✓' if rc_ok else '✗'}")
        print(f"  OQ: gold({g[1]:.2f}) > truncated({t[1]:.2f}) > off_topic({o[1]:.2f})  {'✓' if oq_ok else '✗'}")
        return rc_ok and oq_ok
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="biology")
    ap.add_argument("--n-queries", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/judge_reliability"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / "synthetic_answers.jsonl"
    results = args.out_dir / "judge_scores.jsonl"
    if results.exists():
        results.unlink()
    rows = build_synthetic_answers_jsonl(args.task, args.n_queries, jsonl)
    print(f"Built {len(rows)} synthetic answers in {jsonl}")
    print(f"  ({len(rows)//3} queries × 3 tiers × {args.reps} reps = {len(rows) * args.reps} judge calls)")

    cmd = [
        "python3", str(HERE / "judge.py"),
        "--answers-jsonl", str(jsonl),
        "--save-jsonl", str(results),
        "--reps", str(args.reps),
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print("judge.py failed")
        sys.exit(rc)

    ok = analyze(results)
    print(f"\n{'✓ PASS' if ok else '✗ FAIL'}")


if __name__ == "__main__":
    main()
