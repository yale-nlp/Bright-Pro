#!/usr/bin/env python3
"""NDCG@25 / Recall@25 / MAP@25 evaluator for agent run files.

This replaces the earlier AUC-NDCG implementation: the appendix
reports NDCG@25 (single cut-off), and `retrieval/metrics.py` already
has the BEIR-style `calculate_retrieval_metrics()` function that computes
the full ladder of metrics. This script:

  1. Walks `runs/<generator>/<retriever>/<task>/run_*.json`.
  2. For each run file, parses the ranked list `retrieved_documents_id`
     and assigns decreasing synthetic scores (so earlier rank → higher
     score, preserving order).
  3. Builds `qrels` from BRIGHT-PRO `gold_ids`.
  4. Calls `calculate_retrieval_metrics` with `k_values=[1, 5, 10, 25, 50, 100]`
     and reports the averages.

CLI preserves the existing `--model --task` ergonomics. Old `--per-file`
becomes a no-op (per-run metrics are available via `--save-json`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys as _sys_bp
from typing import Dict, List, Optional, Set, Tuple

# Bootstrap bright_pro_data (walk up to BRIGHT-PRO/)
_bp = os.path.dirname(os.path.abspath(__file__))
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS  # noqa: E402

# retrieval/metrics.py is two levels up (BRIGHT-PRO/retrieval/metrics.py)
_RETRIEVAL_DIR = os.path.join(_bp, "retrieval")
if _RETRIEVAL_DIR not in _sys_bp.path:
    _sys_bp.path.insert(0, _RETRIEVAL_DIR)
from metrics import calculate_retrieval_metrics  # noqa: E402


DEFAULT_K_VALUES = [1, 5, 10, 25, 50, 100]


def load_gold_from_dataset(task: str, cache_dir: Optional[str] = None) -> Dict[str, Set[str]]:
    """{qid: {gold_doc_ids}} from BRIGHT-PRO examples. cache_dir unused (local JSON)."""
    gold_map: Dict[str, Set[str]] = {}
    for ex in load_bright_pro("examples", task):
        gold_map[str(ex["id"])] = {str(d) for d in (ex.get("gold_ids") or [])}
    return gold_map


def build_qrels_and_scores(
    runs_dir: str, qrels_full: Dict[str, Set[str]]
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, float]], Dict]:
    """For every run_*.json under runs_dir, parse (query_id, retrieved_documents_id)
    into the (qrels, results) dict shape expected by calculate_retrieval_metrics.

    Returns (qrels, results, stats) where stats has counts used for printing.
    """
    run_paths = sorted(glob.glob(os.path.join(runs_dir, "run_*.json")))
    qrels: Dict[str, Dict[str, int]] = {}
    results: Dict[str, Dict[str, float]] = {}
    stats = {"files": 0, "files_with_ranked": 0, "files_with_qrels": 0,
             "files_empty_ranked": 0, "files_missing_qid": 0,
             "files_no_qrels": 0, "files_read_error": 0}
    for rp in run_paths:
        stats["files"] += 1
        try:
            with open(rp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            stats["files_read_error"] += 1
            continue

        qid = str(data.get("query_id", "")).strip()
        if not qid:
            stats["files_missing_qid"] += 1
            continue

        ranked_raw = data.get("retrieved_documents_id") or []
        ranked = [str(x).strip() for x in ranked_raw if x is not None]
        if not ranked:
            stats["files_empty_ranked"] += 1

        gold = qrels_full.get(qid, set())
        if not gold:
            stats["files_no_qrels"] += 1
            print(f"WARNING: no qrels for qid={qid!r} in {os.path.basename(rp)}")
            continue
        stats["files_with_qrels"] += 1
        if ranked:
            stats["files_with_ranked"] += 1

        # Assign decreasing synthetic scores to preserve the ranked order.
        # pytrec_eval ranks docs by score (desc), ties broken by doc-id, so
        # strictly decreasing floats suffice.
        n = len(ranked)
        qrels[qid] = {d: 1 for d in gold}
        results[qid] = {d: float(n - i) for i, d in enumerate(ranked)}

    return qrels, results, stats


def evaluate_runs(
    runs_dir: str,
    task: str,
    cache_dir: Optional[str] = None,
    k_values: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], Dict]:
    k_values = list(k_values or DEFAULT_K_VALUES)
    gold_map = load_gold_from_dataset(task=task, cache_dir=cache_dir)
    qrels, results, stats = build_qrels_and_scores(runs_dir, gold_map)
    if not results:
        return {}, stats
    metrics = calculate_retrieval_metrics(results, qrels, k_values=k_values)
    return metrics, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute NDCG@k / Recall@k / MAP@k / P@k / MRR over agent run files."
    )
    parser.add_argument("--runs-dir", type=str, default=None,
                        help="Directory containing run_*.json. If omitted, derives from --generator/--model/--task.")
    parser.add_argument("--model", type=str, default=None,
                        help="Retriever name (used to build runs/<generator>/<model>/<task>/ when --runs-dir is omitted).")
    parser.add_argument("--generator", type=str, default="gpt-5-mini-08-07",
                        help="Generator dir name (default %(default)s)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task; if omitted, iterate SE_TASKS")
    parser.add_argument("--cache-dir", type=str, default="cache", help="(kept for compat; unused — local JSON)")
    parser.add_argument("--k-values", type=str, default="1,5,10,25,50,100",
                        help="comma-separated list of k")
    parser.add_argument("--save-json", type=str, default=None,
                        help="optional path to write aggregate results")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    tasks_to_run = [args.task] if args.task else list(SE_TASKS)

    all_task_metrics: Dict[str, Dict[str, float]] = {}
    for task in tasks_to_run:
        if args.runs_dir:
            runs_dir = args.runs_dir if len(tasks_to_run) == 1 else os.path.join(args.runs_dir, task)
        else:
            if not args.model:
                parser.error("must pass either --runs-dir or --model")
            runs_dir = f"runs/{args.generator}/{args.model}/{task}"
        metrics, stats = evaluate_runs(runs_dir=runs_dir, task=task, cache_dir=args.cache_dir, k_values=k_values)
        all_task_metrics[task] = metrics
        print(f"\n[{task}] runs_dir={runs_dir}")
        print(f"  files={stats['files']}  with_qrels={stats['files_with_qrels']}  "
              f"empty_ranked={stats['files_empty_ranked']}  read_errors={stats['files_read_error']}")
        if not metrics:
            print("  no evaluable runs")
            continue
        for kn in k_values:
            print(f"  NDCG@{kn}={metrics[f'NDCG@{kn}']:.5f}  "
                  f"Recall@{kn}={metrics[f'Recall@{kn}']:.5f}  "
                  f"MAP@{kn}={metrics[f'MAP@{kn}']:.5f}  "
                  f"P@{kn}={metrics[f'P@{kn}']:.5f}")
        print(f"  MRR={metrics['MRR']:.5f}")

    # Overall averages
    if len(tasks_to_run) > 1:
        print("\n" + "=" * 60)
        print(f"OVERALL ({len(all_task_metrics)} tasks, macro-averaged)")
        print("=" * 60)
        keys = [f"NDCG@{k}" for k in k_values] + [f"Recall@{k}" for k in k_values] + ["MRR"]
        for key in keys:
            vals = [m.get(key, 0.0) for m in all_task_metrics.values() if m]
            if not vals:
                continue
            print(f"  {key}: {sum(vals) / len(vals):.5f}")

    if args.save_json:
        payload = {"per_task": all_task_metrics, "k_values": k_values}
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.save_json}")


if __name__ == "__main__":
    main()
