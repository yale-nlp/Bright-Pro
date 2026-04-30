#!/usr/bin/env python3
"""Efficiency-Quality Reward (AER) — paper Eq.(6).

    AER(q) = OQ(q) · exp(-γ · (R(q) - 1))

where:
  - OQ(q) : LLM-as-Judge overall-quality score for query q (Likert 1-5)
  - R(q)  : number of retrieval rounds the agent executed for q
  - γ     : decay parameter (paper uses 0.05)

The per-query AER is averaged across a run to report a single number per
(model, task) or per model.

Inputs
------
* Agent run files (one JSON per query) written by
  `agentic_retrieval/search_agent/openai_new.py`, under
  `runs/<generator>/<retriever>/<task>/run_*.json`. Each contains at minimum:
      { "query_id": "...", "retrieved_round_count": <int>, ... }
* OQ scores JSON — format `{ "<query_id>": <float>, ... }` — produced by the
  (separate) LLM-as-Judge scoring step. Pass via `--oq-json`.

Outputs
-------
* Per-(retriever, task) mean AER.
* Overall mean across tasks per retriever.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_GAMMA = 0.05


def aer(oq: float, rounds: int, gamma: float = DEFAULT_GAMMA) -> float:
    """Compute AER for a single query (paper Eq.(6))."""
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    return float(oq) * math.exp(-gamma * (rounds - 1))


def load_oq_scores(path: Path) -> Dict[str, float]:
    """OQ scores file is a flat dict {qid: score}."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected dict {{qid: oq}}, got {type(raw).__name__}")
    return {str(k): float(v) for k, v in raw.items()}


def load_oq_from_judge_jsonl(path: Path) -> Dict[Tuple[str, str, Optional[str]], float]:
    """Read judge.jsonl and return {(task, qid, retriever): mean_OQ}.

    If multiple reps exist for the same (task, qid, retriever), average them.
    Adaptive runs ignore round; fixed-turn runs are aggregated across rounds
    via the *latest* round's score (max round).
    """
    by_key: Dict[Tuple[str, str, Optional[str]], List[Tuple[Optional[int], float]]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "ok":
            continue
        oq = r.get("overall_quality")
        if oq is None:
            continue
        task = r["task"]; qid = str(r["qid"])
        retr = r.get("retriever")
        rnd = r.get("round")
        by_key[(task, qid, retr)].append((rnd, float(oq)))

    out: Dict[Tuple[str, str, Optional[str]], float] = {}
    for k, vals in by_key.items():
        # If any rounds present, take max-round; else average all
        rounds_present = [r for r, _ in vals if r is not None]
        if rounds_present:
            max_r = max(rounds_present)
            picked = [v for r, v in vals if r == max_r]
            out[k] = sum(picked) / len(picked)
        else:
            out[k] = sum(v for _, v in vals) / len(vals)
    return out


_SE_TASKS = {
    "biology", "earth_science", "economics", "psychology",
    "robotics", "stackoverflow", "sustainable_living",
    "leetcode", "pony", "theoremqa_questions", "theoremqa_theorems", "aops",
}


def iter_run_files(runs_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    """Yield (retriever, task, run_file). Handles all three layouts:
      - runs_dir/<retriever>/<task>/run_*.json
      - runs_dir/<retriever>/<benchmark>/<task>/run_*.json   (e.g. bright-pro)
      - runs_dir/<generator>/<retriever>/<task>/run_*.json   (legacy)
    """
    for retriever_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for child in sorted(p for p in retriever_dir.iterdir() if p.is_dir()):
            if child.name in _SE_TASKS:
                for rf in sorted(child.glob("run_*.json")):
                    yield retriever_dir.name, child.name, rf
            else:
                # benchmark or generator level
                for task_dir in sorted(p for p in child.iterdir() if p.is_dir()):
                    if task_dir.name in _SE_TASKS:
                        for rf in sorted(task_dir.glob("run_*.json")):
                            yield retriever_dir.name, task_dir.name, rf


def parse_qid_and_rounds(run_file: Path) -> Tuple[Optional[str], Optional[int]]:
    try:
        data = json.loads(run_file.read_text())
    except Exception:
        return None, None
    qid = data.get("query_id")
    rounds = data.get("retrieved_round_count")
    if qid is None or rounds is None:
        return None, None
    try:
        return str(qid), int(rounds)
    except Exception:
        return None, None


def compute_aer_for_runs(
    runs_dir: Path,
    oq_lookup,
    gamma: float = DEFAULT_GAMMA,
) -> Dict[Tuple[str, str], Dict]:
    """Return {(retriever, task): summary_dict}.

    `oq_lookup` may be either:
      - dict[str, float]                     keyed by qid (legacy)
      - dict[(task, qid, retriever), float]  keyed by judge tuple (preferred)
    """
    is_tuple_keyed = bool(oq_lookup) and isinstance(next(iter(oq_lookup)), tuple)

    grouped: Dict[Tuple[str, str], List[Tuple[str, int]]] = defaultdict(list)
    for retriever, task, rf in iter_run_files(runs_dir):
        qid, rounds = parse_qid_and_rounds(rf)
        if qid is None:
            continue
        grouped[(retriever, task)].append((qid, rounds))

    out: Dict[Tuple[str, str], Dict] = {}
    for (retriever, task), items in grouped.items():
        aer_vals: List[float] = []
        round_vals: List[int] = []
        oq_vals: List[float] = []
        n_missing = 0
        for qid, rounds in items:
            if is_tuple_keyed:
                key = (task, qid, retriever)
                if key not in oq_lookup:
                    # try retriever-agnostic fallback
                    key = (task, qid, None)
                if key not in oq_lookup:
                    n_missing += 1
                    continue
                oq = oq_lookup[key]
            else:
                if qid not in oq_lookup:
                    n_missing += 1
                    continue
                oq = oq_lookup[qid]
            aer_vals.append(aer(oq, rounds, gamma))
            round_vals.append(rounds)
            oq_vals.append(oq)
        out[(retriever, task)] = {
            "retriever": retriever,
            "task": task,
            "n_queries": len(items),
            "n_evaluated": len(aer_vals),
            "n_missing_oq": n_missing,
            "avg_aer": (sum(aer_vals) / len(aer_vals)) if aer_vals else 0.0,
            "avg_rounds": (sum(round_vals) / len(round_vals)) if round_vals else 0.0,
            "avg_oq": (sum(oq_vals) / len(oq_vals)) if oq_vals else 0.0,
        }
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AER (paper Eq.(6)) from agent run files and OQ scores.")
    parser.add_argument("--runs-dir", type=Path, required=True,
                        help="path to runs/ dir (expects runs/<retriever>/[<benchmark>/]<task>/run_*.json)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--oq-json", type=Path,
                     help="JSON file: {qid: oq_score}  (from LLM-as-Judge)")
    src.add_argument("--judge-jsonl", type=Path,
                     help="judge.jsonl from judge.py; OQ keyed by (task, qid, retriever)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--save-json", type=Path, default=None,
                        help="optional path to write aggregate results as JSON")
    args = parser.parse_args()

    if args.judge_jsonl:
        oq_scores = load_oq_from_judge_jsonl(args.judge_jsonl)
    else:
        oq_scores = load_oq_scores(args.oq_json)
    results = compute_aer_for_runs(args.runs_dir, oq_scores, gamma=args.gamma)

    # Pretty print
    header = f"{'retriever':20s}  {'task':22s}  {'n_ok/n':>8s}  {'avg_R':>6s}  {'avg_OQ':>7s}  {'avg_AER':>8s}"
    print(header)
    print("-" * len(header))
    by_retriever: Dict[str, List[Dict]] = defaultdict(list)
    for key in sorted(results):
        r = results[key]
        by_retriever[r["retriever"]].append(r)
        print(
            f"{r['retriever']:20s}  {r['task']:22s}  "
            f"{r['n_evaluated']:>3d}/{r['n_queries']:<3d}  "
            f"{r['avg_rounds']:>6.2f}  {r['avg_oq']:>7.3f}  {r['avg_aer']:>8.4f}"
        )
    print("-" * len(header))
    for retriever, rows in by_retriever.items():
        denom = sum(r["n_evaluated"] for r in rows)
        if denom == 0:
            continue
        # Micro-average: per-query values across tasks
        weighted_aer = sum(r["avg_aer"] * r["n_evaluated"] for r in rows) / denom
        weighted_R = sum(r["avg_rounds"] * r["n_evaluated"] for r in rows) / denom
        weighted_OQ = sum(r["avg_oq"] * r["n_evaluated"] for r in rows) / denom
        print(
            f"{retriever:20s}  {'<OVERALL>':22s}  "
            f"{denom:>3d}/{sum(r['n_queries'] for r in rows):<3d}  "
            f"{weighted_R:>6.2f}  {weighted_OQ:>7.3f}  {weighted_aer:>8.4f}"
        )

    if args.save_json:
        payload = {
            "gamma": args.gamma,
            "rows": [
                {**r, "_key": f"{r['retriever']}::{r['task']}"}
                for r in results.values()
            ],
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.save_json}")


if __name__ == "__main__":
    main()
