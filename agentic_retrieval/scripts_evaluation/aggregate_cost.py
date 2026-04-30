#!/usr/bin/env python3
"""Aggregate token usage + USD cost across agentic-run JSON files.

Reads every `results/agentic_*/runs/<retriever>/<task>/run_*.json` (or a
user-supplied glob), sums `usage.{input_tokens, output_tokens,
input_tokens_cached, included_reasoning_tokens}`, and prints a per-(retriever,
task) table plus grand total.

Model pricing (USD / 1M tokens) — update if OpenAI revises:
  gpt-5-mini  input=$0.250  cached=$0.025  output=$2.000
  gpt-5       input=$1.250  cached=$0.125  output=$10.000
"""
from __future__ import annotations
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

PRICING = {
    "gpt-5-mini":   {"input": 0.25,  "cached": 0.025, "output": 2.00},
    "gpt-5":        {"input": 1.25,  "cached": 0.125, "output": 10.00},
    # fallback for unknown
    "_default":     {"input": 0.25,  "cached": 0.025, "output": 2.00},
}


def cost_for(model: str, usage: dict) -> float:
    prices = PRICING.get(model, PRICING["_default"])
    in_tok = usage.get("input_tokens", 0) - usage.get("input_tokens_cached", 0)
    cached = usage.get("input_tokens_cached", 0)
    out_tok = usage.get("output_tokens", 0)
    return (in_tok * prices["input"] + cached * prices["cached"] + out_tok * prices["output"]) / 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(Path(__file__).resolve().parents[2] /
                                          "results/agentic_*/runs/*/*/*/run_*.json"),
                    help="JSON glob (default: all fixed-round agentic + adaptive-round agentic runs, "
                         "layout: <retriever>/<benchmark>/<task>/run_*.json)")
    args = ap.parse_args()

    by_pair: dict = defaultdict(lambda: {"in": 0, "cached": 0, "out": 0,
                                         "reasoning": 0, "n": 0, "usd": 0.0,
                                         "model": None})
    for p in sorted(glob.glob(args.glob)):
        try:
            d = json.loads(Path(p).read_text())
        except Exception as e:
            print(f"  [skip] {p}: {e}")
            continue
        u = d.get("usage", {})
        model = d.get("metadata", {}).get("model", "?")
        # path layout: .../runs/<retriever>/<benchmark>/<task>/run_*.json
        parts = Path(p).parts
        retriever, task = parts[-4], parts[-2]
        k = (retriever, task)
        by_pair[k]["in"] += u.get("input_tokens", 0)
        by_pair[k]["cached"] += u.get("input_tokens_cached", 0)
        by_pair[k]["out"] += u.get("output_tokens", 0)
        by_pair[k]["reasoning"] += u.get("included_reasoning_tokens", 0)
        by_pair[k]["n"] += 1
        by_pair[k]["usd"] += cost_for(model, u)
        by_pair[k]["model"] = model

    if not by_pair:
        print("no runs found (glob matched 0 files)")
        return

    # Grand totals
    tot_in = tot_cached = tot_out = tot_reasoning = tot_n = 0
    tot_usd = 0.0
    print(f"{'retriever':<22} {'task':<20} {'n':>4} {'input':>10} {'cached':>9} {'output':>10} {'usd':>8}  model")
    for (r, t), v in sorted(by_pair.items()):
        tot_in += v['in']; tot_cached += v['cached']; tot_out += v['out']
        tot_reasoning += v['reasoning']; tot_n += v['n']; tot_usd += v['usd']
        print(f"{r:<22} {t:<20} {v['n']:>4} {v['in']:>10,} {v['cached']:>9,} "
              f"{v['out']:>10,} ${v['usd']:>7.3f}  {v['model']}")
    print(f"\n  TOTAL:  n={tot_n}  input={tot_in:,}  cached={tot_cached:,}  "
          f"output={tot_out:,}  reasoning={tot_reasoning:,}")
    print(f"  TOTAL USD: ${tot_usd:,.4f}")
    # Per-query avg
    if tot_n:
        print(f"  avg usd/query: ${tot_usd/tot_n:.4f}")


if __name__ == "__main__":
    main()
