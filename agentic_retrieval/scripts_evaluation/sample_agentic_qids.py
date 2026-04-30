#!/usr/bin/env python3
"""Fix the 25 queries per task that all agentic-retrieval runs will share.

Run ONCE — writes `agentic_sample_ids.json` at the repo root mapping
`{task: [qid, qid, ...]}` (25 qids per task, seed=42). Every agentic run
reads this file so every retriever is scored on the exact same subset.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BPRO = HERE.parent.parent
sys.path.insert(0, str(BPRO))
from bright_pro_data import load_bright_pro, SE_TASKS  # noqa: E402

SEED = 42
N_PER_TASK = 25
OUT = BPRO / "agentic_sample_ids.json"


def main():
    rng = random.Random(SEED)
    samples: dict[str, list] = {}
    for task in SE_TASKS:
        ex = load_bright_pro('examples', task)
        qids = [str(e['id']) for e in ex]
        if len(qids) <= N_PER_TASK:
            samples[task] = qids
        else:
            samples[task] = sorted(rng.sample(qids, N_PER_TASK),
                                   key=lambda s: (len(s), s))
        print(f"  {task:22s}  pool={len(qids):3d}  picked={len(samples[task])}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "n_per_task": N_PER_TASK,
                               "tasks": samples}, indent=2))
    print(f"\nwrote {OUT}  ({sum(len(v) for v in samples.values())} total qids)")


if __name__ == "__main__":
    main()
