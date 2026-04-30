"""Loader for the Bright-Pro benchmark.

Default source is the HuggingFace dataset ``yale-nlp/Bright-Pro``. To use a
local mirror (e.g. for offline runs), set the ``BRIGHT_PRO_DATA_ROOT``
environment variable to a directory laid out as::

    <root>/
        examples/<task>.json
        documents/<task>.json
        aspects/<task>.json

Usage:
    from bright_pro_data import load_bright_pro, SE_TASKS
    docs = load_bright_pro("documents", "biology")    # list[dict]
    for ex in load_bright_pro("examples", "biology"):
        ...
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

# 7 StackExchange subsets reported in the paper.
SE_TASKS: List[str] = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
]

_VALID_CONFIGS = ("examples", "documents", "aspects")
_HF_REPO = "yale-nlp/Bright-Pro"


def _local_root() -> Path | None:
    raw = os.environ.get("BRIGHT_PRO_DATA_ROOT")
    return Path(raw) if raw else None


@lru_cache(maxsize=None)
def _load_from_hf(config: str, task: str) -> tuple:
    from datasets import load_dataset
    ds = load_dataset(_HF_REPO, config, split=task)
    return tuple(dict(row) for row in ds)


def _load_from_local(root: Path, config: str, task: str) -> List[Dict]:
    path = root / config / f"{task}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_bright_pro(config: str, task: str) -> List[Dict]:
    """Load a list of rows for (config, task)."""
    if config not in _VALID_CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {_VALID_CONFIGS}")
    if task not in SE_TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {SE_TASKS}")
    root = _local_root()
    if root is not None:
        return _load_from_local(root, config, task)
    return list(_load_from_hf(config, task))


def load_bright_pro_all(config: str) -> Dict[str, List[Dict]]:
    """Load all 7 SE tasks for a given config."""
    return {task: load_bright_pro(config, task) for task in SE_TASKS}


def build_doc_to_aspect_id(task: str) -> Dict[str, str]:
    """Return ``{doc_id: aspect_id}`` for every gold doc in ``task``.

    Uses each aspect's ``supporting_docs`` list as the authoritative mapping.
    Non-gold documents are absent — use ``.get(doc_id)`` to look up.
    """
    aspects = load_bright_pro("aspects", task)
    mapping: Dict[str, str] = {}
    for a in aspects:
        for d in a.get("supporting_docs", []) or []:
            mapping[str(d)] = str(a["id"])
    return mapping


def build_aspect_weights(task: str) -> Dict[str, float]:
    """Return ``{aspect_id: normalized_weight}`` with Σ weights = 1 per query.

    Aspects store the raw Likert weight ∈ {1, 2, 3}; metric code expects
    per-query probabilities, so we normalize at read time.
    """
    import re
    aspects = load_bright_pro("aspects", task)
    suffix_re = re.compile(r"-(?:a\d+|aspect-\d+)$")
    per_query_sum: Dict[str, float] = {}
    aspect_to_stem: Dict[str, str] = {}
    aspect_raw: Dict[str, float] = {}
    for a in aspects:
        aid = str(a["id"])
        stem = suffix_re.sub("", aid)
        w = float(a["weight"])
        aspect_to_stem[aid] = stem
        aspect_raw[aid] = w
        per_query_sum[stem] = per_query_sum.get(stem, 0.0) + w
    out: Dict[str, float] = {}
    for aid, w in aspect_raw.items():
        total = per_query_sum[aspect_to_stem[aid]]
        out[aid] = (w / total) if total > 0 else 0.0
    return out
