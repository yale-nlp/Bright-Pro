"""Sentence-transformers-based searcher for the Qwen3-Embedding family.

Covers:
  * Qwen/Qwen3-Embedding-8B            (``model_id='qwen3-embed'``)
  * AQ-MedAI/Diver-Retriever-4B        (``model_id='diver-retriever'``)
  * AQ-MedAI/Diver-Retriever-4B-1020   (``model_id='diver-1020'``)
  * BAAI/bge-reasoner-embed-qwen3-8b-0923 (``model_id='bge-reasoner'``)
  * yale-nlp/RTriever-4B               (``model_id='rtriever-4b'``)

Design goals:
  * Model / tokenizer stay on GPU across ``set_task()`` calls (only doc
    embeddings + per-task query instruction get swapped — zero model reload).
  * Reuses the embedding cache layout
    ``{cache}/doc_emb/<cache_name>/<task>/long_False/0.npy`` so we don't
    re-encode 500k docs across runs.
  * Query prefix follows the official Instruct:/Query: template.
"""
from __future__ import annotations

# Apply transformers 4.x/5.x compat shims for custom-code HF modeling files.
import sys as _sys_compat
import os as _os_compat
_compat_dir = _os_compat.path.dirname(_os_compat.path.abspath(__file__))
while _compat_dir != "/" and not _os_compat.path.isfile(_os_compat.path.join(_compat_dir, "retrieval", "_tf_compat.py")):
    _compat_dir = _os_compat.path.dirname(_compat_dir)
_sys_compat.path.insert(0, _os_compat.path.join(_compat_dir, "retrieval"))
import _tf_compat  # noqa: F401


import json
import logging
import os
import os.path
import sys as _sys
from typing import Any, Dict, Optional

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_bp = _here
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys.path:
    _sys.path.insert(0, _bp)
# Also add retrieval/ for Qwen3EmbeddingModel
_ret = os.path.join(_bp, "retrieval")
if _ret not in _sys.path:
    _sys.path.insert(0, _ret)

from bright_pro_data import load_bright_pro  # noqa: E402
from qwen3_embedding import Qwen3EmbeddingModel  # noqa: E402

from .base import BaseSearcher

logger = logging.getLogger(__name__)

# (cli model_id) -> (HF repo / local path, config dir under agentic_retrieval/configs/)
MODEL_REGISTRY = {
    "qwen3-embed": ("Qwen/Qwen3-Embedding-8B", "qwen3-embed"),
    "diver-retriever": ("AQ-MedAI/Diver-Retriever-4B", "diver-retriever"),
    "diver-1020": ("AQ-MedAI/Diver-Retriever-4B-1020", "diver-1020"),
    "bge-reasoner": ("BAAI/bge-reasoner-embed-qwen3-8b-0923", "bge-reasoner"),
    "rtriever-4b": ("yale-nlp/RTriever-4B", "rtriever-4b"),
}


class Qwen3FamilySearcher(BaseSearcher):
    """One searcher class, multi-model via ``args.qwen3_model_id``."""

    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--qwen3-model-id", type=str, required=True,
            choices=sorted(MODEL_REGISTRY.keys()),
            help="Which Qwen3-family model to use",
        )
        parser.add_argument(
            "--qwen3-batch-size", type=int, default=8,
            help="Batch size for ST encode() (docs; queries encoded one-at-a-time during search)",
        )
        parser.add_argument(
            "--qwen3-max-length", type=int, default=4096,
            help="Max sequence length (post-tokenizer truncation)",
        )
        parser.add_argument(
            "--qwen3-ignore-cache", action="store_true",
            help="Ignore static-eval doc-emb cache and re-encode from scratch.",
        )

    def __init__(self, args):
        self.args = args
        self.cli_id = args.qwen3_model_id
        self.model_path, self.config_subdir = MODEL_REGISTRY[self.cli_id]
        self.batch_size = args.qwen3_batch_size
        self.max_length = args.qwen3_max_length

        # Load model ONCE; stays on GPU for the lifetime of the Python process.
        logger.info(f"Loading {self.cli_id} ({self.model_path}) on GPU...")
        model_cache = getattr(args, "model_cache_dir", None)
        self.model = Qwen3EmbeddingModel(
            self.model_path, max_length=self.max_length,
            cache_dir=model_cache, batch_size=self.batch_size,
        )
        logger.info(f"{self.cli_id} loaded.")

        # Per-task state. Populated by set_task() / the constructor's initial
        # set_task call if args.task is set.
        self.task = None
        self.doc_ids = None
        self.documents = None
        self.doc_emb = None
        self.query_prefix = ""

        if getattr(args, "task", None):
            self.set_task(args.task)

    # ---- task swap (cheap: ~1-5 s) -----------------------------------------
    def set_task(self, task: str) -> None:
        logger.info(f"[Qwen3FamilySearcher] set_task({task}) for {self.cli_id}")
        self.task = task

        # Corpus
        doc_pairs = load_bright_pro("documents", task)
        self.doc_ids = [dp["id"] for dp in doc_pairs]
        self.documents = [dp["content"] for dp in doc_pairs]

        # Per-task instruction (query prefix). Prefer agentic config; fall back
        # to retrieval/configs/<config_subdir>/<task>.json (same instructions).
        self.query_prefix = self._resolve_query_prefix(task)

        # Doc embeddings — try static-eval cache locations first.
        self.doc_emb = self._load_or_compute_doc_emb(task)

    # ---- search (zero-reload) ----------------------------------------------
    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if not query:
            return []
        # Encode query using model-specific prefix
        wrapped = f"{self.query_prefix}{query}"
        q_emb = self.model.model.encode(
            [wrapped], batch_size=1,
            convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        # Cosine similarity against cached doc_emb
        scores = self.doc_emb @ q_emb  # (N,) — embeddings are normalized
        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores.tolist())

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        try:
            idx = self.doc_ids.index(docid)
            return {"docid": docid, "text": self.documents[idx]}
        except ValueError:
            return None

    @property
    def search_type(self) -> str:
        return self.cli_id

    def search_description(self, k: int = 10) -> str:
        return (f"Perform a dense {self.cli_id} search on the document collection. "
                f"Returns top-{k} hits with docid, score, and snippet.")

    # ---- internals ---------------------------------------------------------
    def _resolve_query_prefix(self, task: str) -> str:
        """Load per-task instructions from agentic config (and fall back to
        the static-eval retrieval config if not found)."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        for cfg_dir in (os.path.join(project_root, "configs", self.config_subdir),
                        os.path.join(_bp, "retrieval", "configs", self.config_subdir)):
            p = os.path.join(cfg_dir, f"{task}.json")
            if os.path.isfile(p):
                with open(p) as f:
                    cfg = json.load(f)
                q = cfg.get("instructions", {}).get("query", "") or ""
                if q:
                    return q.format(task=task)
        # Default fallback — paper instruction for Qwen3 family
        return (f"Instruct: Given a {task} post, retrieve relevant passages "
                f"that help answer the post\nQuery:")

    def _load_or_compute_doc_emb(self, task: str) -> np.ndarray:
        """Try static-eval cache first (several layouts), then fall back to encoding."""
        cache_root = (getattr(self.args, "cache_dir", None)
                      or os.environ.get("BRIGHT_PRO_CACHE_DIR")
                      or os.path.join(os.getcwd(), "cache", "retrieval"))
        candidates = [
            # Static-eval cache layout
            os.path.join(cache_root, "doc_emb", self.config_subdir, task, "long_False", "0.npy"),
            os.path.join(cache_root, "doc_emb", self.config_subdir, task, "0.npy"),
            # Per-batch fallback
            os.path.join(cache_root, "doc_emb", self.config_subdir, task,
                         f"long_False_{self.batch_size}.npy"),
        ]
        if not getattr(self.args, "qwen3_ignore_cache", False):
            for p in candidates:
                if os.path.isfile(p):
                    emb = np.load(p, allow_pickle=True)
                    if emb.shape[0] == len(self.doc_ids):
                        logger.info(f"[Qwen3FamilySearcher] loaded doc emb from cache {p}")
                        return emb.astype(np.float32, copy=False)
                    logger.warning(
                        f"cache {p} has shape {emb.shape} but corpus has "
                        f"{len(self.doc_ids)} docs; ignoring")
        else:
            logger.info(f"[Qwen3FamilySearcher] --qwen3-ignore-cache set; re-encoding")

        # Fresh encode
        out = candidates[0]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        logger.info(f"[Qwen3FamilySearcher] no cache hit — encoding {len(self.documents)} docs "
                    f"and saving to {out}")
        embs = self.model.embed_docs(self.documents, doc_prefix="")
        arr = np.asarray(embs, dtype=np.float32)
        # Safety: renormalize just in case
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-12, None)
        np.save(out, arr)
        return arr
