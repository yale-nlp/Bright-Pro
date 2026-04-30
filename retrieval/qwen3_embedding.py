"""Wrapper for Qwen3-Embedding family (incl. AQ-MedAI/Diver-Retriever-4B).

Extracted from `retrievers.py` so other modules can import this class without
pulling in the rest of the retriever backends.

Official convention (Qwen3-Embedding model card, DIVER-Retriever-4B model card):
  * Queries are wrapped as ``f"Instruct: {task_description}\\nQuery:{q}"``.
  * Documents are passed raw (no prefix).
  * Tokenizer truncation is applied AFTER the wrapper is prepended, so the
    prompt is never accidentally truncated away.

The wrapper uses ``sentence-transformers`` (HF model card shows this is the
officially supported path). We intentionally avoid pinning to vLLM because
(a) it doesn't give us model-card-correct pooling for free, and (b) the vLLM
pinned torch/CUDA combo isn't always available on the target GPU generation
(e.g. RTX PRO 6000 Blackwell, SM 12.0, needs cu128).
"""
from __future__ import annotations

import os


class Qwen3EmbeddingModel:
    def __init__(self, model_path: str, max_length: int = 16384, cache_dir=None,
                 batch_size: int = 8):
        from sentence_transformers import SentenceTransformer
        import torch as _torch

        st_kwargs = {
            "trust_remote_code": True,
            "model_kwargs": {"torch_dtype": _torch.bfloat16, "device_map": "auto"},
        }
        if cache_dir is not None:
            st_kwargs["cache_folder"] = cache_dir
        self.model = SentenceTransformer(model_path, **st_kwargs)
        self.model.max_seq_length = max_length
        self.max_length = max_length
        self.batch_size = batch_size
        # Causal LM encoders with last-token pooling require left-padding when
        # batch > 1 — the final position must always be the real last token.
        # ST defaults to whatever the tokenizer says; we force left here for
        # safety. No effect if the tokenizer was already left-padding.
        try:
            self.model.tokenizer.padding_side = 'left'
        except Exception:
            pass

    def _embed_with_prefix(self, texts, prefix: str):
        wrapped = [f"{prefix}{t}" for t in texts]
        embs = self.model.encode(
            wrapped,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embs.tolist()

    def embed_queries(self, queries, query_prefix: str):
        """Queries always need the Instruct:/Query: wrapper. Caller passes the
        already-formatted prefix (typically
        ``instructions['query'].format(task=task)`` from the model config)."""
        if not query_prefix:
            raise ValueError(
                "embed_queries requires a non-empty query_prefix; pass the "
                "formatted Instruct:/Query: template from the model config."
            )
        return self._embed_with_prefix(queries, query_prefix)

    def embed_docs(self, docs, doc_prefix: str = ""):
        """Docs are raw by default (Qwen3/DIVER convention). A non-empty
        ``doc_prefix`` is supported for models that were trained with one."""
        return self._embed_with_prefix(docs, doc_prefix)

    def embed_query(self, query, query_prefix: str):
        return self.embed_queries([query], query_prefix=query_prefix)[0]

    def embed_doc(self, doc, doc_prefix: str = ""):
        return self.embed_docs([doc], doc_prefix=doc_prefix)[0]


def resolve_qwen3_prefixes(instructions: dict, task: str):
    """Pull the query / document prefix from the retriever's config for this
    task. Expected schema:

        {"query": "<template with {task}>", "document": "<optional>"}

    ``query`` is required (Qwen3/DIVER queries must be wrapped).
    ``document`` is optional and defaults to empty (DIVER/Qwen3 docs are raw
    per the official HF model cards).
    """
    if not instructions or "query" not in instructions:
        raise ValueError(
            "missing 'instructions.query' in model config; see "
            "retrieval/configs/qwen3-embed/<task>.json for the required "
            "Instruct:/Query: template"
        )
    query_prefix = instructions["query"].format(task=task)
    doc_prefix = instructions.get("document", "") or ""
    if doc_prefix:
        doc_prefix = doc_prefix.format(task=task)
    return query_prefix, doc_prefix
