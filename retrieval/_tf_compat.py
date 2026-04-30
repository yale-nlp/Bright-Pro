"""Transformers compat shims for HF custom modeling files.

Several retrievers (GritLM-7B, gte-Qwen2-7B-instruct, ReasonIR-8B,
inf-retriever-v1-pro) load HF modeling files via ``trust_remote_code=True``.
Those files depend on transformers APIs that were removed in:
  * 4.46+: ``DynamicCache.get_usable_length`` / ``seen_tokens`` / ``get_max_length``
  * 5.5: ``DynamicCache.{from,to}_legacy_cache``, ``config.rope_theta`` direct attr

Importing this module installs version-agnostic shims (each only fires if the
attribute is missing). Safe to import multiple times.

Usage: ``import _tf_compat`` at the top of any module that loads a custom-code
HF model with ``trust_remote_code=True``.
"""
from __future__ import annotations
import warnings

import transformers as _tf

_tf_major = int(_tf.__version__.split('.')[0])

# ─── tf 5.x: rope_theta moved into config.rope_parameters ───────────────────
if _tf_major >= 5:
    warnings.warn(
        f"transformers {_tf.__version__} detected. BRIGHT-PRO is validated on "
        "transformers 4.55.x; custom-code retrievers (grit/gte-qwen2/reasonir/"
        "inf-retriever-pro) silently produce bad embeddings on 5.x because the "
        "bidirectional-attention toggle in legacy modeling_attn_mask_utils stops "
        "propagating. Use the pinned `brightpro-eval` env (environment.yml).",
        RuntimeWarning,
    )
    from transformers import PretrainedConfig as _PretrainedConfig
    if not hasattr(_PretrainedConfig, "_brightpro_rope_shimmed"):
        _orig = _PretrainedConfig.__getattribute__
        _LEGACY = {"rope_theta"}

        def _legacy_rope_attr(self, name):
            try:
                return _orig(self, name)
            except AttributeError:
                if name in _LEGACY:
                    try:
                        rp = _orig(self, "rope_parameters")
                        if rp and name in rp:
                            return rp[name]
                    except Exception:
                        pass
                raise
        _PretrainedConfig.__getattribute__ = _legacy_rope_attr
        _PretrainedConfig._brightpro_rope_shimmed = True

# ─── DynamicCache shims (version-agnostic; methods missing in 4.46+ AND 5.x) ─
try:
    from transformers.cache_utils import DynamicCache as _DynamicCache
    if not hasattr(_DynamicCache, "from_legacy_cache"):
        def _from_legacy_cache(cls, past_key_values=None):
            if past_key_values is None:
                return cls()
            return cls(past_key_values)
        _DynamicCache.from_legacy_cache = classmethod(_from_legacy_cache)
    if not hasattr(_DynamicCache, "to_legacy_cache"):
        def _to_legacy_cache(self):
            if hasattr(self, "layers"):
                return tuple((layer.keys, layer.values) for layer in self.layers)
            return tuple(zip(self.key_cache, self.value_cache))
        _DynamicCache.to_legacy_cache = _to_legacy_cache
    if not hasattr(_DynamicCache, "get_usable_length"):
        def _get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx) if hasattr(self, "get_seq_length") else 0
        _DynamicCache.get_usable_length = _get_usable_length
    if not hasattr(_DynamicCache, "seen_tokens"):
        _DynamicCache.seen_tokens = property(
            lambda self: self.get_seq_length() if hasattr(self, "get_seq_length") else 0
        )
    if not hasattr(_DynamicCache, "get_max_length"):
        _DynamicCache.get_max_length = lambda self: None
except Exception:
    pass
