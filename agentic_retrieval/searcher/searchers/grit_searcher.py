"""
GritLM embedding searcher implementation.
Follows the same structure and caching conventions as Diver/Qwen/ReasonIR searchers.
"""

# Apply transformers 4.x/5.x compat shims for custom-code HF modeling files.
import sys as _sys_compat
import os as _os_compat
_compat_dir = _os_compat.path.dirname(_os_compat.path.abspath(__file__))
while _compat_dir != "/" and not _os_compat.path.isfile(_os_compat.path.join(_compat_dir, "retrieval", "_tf_compat.py")):
    _compat_dir = _os_compat.path.dirname(_compat_dir)
_sys_compat.path.insert(0, _os_compat.path.join(_compat_dir, "retrieval"))
import _tf_compat  # noqa: F401


import logging
import os
import os.path
import sys as _sys_bp
from typing import Any, Dict, Optional, List
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_bp = os.path.dirname(os.path.abspath(__file__))
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS

from .base import BaseSearcher, find_existing_doc_emb

logger = logging.getLogger(__name__)


class GritEmbeddingModel:
    def __init__(self, model_path: str = 'GritLM/GritLM-7B', max_length_docs: int = 2048, max_length_queries: int = 256, args=None, instructions: Optional[Dict[str, str]] = None):
        """
        Lightweight wrapper around GritLM embedding interface.
        Uses lazy import to avoid import cost if not used.
        """
        # Defer import to runtime to prevent mandatory dependency at import time
        from gritlm import GritLM  # type: ignore

        # Derive task and initialize instructions (to be set from configs)
        task = args.task if args else None
        self.query_instruction = ''
        self.doc_instruction = ''

        # Initialize model with memory-efficient settings
        self.model = GritLM(
            model_path,
            torch_dtype="auto",
            mode="embedding",
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.max_length_docs = max_length_docs
        self.max_length_queries = max_length_queries

        # Override from provided instructions
        if instructions and task:
            if 'query' in instructions and isinstance(instructions['query'], str):
                self.query_instruction = instructions['query'].format(task=task)
            if 'document' in instructions and isinstance(instructions['document'], str):
                self.doc_instruction = instructions['document']
    @torch.no_grad()
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        embs = self.model.encode(
            documents,
            instruction=self.doc_instruction,
            batch_size=1,
            max_length=self.max_length_docs,
        )
        return np.array(embs)

    @torch.no_grad()
    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            [query],
            instruction=self.query_instruction,
            batch_size=1,
            max_length=self.max_length_queries,
        )
        return np.array(emb[0])


class GritSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument('--grit_doc_max_length', type=int, default=2048,
                            help='Maximum document token length for GritLM encoding')
        parser.add_argument('--grit_query_max_length', type=int, default=256,
                            help='Maximum query token length for GritLM encoding')
        parser.add_argument('--grit_model', type=str, default='GritLM/GritLM-7B',
                            help='GritLM model identifier')

    def __init__(self, args):
        self.args = args
        self.task = None
        self.model = None
        self.doc_emb = None
        self.doc_ids = None
        self.documents = None

        cache_model_name = 'grit'

        grit_model_id = getattr(self.args, 'grit_model', 'GritLM/GritLM-7B')
        doc_max_len = getattr(self.args, 'grit_doc_max_length', 2048)
        query_max_len = getattr(self.args, 'grit_query_max_length', 256)

        # Load Grit instructions from configs/grit/{task}.json when available
        instructions: Dict[str, str] = {}
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cfg_path = os.path.join(project_root, 'configs', 'grit', f'{self.args.task}.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as jf:
                    cfg = json.load(jf)
                    instructions = cfg['instructions']
        except Exception as e:
            logger.warning(f"Failed to load Grit instructions for task {getattr(self.args, 'task', '')}: {e}")

        # Initialize embedding model
        self.model = GritEmbeddingModel(
            model_path=grit_model_id,
            max_length_docs=doc_max_len,
            max_length_queries=query_max_len,
            args=self.args,
            instructions=instructions,
        )

        # Prepare cache path (fixed-round agentic default: long_False_1.npy)
        cache_doc_emb_dir = os.path.join(self.args.cache_dir, 'doc_emb', cache_model_name, self.args.task)
        os.makedirs(cache_doc_emb_dir, exist_ok=True)
        cur_cache_file = os.path.join(cache_doc_emb_dir, f'long_{False}_1.npy')

        # Resolve cache: try fixed-round agentic path then known static-eval layouts.
        # static-eval grit had both flat and subdir variants depending on bs.
        existing = find_existing_doc_emb(
            self.args.cache_dir, cache_model_name, self.args.task,
            primary='long_False_1.npy',
            extra_candidates=[f'long_False_{bs}.npy' for bs in (24, 16, 8, 4)] +
                              [f'long_False_{bs}/0.npy' for bs in (1, 4, 8, 16, 24)],
        )

        if existing is not None:
            logger.info(f"Loading cached document embeddings from {existing}")
            doc_emb = np.load(existing, allow_pickle=True)
            self.doc_emb = doc_emb

            if self.doc_ids is None or self.documents is None:
                doc_pairs = load_bright_pro('documents', self.args.task)
                self.doc_ids = []
                self.documents = []
                for dp in doc_pairs:
                    self.doc_ids.append(dp['id'])
                    self.documents.append(dp['content'])
        else:
            # Load documents and build embeddings
            doc_pairs = load_bright_pro('documents', self.args.task)
            self.doc_ids = []
            self.documents = []
            for dp in doc_pairs:
                self.doc_ids.append(dp['id'])
                self.documents.append(dp['content'])

            with torch.inference_mode():
                doc_emb_list = self.model.embed_documents(self.documents)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.doc_emb = np.array(doc_emb_list)
            np.save(cur_cache_file, self.doc_emb)

        print("Shape of doc emb", self.doc_emb.shape)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.model.embed_query(query)
        query_emb = np.array([query_emb])

        scores = cosine_similarity(query_emb, self.doc_emb)
        scores = scores.tolist()[0]
        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        return {
            "docid": docid,
            "text": "place-holder-text",
        }

    @property
    def search_type(self) -> str:
        return "grit"


