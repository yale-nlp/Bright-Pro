"""
BM25 searcher implementation using pyserini analyzer and gensim's LuceneBM25Model.
Follows the same structure and dataset-loading conventions as other searchers.
"""

import logging
import os
import os.path
import sys as _sys_bp
from typing import Any, Dict, Optional

import numpy as np

_bp = os.path.dirname(os.path.abspath(__file__))
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS

from .base import BaseSearcher


logger = logging.getLogger(__name__)


class BM25Searcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument('--bm25_k1', type=float, default=0.9, help='BM25 k1 parameter')
        parser.add_argument('--bm25_b', type=float, default=0.4, help='BM25 b parameter')
        # relies on common args: --cache_dir, --model_cache_dir, --task
        return
    
    def __init__(self, args):
        self.args = args
        self.doc_ids = None
        self.documents = None
        self.dictionary = None
        self.model = None
        self.bm25_index = None
        self.analyzer = None
        self.task = None
        # analyzer is task-agnostic — build once
        from pyserini import analysis
        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        if getattr(args, "task", None):
            self.set_task(args.task)

    def set_task(self, task: str) -> None:
        """Rebuild the BM25 index for a new task. The analyzer is re-used;
        the gensim dictionary and sparse matrix are task-specific so they get
        rebuilt (~10-30 s for 50-120k docs)."""
        self.task = task
        doc_pairs = load_bright_pro('documents', task)
        self.doc_ids = [dp['id'] for dp in doc_pairs]
        self.documents = [dp['content'] for dp in doc_pairs]

        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity
        corpus_tokens = [self.analyzer.analyze(x) for x in self.documents]
        self.dictionary = Dictionary(corpus_tokens)
        self.model = LuceneBM25Model(
            dictionary=self.dictionary,
            k1=getattr(self.args, 'bm25_k1', 0.9),
            b=getattr(self.args, 'bm25_b', 0.4),
        )
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus_tokens))]
        self.bm25_index = SparseMatrixSimilarity(
            bm25_corpus,
            num_docs=len(corpus_tokens),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )
        logger.info("BM25 index built for %s: %d documents", task, len(self.documents))
        
    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        if not query:
            return []
        query_tokens = self.analyzer.analyze(query)
        bm25_query = self.model[self.dictionary.doc2bow(query_tokens)]
        scores = self.bm25_index[bm25_query].tolist()
        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)
    
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        try:
            idx = self.doc_ids.index(docid)
            return {
                "docid": docid,
                "text": self.documents[idx],
            }
        except ValueError:
            return None
    
    @property
    def search_type(self) -> str:
        return "bm25"
    
    def search_description(self, k: int = 10) -> str:
        return f"Perform a BM25 search on the document collection. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."
    
    def get_document_description(self) -> str:
        return "Retrieve a full document by its docid."