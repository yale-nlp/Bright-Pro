"""
Abstract base class for search implementations.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
import argparse

_logger = logging.getLogger(__name__)


def find_existing_doc_emb(
    cache_dir: str,
    model_name: str,
    task: str,
    primary: str,
    extra_candidates: Sequence[str] = (),
) -> Optional[str]:
    """Resolve a doc embedding cache file across static-eval / fixed-round agentic layouts.

    static-eval (retrieval/) and fixed-round agentic (agentic_retrieval/) historically chose
    slightly different filenames per retriever (different default batch_size,
    different long_context flag spelling, flat .npy vs subdir/0.npy). This
    helper accepts the searcher's preferred filename plus a list of legacy
    candidates and returns the first existing path, or None if every probe
    misses (in which case the caller re-encodes and writes to ``primary``).

    All candidates are joined under ``<cache_dir>/doc_emb/<model_name>/<task>/``.
    """
    base = os.path.join(cache_dir, "doc_emb", model_name, task)
    candidates = [primary, *extra_candidates]
    for rel in candidates:
        p = os.path.join(base, rel)
        if os.path.isfile(p):
            if rel != primary:
                _logger.info(f"[cache fallback] {model_name}/{task}: using legacy path {rel}")
            return p
    return None


class BaseSearcher(ABC):
    """Abstract base class for all search implementations."""
    
    @classmethod
    @abstractmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add searcher-specific arguments to the argument parser."""
        pass
    
    @abstractmethod
    def __init__(self, args):
        """Initialize the searcher with parsed arguments."""
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search and return results.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with format: {"docid": str, "score": float, "snippet": str}
        """
        pass
    
    @abstractmethod
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.
        
        Args:
            docid: Document ID to retrieve
            
        Returns:
            Document dictionary with format: {"docid": str, "snippet": str} or None if not found
        """
        pass
    
    @property
    @abstractmethod
    def search_type(self) -> str:
        """Return the type of search (e.g., 'BM25', 'FAISS')."""
        pass

    def set_task(self, task: str) -> None:
        """Swap task-specific state (corpus, embeddings, per-task instructions)
        without reloading the underlying model / tokenizer / GPU state.

        Default implementation: mutate ``self.args.task`` and re-run ``__init__``.
        Subclasses with expensive model loads should override this to only swap
        the task-specific slots (typically ``self.doc_ids``, ``self.doc_emb``,
        and per-task instruction strings).
        """
        if getattr(self, "args", None) is None:
            raise RuntimeError(
                "set_task() default fallback expects self.args; subclasses "
                "without args must override set_task")
        self.args.task = task
        # Re-init reuses this searcher object identity so tool_handler's
        # searcher reference stays valid.
        self.__init__(self.args)

    def search_description(self, k: int = 10) -> str:
        """
        Description of the search tool to be passed to the LLM.
        """
        return f"Perform a search on a knowledge source. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."
    
    def get_document_description(self) -> str:
        """
        Description of the get_document tool to be passed to the LLM.
        """
        return "Retrieve a full document by its docid."
    
    def get_top_k_scores(self, k: int, doc_ids: list, documents: list, scores: list) -> list[dict[str, Any]]:
        """
        Get top k documents with highest scores.
        
        Args:
            k: Number of top documents to return
            doc_ids: List of document IDs
            documents: List of document texts
            scores: List of similarity scores
            
        Returns:
            List of dictionaries with docid, score, and text
        """
        # Create tuples of (score, doc_id, document) and sort by score in descending order
        scored_docs = list(zip(scores, doc_ids, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Get top k results
        top_k_docs = scored_docs[:k]
        
        # Format results
        results = []
        for score, doc_id, document in top_k_docs:
            results.append({
                "docid": doc_id,
                "score": float(score),
                "text": document
            })
        
        return results


