"""
Searchers package for different search implementations.

Registers the 10 retrievers used in the paper's agentic evaluation:
BM25, Instructor-XL, GTE-Qwen2-7B-Instruct (paper's "GTE-7B"), GritLM-7B,
ReasonIR-8B, DIVER-Retriever-4B, DIVER-Retriever-4B-1020, Qwen3-Embedding-8B,
BGE-Reasoner-Embed-Qwen3-8B, and RTriever-4B.
"""

from enum import Enum
from .base import BaseSearcher
from .bm25_searcher import BM25Searcher
from .inst_searcher import InstSearcher
from .qwen_searcher import QwenSearcher
from .reasonir_searcher import ReasonIRSearcher
from .grit_searcher import GritSearcher
from .qwen3_family_searcher import Qwen3FamilySearcher


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""
    BM25 = ("bm25", BM25Searcher)
    REASONIR_EMBED = ("reasonir", ReasonIRSearcher)
    DIVER_RETRIEVER = ("diver-retriever", Qwen3FamilySearcher)
    INSTRUCTOR_XL = ("inst-xl", InstSearcher)
    GTE_QWEN2 = ("gte-qwen2", QwenSearcher)
    GRIT = ("grit", GritSearcher)
    # Qwen3 family — sentence-transformers based, one class, CLI via --qwen3-model-id
    QWEN3_EMBED = ("qwen3-embed", Qwen3FamilySearcher)
    DIVER_1020 = ("diver-1020", Qwen3FamilySearcher)
    BGE_REASONER = ("bge-reasoner", Qwen3FamilySearcher)
    RTRIEVER_4B = ("rtriever-4b", Qwen3FamilySearcher)

    def __init__(self, cli_name, searcher_class):
        self.cli_name = cli_name
        self.searcher_class = searcher_class

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = [
    "BaseSearcher",
    "SearcherType",
]
