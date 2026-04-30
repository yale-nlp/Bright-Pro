"""
Instructor searcher implementation using SentenceTransformer.
"""

import logging
import os
import os.path
import sys as _sys_bp
from typing import Any, Dict, Optional
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

_bp = os.path.dirname(os.path.abspath(__file__))
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS

from .base import BaseSearcher, find_existing_doc_emb

logger = logging.getLogger(__name__)


class InstructorEmbeddingModel:
    def __init__(self, model_id, max_length=2048, device="auto", args=None, task=None, instructions=None):
        """
        Initialize the Instructor embedding model (instructor-xl).

        Args:
            model_id: Model identifier (must be 'inst-xl')
            max_length: Maximum sequence length
            device: Device to use (auto, cuda, cpu)
            args: Arguments containing cache directories
            task: Task name for instruction formatting
            instructions: Dictionary containing query and document instructions
        """
        cache_dir = getattr(args, 'model_cache_dir', None) if args else None
        self.model = SentenceTransformer('hkunlp/instructor-xl', cache_folder=cache_dir)
        self.model.set_pooling_include_prompt(False)
        self.model.max_seq_length = max_length
        self.max_length = max_length
        self.task = task
        self.instructions = instructions
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
        else:
            self.device = device
        
        if torch.cuda.is_available() and self.device == "cuda":
            self.model = self.model.cuda()

    def embed_query(self, query):
        """Embed a single query with instruction prompt."""
        if self.instructions and 'query' in self.instructions:
            prompt = self.instructions['query'].format(task=self.task)
        else:
            prompt = f"Represent this query for retrieving relevant documents: "
        
        embedding = self.model.encode(
            query, 
            prompt=prompt,
            normalize_embeddings=True
        )
        return embedding

    def embed_queries(self, queries, batch_size=4):
        """Embed multiple queries with instruction prompts."""
        if self.instructions and 'query' in self.instructions:
            prompt = self.instructions['query'].format(task=self.task)
        else:
            prompt = f"Represent this query for retrieving relevant documents: "
        
        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            prompt=prompt,
            normalize_embeddings=True
        )
        return embeddings

    def embed_doc(self, doc):
        """Embed a single document with instruction prompt."""
        if self.instructions and 'document' in self.instructions:
            prompt = self.instructions['document'].format(task=self.task)
        else:
            prompt = f"Represent this document for retrieval: "
        
        embedding = self.model.encode(
            doc,
            prompt=prompt,
            normalize_embeddings=True
        )
        return embedding

    def embed_docs(self, docs, batch_size=4):
        """Embed multiple documents with instruction prompts."""
        if self.instructions and 'document' in self.instructions:
            prompt = self.instructions['document'].format(task=self.task)
        else:
            prompt = f"Represent this document for retrieval: "
        
        embeddings = self.model.encode(
            docs,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt=prompt
        )
        return embeddings


class InstSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        """Parse arguments from CLI that you will need to instantiate the searcher."""
        parser.add_argument('--inst_batch_size', type=int, default=4,
                           help='Batch size for encoding')
        parser.add_argument('--inst_max_length', type=int, default=2048,
                           help='Maximum sequence length for documents')

    def __init__(self, args):
        """Initialize the searcher with the arguments."""
        self.args = args
        self.task = None
        self.searcher = None
        self.model = None
        self.doc_emb = None
        self.doc_ids = None
        self.documents = None

        model_id = 'inst-xl'
        batch_size = getattr(args, 'inst_batch_size', 4)
        max_length = getattr(args, 'inst_max_length', 2048)

        cache_model_name = model_id

        # Load Instructor instructions from configs/inst-xl/{task}.json when available
        instructions: Dict[str, str] = {}
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cfg_path = os.path.join(project_root, 'configs', 'inst-xl', f'{self.args.task}.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as jf:
                    cfg = json.load(jf)
                    if 'instructions' in cfg:
                        instructions = cfg['instructions']
        except Exception as e:
            logger.warning(f"Failed to load Instructor instructions for task {getattr(self.args, 'task', '')}: {e}")
        
        # Initialize model
        self.model = InstructorEmbeddingModel(
            model_id=model_id,
            max_length=max_length,
            args=args,
            task=args.task,
            instructions=instructions
        )
        
        # Setup caching directory (fixed-round agentic layout)
        cache_doc_emb_dir = os.path.join(
            self.args.cache_dir,
            'doc_emb',
            cache_model_name,
            self.args.task,
            f"long_False_{batch_size}"
        )
        os.makedirs(cache_doc_emb_dir, exist_ok=True)
        cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

        # Resolve cache: try fixed-round agentic path first, then known static-eval layouts.
        # static-eval wrote with batch_size 4/8/16 depending on task and used the
        # same long_False_{bs}/0.npy directory pattern.
        primary_rel = f"long_False_{batch_size}/0.npy"
        legacy_rels = [
            f"long_False_{bs}/0.npy" for bs in (4, 8, 16, 24, 32)
            if bs != batch_size
        ]
        existing = find_existing_doc_emb(
            self.args.cache_dir, cache_model_name, self.args.task,
            primary=primary_rel, extra_candidates=legacy_rels,
        )

        # Load or compute document embeddings
        if existing is not None:
            logger.info(f"Loading cached document embeddings from {existing}")
            doc_emb = np.load(existing, allow_pickle=True)
            self.doc_emb = doc_emb
            
            # Load document IDs and texts if not already loaded
            if self.doc_ids is None or self.documents is None:
                doc_pairs = load_bright_pro('documents', self.args.task)
                self.doc_ids = []
                self.documents = []
                for dp in doc_pairs:
                    self.doc_ids.append(dp['id'])
                    self.documents.append(dp['content'])
        else:
            logger.info("Computing document embeddings...")
            # Load documents from dataset
            doc_pairs = load_bright_pro('documents', self.args.task)
            self.doc_ids = []
            self.documents = []

            for dp in doc_pairs:
                self.doc_ids.append(dp['id'])
                self.documents.append(dp['content'])
            
            # Compute embeddings
            with torch.no_grad():
                doc_emb = self.model.embed_docs(self.documents, batch_size=batch_size)
            torch.cuda.empty_cache()
            
            # Convert to numpy array and save
            doc_emb = np.array(doc_emb)
            self.doc_emb = doc_emb
            np.save(cur_cache_file, self.doc_emb)
            logger.info(f"Saved document embeddings to {cur_cache_file}")
        
        logger.info(f"Document embeddings shape: {self.doc_emb.shape}")
        
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Perform search and return results.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with format: {"docid": str, "score": float, "text": str}
        """
        # Embed the query
        query_emb = self.model.embed_query(query)
        query_emb = np.array([query_emb])  # Add batch dimension
        
        # Compute cosine similarity scores
        scores = cosine_similarity(query_emb, self.doc_emb)
        scores = scores.tolist()[0]  # Remove batch dimension
        
        # Return top-k results using the base class method
        return self.get_top_k_scores(k, self.doc_ids, self.documents, scores)
    
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve full document by ID.
        
        Args:
            docid: Document ID to retrieve
            
        Returns:
            Document dictionary with format: {"docid": str, "text": str} or None if not found
        """
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
        """Return the type of search."""
        return "instructor"
