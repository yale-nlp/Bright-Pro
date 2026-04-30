
# Apply transformers 4.x/5.x compat shims for custom-code HF modeling files.
import sys as _sys_compat
import os as _os_compat
_compat_dir = _os_compat.path.dirname(_os_compat.path.abspath(__file__))
while _compat_dir != "/" and not _os_compat.path.isfile(_os_compat.path.join(_compat_dir, "retrieval", "_tf_compat.py")):
    _compat_dir = _os_compat.path.dirname(_compat_dir)
_sys_compat.path.insert(0, _os_compat.path.join(_compat_dir, "retrieval"))
import _tf_compat  # noqa: F401


"""
Qwen embedding searcher implementation.
Uses the Qwen2-7B-Instruct embedding model (Alibaba-NLP/gte-Qwen2-7B-instruct).
"""

import logging
import os
import os.path
import sys as _sys_bp
from typing import Any, Dict, Optional, List
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange

_bp = os.path.dirname(os.path.abspath(__file__))
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS

from .base import BaseSearcher, find_existing_doc_emb

logger = logging.getLogger(__name__)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token from the hidden states based on attention mask."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def add_instruct_concatenate(texts: List[str], task: str, instruction: str) -> List[str]:
    """Add instruction to texts for better embedding performance."""
    return [f"{instruction} {text}" for text in texts]


def get_scores(query_ids: List[str], doc_ids: List[str], scores: List[List[float]], excluded_ids: set = None) -> Dict[str, Dict[str, float]]:
    """Convert scores to the expected format."""
    if excluded_ids is None:
        excluded_ids = set()
    
    result = {}
    for i, query_id in enumerate(query_ids):
        result[query_id] = {}
        for j, doc_id in enumerate(doc_ids):
            if doc_id not in excluded_ids:
                result[query_id][doc_id] = scores[i][j]
    
    return result


class QwenEmbeddingModel:
    def __init__(self, max_length: int = 8192, device: str = "auto", args=None, **kwargs):
        """
        Initialize the Qwen2 embedding model.
        
        Args:
            max_length: Maximum sequence length
            device: Device to use
            args: Arguments containing cache directories
        """
        self.max_length = kwargs.get('doc_max_length', max_length)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        model_path = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
        
        # Get cache directory
        cache_dir = getattr(args, 'model_cache_dir', None) if args else None
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_path, 
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        ).eval()
        
        self.batch_size = kwargs.get('encode_batch_size', 64)

    def embed_queries(self, queries: List[str], task: str = None, instruction: str = None) -> List[np.ndarray]:
        """Embed multiple queries."""
        if instruction:
            queries = add_instruct_concatenate(queries, task, instruction)
        
        query_embeddings = []
        for start_idx in trange(0, len(queries), self.batch_size):
            batch_queries = queries[start_idx:start_idx + self.batch_size]
            batch_dict = self.tokenizer(
                batch_queries, 
                max_length=self.max_length, 
                padding=True,
                truncation=True, 
                return_tensors='pt'
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
                query_embeddings.extend([emb.numpy() for emb in embeddings])
        
        return query_embeddings

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed multiple documents with caching."""
        doc_embeddings = []
        
        for start_idx in trange(0, len(documents), self.batch_size):
            batch_docs = documents[start_idx:start_idx + self.batch_size]
            batch_dict = self.tokenizer(
                batch_docs, 
                max_length=self.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
                doc_embeddings.extend([emb.numpy() for emb in embeddings])
        
        return np.array(doc_embeddings)

    def embed_query(self, query: str, task: str = None, instruction: str = None) -> np.ndarray:
        """Embed a single query."""
        if instruction:
            query = f"{instruction} {query}"
        
        batch_dict = self.tokenizer(
            [query], 
            max_length=self.max_length, 
            padding=True,
            truncation=True, 
            return_tensors='pt'
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
            return embedding[0].numpy()


class QwenSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        """Parse arguments from CLI that you will need to instantiate the searcher."""
        parser.add_argument('--qwen_batch_size', type=int, default=1,
                          help='Batch size for encoding')
        parser.add_argument('--qwen_doc_max_length', type=int, default=None,
                          help='Maximum document length (model-dependent default)')
    
    def __init__(self, args):
        """Initialize the searcher with the arguments."""
        self.args = args
        self.task = None
        self.model = None
        self.doc_emb = None
        self.doc_ids = None
        self.documents = None
        
        # Wraps gte-qwen2-7b-instruct (paper calls this "GTE-7B"); invoked via
        # CLI name 'gte-qwen2'.
        batch_size = getattr(args, 'qwen_batch_size', 1)
        doc_max_length = getattr(args, 'qwen_doc_max_length', None)

        cache_model_name = 'gte-qwen2'

        # Load instructions from configs/gte-qwen2/{task}.json when available.
        self.instructions: Dict[str, str] = {}
        self.query_instruction = None
        try:
            # Resolve project root relative to this file: searcher/searchers/ -> project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cfg_path = os.path.join(project_root, 'configs', 'gte-qwen2', f'{self.args.task}.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as jf:
                    cfg = json.load(jf)
                    if 'instructions' in cfg:
                        self.instructions = cfg['instructions']
                        if 'query' in self.instructions and isinstance(self.instructions['query'], str):
                            self.query_instruction = self.instructions['query'].format(task=self.args.task)
        except Exception as e:
            logger.warning(f"Failed to load Qwen instructions for task {getattr(self.args, 'task', '')}: {e}")
        
        # Initialize the embedding model
        kwargs = {
            'encode_batch_size': batch_size,
        }
        if doc_max_length:
            kwargs['doc_max_length'] = doc_max_length
            
        self.model = QwenEmbeddingModel(
            device="auto",
            args=args,
            **kwargs
        )
        
        # Check if documents are already encoded
        cache_doc_emb_dir = os.path.join(self.args.cache_dir, 'doc_emb', cache_model_name, self.args.task)
        os.makedirs(cache_doc_emb_dir, exist_ok=True)
        cur_cache_file = os.path.join(cache_doc_emb_dir, f'long_context_{batch_size}.npy')

        # Resolve cache: try fixed-round agentic path, then static-eval's flat-file naming.
        # static-eval used `long_False_{bs}.npy` (commonly bs=24 here), and an
        # older subdir-style `long_False_{bs}/0.npy` is also seen in the wild.
        primary_rel = f"long_context_{batch_size}.npy"
        legacy_rels = []
        for bs in (batch_size, 1, 4, 8, 16, 24, 32):
            legacy_rels.append(f"long_False_{bs}.npy")
            legacy_rels.append(f"long_False_{bs}/0.npy")
        existing = find_existing_doc_emb(
            self.args.cache_dir, cache_model_name, self.args.task,
            primary=primary_rel, extra_candidates=legacy_rels,
        )

        # Load documents from dataset first
        logger.info("Loading documents from dataset...")
        doc_pairs = load_bright_pro('documents', self.args.task)
        self.doc_ids = []
        self.documents = []

        for dp in doc_pairs:
            self.doc_ids.append(dp['id'])
            self.documents.append(dp['content'])

        total_docs = len(self.documents)
        logger.info(f"Total documents to encode: {total_docs}")

        # Check if cached embeddings exist and if they're complete
        doc_emb = None
        if existing is not None:
            logger.info(f"Loading cached document embeddings from {existing}")
            doc_emb = np.load(existing, allow_pickle=True)
            num_cached = doc_emb.shape[0]
            logger.info(f"Found {num_cached} cached embeddings out of {total_docs} documents")

            if num_cached >= total_docs:
                logger.info("All documents already encoded")
                self.doc_emb = doc_emb
            else:
                logger.info(f"Resuming encoding from document {num_cached}")
        else:
            logger.info("No cached embeddings found. Starting from scratch.")
        
        # Compute embeddings if needed (either from scratch or resume)
        if doc_emb is None or doc_emb.shape[0] < total_docs:
            start_from = 0 if doc_emb is None else doc_emb.shape[0]
            logger.info(f"Computing document embeddings from index {start_from}...")
            
            for start_idx in trange(start_from, total_docs, batch_size):
                batch_docs = self.documents[start_idx:start_idx + batch_size]
                batch_dict = self.model.tokenizer(
                    batch_docs, 
                    max_length=self.model.max_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                ).to(self.model.model.device)
                
                with torch.no_grad():
                    outputs = self.model.model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
                    
                    if doc_emb is None:
                        doc_emb = embeddings.numpy()
                    else:
                        doc_emb = np.concatenate((doc_emb, embeddings.numpy()), axis=0)
                
                # Save every 1000 iterations
                if (start_idx + batch_size) % 1000 == 0:
                    np.save(cur_cache_file, doc_emb)
                    logger.info(f"Saved {doc_emb.shape[0]} embeddings to cache")
            
            # Final save
            self.doc_emb = doc_emb
            np.save(cur_cache_file, self.doc_emb)
            logger.info(f"Completed encoding. Saved {self.doc_emb.shape[0]} embeddings to {cur_cache_file}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Document embeddings shape: {self.doc_emb.shape}")
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform search and return results.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of search results with format: {"docid": str, "score": float, "text": str}
        """
        # Embed the query with instruction from config
        query_emb = self.model.embed_query(query, instruction=self.query_instruction)
        query_emb = np.array([query_emb])  # Add batch dimension
        
        # Normalize embeddings
        query_emb = F.normalize(torch.tensor(query_emb), p=2, dim=1).numpy()
        doc_emb_normalized = F.normalize(torch.tensor(self.doc_emb), p=2, dim=1).numpy()
        
        # Compute cosine similarity scores  
        scores = cosine_similarity(query_emb, doc_emb_normalized)
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
        return "qwen"
