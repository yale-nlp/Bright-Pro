import os
import os.path
import time
import torch
import json
import numpy as np
import pytrec_eval
import tiktoken
from tqdm import tqdm,trange
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
# transformers 4.x/5.x compatibility shims for HF custom modeling files
# loaded via trust_remote_code=True. See retrieval/_tf_compat.py.
import os.path as _osp_compat
import sys as _sys_compat
_sys_compat.path.insert(0, _osp_compat.dirname(_osp_compat.abspath(__file__)))
import _tf_compat  # noqa: F401  — imports for side-effects (installs shims)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

import pandas as pd
from collections import defaultdict
# Backends are imported lazily inside the specific retrieval functions that
# need them — avoids dragging in heavy deps when the user only wants to run
# e.g. BM25 or ReasonIR.
#   InstructorEmbedding — retrieval_instructor (inst-xl)
#   gritlm              — retrieval_grit
#   openai              — retrieval_openai (text-embedding-3-Large)

def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def get_embedding_openai(texts, openai_client,tokenizer,model="text-embedding-3-large"):
    texts =[json.dumps(text.replace("\n", " ")) for text in texts]
    success = False
    threshold = 6000
    count = 0
    cur_emb = None
    exec_count = 0
    last_err = None
    while not success:
        exec_count += 1
        if exec_count>5:
            raise RuntimeError(f"openai embeddings retry budget exhausted (exec>5); last error: {last_err}")
        try:
            emb_obj = openai_client.embeddings.create(input=texts, model=model).data
            cur_emb = [e.embedding for e in emb_obj]
            success = True
        except Exception as e:
            print(e)
            last_err = e
            count += 1
            threshold -= 500
            if count>4:
                raise RuntimeError(f"openai embeddings failed 5x; last error: {last_err}") from last_err
            new_texts = []
            for t in texts:
                new_texts.append(cut_text_openai(text=t, tokenizer=tokenizer,threshold=threshold))
            texts = new_texts
    if cur_emb is None:
        raise ValueError("Fail to embed, openai")
    return cur_emb

TASK_MAP = {
    'biology': 'Biology',
    'earth_science': 'Earth Science',
    'economics': 'Economics',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'Stack Overflow',
    'sustainable_living': 'Sustainable Living',
}

def add_instruct_concatenate(texts,task,instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_list(texts,task,instruction):
    return [[instruction.format(task=task),t] for t in texts]

def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_scores(query_ids, doc_ids, scores, excluded_ids=None):
    """Build {qid: {doc_id: score}}, keep top-1000 per query, optionally remove
    any doc_ids present in `excluded_ids[qid]`.

    `excluded_ids` is kept for back-compat with upstream BRIGHT (which had
    non-empty lists for some queries). BRIGHT-PRO examples never set it, so
    the default ``None`` is used and the removal step is a no-op."""
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(doc_ids), f"{len(scores[0])}, {len(doc_ids)}"
    excluded_ids = excluded_ids or {}
    emb_scores = {}
    for query_id, doc_scores in zip(query_ids, scores):
        cur_scores = {str(did): s for did, s in zip(doc_ids, doc_scores)}
        for did in set(excluded_ids.get(str(query_id), [])):
            if did != "N/A":
                cur_scores.pop(did, None)  # .pop with default handles doc not in dict
        cur_scores = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)[:1000]
        emb_scores[str(query_id)] = dict(cur_scores)
    return emb_scores


@torch.no_grad()
def retrieval_gte_qwen2(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    """GTE-Qwen2-7B-Instruct (paper's "GTE-7B")."""
    model_cache_folder = kwargs.get('model_cache_folder', None)
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, cache_dir=model_cache_folder)
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', device_map="auto", trust_remote_code=True, cache_dir=model_cache_folder).eval()
    max_length = kwargs.get('doc_max_length', 8192)
    # Causal LM encoder with last-token pooling: left-padding ensures position -1
    # is always the real last token at batch>1 (right-padding produces garbage
    # embeddings via attention_mask-picked padded positions for shorter items).
    tokenizer.padding_side = 'left'
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    # run.py passes --encode_batch_size CLI arg as kwargs['batch_size'] (see run.py:108);
    # older code read 'encode_batch_size' which silently fell back to 1.
    batch_size = kwargs.get('batch_size', 1)

    # With padding='longest', a batch pads to the longest item — wasted compute
    # on corpora with high length variance. Sort texts by length, encode in
    # sorted batches, then un-sort embeddings back to original order.
    def _encode_sorted(texts):
        lengths = np.array([len(t) for t in texts])
        order = np.argsort(lengths)
        sorted_texts = [texts[i] for i in order]
        sorted_embs = []
        for s in trange(0, len(sorted_texts), batch_size, desc="encode"):
            batch_dict = tokenizer(sorted_texts[s:s+batch_size], max_length=max_length,
                                   padding=True, truncation=True, return_tensors='pt').to(model.device)
            outputs = model(**batch_dict)
            embs = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
            sorted_embs.append(np.asarray(embs))
        sorted_embs = np.concatenate(sorted_embs, axis=0)
        out = np.empty_like(sorted_embs)
        out[order] = sorted_embs  # un-sort
        return out

    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    shard_dir = cache_path[:-4] + '_shards'

    # Shard-encode mode: encode just our slice of documents and write a partial
    # cache file. Caller (run.py) sees a None return value and exits before
    # query encoding/scoring. Used to parallelise heavy doc-encoding (leetcode
    # × 7B+ models) across multiple GPUs.
    shard_id = kwargs.get('shard_id', None)
    n_shards = kwargs.get('n_shards', None)
    if shard_id is not None and n_shards is not None:
        os.makedirs(shard_dir, exist_ok=True)
        n = len(documents)
        start = shard_id * n // n_shards
        end   = (shard_id + 1) * n // n_shards
        print(f"shard {shard_id}/{n_shards}: docs[{start}:{end}] = {end-start} docs")
        shard_emb = _encode_sorted(documents[start:end])
        shard_path = os.path.join(shard_dir, f"shard_{shard_id}_of_{n_shards}.npy")
        np.save(shard_path, shard_emb)
        print(f"saved shard to {shard_path}")
        return None  # signal run.py to skip scoring

    if os.path.isfile(cache_path):
        doc_emb = np.load(cache_path, allow_pickle=True)
    else:
        # Try to assemble from a previously-encoded shard set. Detect n_shards
        # from the filenames in shard_dir rather than guessing common values.
        doc_emb = None
        if os.path.isdir(shard_dir):
            import re
            n_match = re.compile(r"^shard_\d+_of_(\d+)\.npy$")
            counts = {}
            for fn in os.listdir(shard_dir):
                m = n_match.match(fn)
                if m:
                    counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
            for n_try, found in counts.items():
                if found != n_try:
                    continue
                shard_files = [os.path.join(shard_dir, f"shard_{i}_of_{n_try}.npy") for i in range(n_try)]
                if all(os.path.isfile(f) for f in shard_files):
                    print(f"merging {n_try} shards from {shard_dir}")
                    shards = [np.load(f, allow_pickle=True) for f in shard_files]
                    doc_emb = np.concatenate(shards, axis=0)
                    np.save(cache_path, doc_emb)
                    print(f"merged cache saved to {cache_path}  shape={doc_emb.shape}")
                    break
        if doc_emb is None:
            doc_emb = _encode_sorted(documents)
            np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:", doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = torch.tensor(_encode_sorted(queries))
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_bm25(queries,query_ids,documents,doc_ids,excluded_ids,long_context,**kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    # Collect raw similarity arrays per query, then route through `get_scores`
    # so `excluded_ids` filtering matches every other retriever (and upstream
    # BRIGHT). Previously this function returned `all_scores` directly which
    # left excluded "self-docs" in the top-K — leetcode NDCG@10 was ~9 points
    # below the BRIGHT paper because of this.
    raw_scores = []
    bar = tqdm(queries, desc="BM25 retrieval")
    for query in queries:
        bar.update(1)
        analyzed = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(analyzed)]
        raw_scores.append(bm25_index[bm25_query].tolist())
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=raw_scores, excluded_ids=excluded_ids)

@torch.no_grad()
def retrieval_instructor(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    """Instructor-XL (hkunlp/instructor-xl)."""
    model_cache_folder = kwargs.get('model_cache_folder', None)
    model = SentenceTransformer('hkunlp/instructor-xl', cache_folder=model_cache_folder)
    model.set_pooling_include_prompt(False)

    batch_size = kwargs.get('batch_size',4)
    model.max_seq_length = kwargs.get('doc_max_length',2048)
    # queries = add_instruct_list(texts=queries,task=task,instruction=instructions['query'])
    # documents = add_instruct_list(texts=documents,task=task,instruction=instructions['document'])

    query_embs = model.encode(queries,batch_size=batch_size,show_progress_bar=True,prompt=instructions['query'].format(task=task),normalize_embeddings=True)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_embs = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_embs = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True,prompt=instructions['document'].format(task=task))
        np.save(cur_cache_file, doc_embs)
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_grit(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    from gritlm import GritLM  # lazy
    model_cache_folder = kwargs.get('model_cache_folder', None)
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'GritLM/GritLM-7B'
    else:
        print('use',customized_checkpoint)
    # xlang-ai/BRIGHT upstream passes only torch_dtype+mode. Adding device_map
    # or low_cpu_mem_usage here silently breaks GritLM's internal encode (the
    # gritlm library's bidirectional attention toggle assumes a single device).
    if model_cache_folder is not None:
        model = GritLM(customized_checkpoint, torch_dtype="auto", mode="embedding", cache_dir=model_cache_folder)
    else:
        model = GritLM(customized_checkpoint, torch_dtype="auto", mode="embedding")
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    # xlang-ai/BRIGHT upstream uses 256/2048 (ReasonIR fork's 32768 was their
    # custom change). Match paper defaults.
    query_max_length = kwargs.get('query_max_length', 256)
    doc_max_length = kwargs.get('doc_max_length', 2048)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    
    # Clear GPU cache before encoding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file, allow_pickle=True)
    elif ignore_cache:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
    else:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
        np.save(cur_cache_file, doc_emb)
    
    # Clear GPU cache between document and query encoding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_openai(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # model_id -> embedding deployment/model name mapping
    #   openai -> text-embedding-3-large
    embed_model = {
        'openai': 'text-embedding-3-large',
    }.get(model_id, 'text-embedding-3-large')

    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q,tokenizer=tokenizer))
    queries = new_queries
    new_documents = []
    for d in documents:
        new_documents.append(cut_text_openai(text=d,tokenizer=tokenizer))
    documents = new_documents
    doc_emb = []
    batch_size = kwargs.get('batch_size',1024)
    # Prefer OPENAI_API_KEY (public OpenAI). Fall back to Azure only if no
    # OPENAI_API_KEY is set AND Azure has an embeddings deployment.
    if os.environ.get('OPENAI_API_KEY'):
        from openai import OpenAI  # lazy
        openai_client = OpenAI()
    elif os.environ.get('AZURE_OPENAI_ENDPOINT') and os.environ.get('AZURE_OPENAI_API_KEY'):
        from openai import AzureOpenAI  # lazy
        openai_client = AzureOpenAI(
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ['AZURE_OPENAI_API_KEY'],
            api_version=os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview'),
        )
    else:
        raise RuntimeError("Neither OPENAI_API_KEY nor Azure credentials are set")
    # OpenAI embeddings API caps a single request at 300k tokens total across
    # the input list. Build dynamic batches that never exceed ~280k tokens
    # (10k safety margin for tokenizer differences). `batch_size` becomes the
    # *maximum* items per request rather than a fixed count.
    # OpenAI limit is 300k, but `cut_text_openai` does a newline→space substitution
    # that changes the tokenization slightly (the count we measured here is not
    # identical to what the API tokenizer sees). 250k is a safe margin.
    MAX_TOKENS_PER_REQ = 250_000

    def _batch_by_tokens(texts):
        """Yield (start_idx, end_idx) slices that stay under MAX_TOKENS_PER_REQ."""
        i = 0
        while i < len(texts):
            total = 0
            j = i
            while j < len(texts) and j - i < batch_size:
                tlen = min(len(tokenizer.encode(texts[j])), 6000)
                if total + tlen > MAX_TOKENS_PER_REQ and j > i:
                    break
                total += tlen
                j += 1
            yield i, j
            i = j

    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for i, j in tqdm(list(_batch_by_tokens(documents)), desc="doc batches"):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{i}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            cur_emb = get_embedding_openai(texts=documents[i:j],openai_client=openai_client,tokenizer=tokenizer, model=embed_model)
            with open(cur_cache_file,'w') as f:
                json.dump(cur_emb,f,indent=2)
        doc_emb += cur_emb
    query_emb = []
    for i, j in _batch_by_tokens(queries):
        cur_emb = get_embedding_openai(texts=queries[i:j], openai_client=openai_client,
                                       tokenizer=tokenizer, model=embed_model)
        query_emb += cur_emb
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_reasonir(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # NOTE: HF version does not come with pooling function, need to add it manually.
    model_cache_folder = kwargs.get('model_cache_folder', None)
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'reasonir/ReasonIR-8B'
    else:
        print('use',customized_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True, cache_dir=model_cache_folder)
    model = AutoModel.from_pretrained(customized_checkpoint, torch_dtype="auto", trust_remote_code=True, cache_dir=model_cache_folder)
    model.eval()
    model.to(device)
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    # facebookresearch/ReasonIR BRIGHT eval uses 32768 for both, but long-doc
    # tasks (earth_science, stackoverflow) OOM at 32k mask on 178 GB B200. Paper
    # itself truncates docs to 2048 — 4096 here is safe headroom.
    query_max_length = kwargs.get('query_max_length', 4096)
    doc_max_length = kwargs.get('doc_max_length', 4096)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)

    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    if not os.path.isdir(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    skip_doc_emb = kwargs.pop('skip_doc_emb',False)
    if not skip_doc_emb:
        if os.path.isfile(cur_cache_file):
            doc_emb = np.load(cur_cache_file, allow_pickle=True)
        elif ignore_cache:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=batch_size, max_length=doc_max_length)
        else:
            doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=batch_size, max_length=doc_max_length)
            np.save(cur_cache_file, doc_emb)
    cur_cache_file = os.path.join(cache_dir, 'query_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        query_emb = np.load(cur_cache_file, allow_pickle=True)
    elif ignore_cache:
        query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
    else:
        query_emb = model.encode(queries, instruction=query_instruction, batch_size=batch_size, max_length=query_max_length)
        np.save(cur_cache_file, query_emb)
    if skip_doc_emb:
        exit()
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


# Qwen3EmbeddingModel lives in a standalone module so callers can import it
# without dragging in the rest of the retriever backends.
from qwen3_embedding import Qwen3EmbeddingModel, resolve_qwen3_prefixes as _resolve_qwen3_prefixes  # noqa: F401,E402


@torch.no_grad()
def retrieval_qwen3_diver(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model_cache_folder = kwargs.get('model_cache_folder', None)
    cache_model_name = kwargs.get('model_name', 'diver')
    batch_size = kwargs.get('batch_size', 8)

    model_path = 'AQ-MedAI/Diver-Retriever-4B'
    model = Qwen3EmbeddingModel(model_path, max_length=4096,
                                 cache_dir=model_cache_folder, batch_size=batch_size)
    query_prefix, doc_prefix = _resolve_qwen3_prefixes(instructions, task)

    # Check if documents are already encoded
    document_postfix = '_'+kwargs.get('document_postfix', '') if len(kwargs.get('document_postfix', '')) > 0 else ''
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb'+document_postfix, cache_model_name, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, f'0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        with torch.inference_mode():
            doc_emb = model.embed_docs(documents, doc_prefix=doc_prefix)
        torch.cuda.empty_cache()

        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
    print("Shape of doc emb", doc_emb.shape)

    query_emb = []
    with torch.inference_mode():
        query_emb = model.embed_queries(queries, query_prefix=query_prefix)
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()

    if len(kwargs.get('document_postfix', '')) > 0:  # rechunk setting
        dedup_doc_ids = set(doc_ids)
        dedup_scores = []  # shape:[len(scores), len(dedup_doc_ids)], save only the best score for each query-doc pair
        for query_idx in range(len(query_emb)):
            best_scores = {}  # for each query, save the best score for each doc_id
            for idx, score in enumerate(scores[query_idx]):
                doc_id = doc_ids[idx]
                if doc_id not in best_scores or score > best_scores[doc_id]:
                    best_scores[doc_id] = score
            q_doc_scores = []
            for doc_id in dedup_doc_ids:
                q_doc_scores.append(best_scores.get(doc_id))
            dedup_scores.append(q_doc_scores)

        doc_ids, scores = dedup_doc_ids, dedup_scores
        print("Dedup Scores shape:", len(scores[0]))
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_qwen3_embedding(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    model_cache_folder = kwargs.get('model_cache_folder', None)
    cache_model_name = kwargs.get('model_name', 'qwen3-embed')
    batch_size = kwargs.get('batch_size', 8)

    model_path = 'Qwen/Qwen3-Embedding-8B'
    model = Qwen3EmbeddingModel(model_path, max_length=4096,
                                 cache_dir=model_cache_folder, batch_size=batch_size)
    query_prefix, doc_prefix = _resolve_qwen3_prefixes(instructions, task)

    # Check if documents are already encoded
    document_postfix = '_'+kwargs.get('document_postfix', '') if len(kwargs.get('document_postfix', '')) > 0 else ''
    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb'+document_postfix, cache_model_name, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, f'0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        with torch.inference_mode():
            doc_emb = model.embed_docs(documents, doc_prefix=doc_prefix)
        torch.cuda.empty_cache()

        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
    print("Shape of doc emb", doc_emb.shape)

    query_emb = []
    with torch.inference_mode():
        query_emb = model.embed_queries(queries, query_prefix=query_prefix)
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()

    if len(kwargs.get('document_postfix', '')) > 0:  # rechunk setting
        dedup_doc_ids = set(doc_ids)
        dedup_scores = []  # shape:[len(scores), len(dedup_doc_ids)], save only the best score for each query-doc pair
        for query_idx in range(len(query_emb)):
            best_scores = {}  # for each query, save the best score for each doc_id
            for idx, score in enumerate(scores[query_idx]):
                doc_id = doc_ids[idx]
                if doc_id not in best_scores or score > best_scores[doc_id]:
                    best_scores[doc_id] = score
            q_doc_scores = []
            for doc_id in dedup_doc_ids:
                q_doc_scores.append(best_scores.get(doc_id))
            dedup_scores.append(q_doc_scores)

        doc_ids, scores = dedup_doc_ids, dedup_scores
        print("Dedup Scores shape:", len(scores[0]))
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


# ---------------------------------------------------------------------------
# Reasoning-intensive retrievers added in the paper.
# All four are ``sentence-transformers``-compatible. For the three Qwen/BGE
# models we reuse ``Qwen3EmbeddingModel`` (last-token pooling + Instruct:/Query:
# wrapper); EmbeddingGemma-300M uses its own task-prefix convention.
# ---------------------------------------------------------------------------

def _run_qwen3_family(model_path, cache_model_name, task, instructions,
                     queries, query_ids, documents, doc_ids,
                     cache_dir, excluded_ids, long_context, **kwargs):
    """Shared encode+score path for Qwen3/Qwen2.5 embed models that use the
    ``Instruct: {task}\\nQuery:{q}`` wrapper."""
    model_cache_folder = kwargs.get('model_cache_folder', None)
    batch_size = kwargs.get('batch_size', 8)
    model = Qwen3EmbeddingModel(model_path, max_length=4096,
                                cache_dir=model_cache_folder, batch_size=batch_size)
    query_prefix, doc_prefix = _resolve_qwen3_prefixes(instructions, task)

    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', cache_model_name, task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file, allow_pickle=True)
    else:
        with torch.inference_mode():
            doc_emb = model.embed_docs(documents, doc_prefix=doc_prefix)
        torch.cuda.empty_cache()
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
    print("Shape of doc emb", doc_emb.shape)

    with torch.inference_mode():
        query_emb = model.embed_queries(queries, query_prefix=query_prefix)
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    scores = cosine_similarity(query_emb, doc_emb).tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_bge_reasoner(queries, query_ids, documents, doc_ids, task, model_id,
                           instructions, cache_dir, excluded_ids, long_context, **kwargs):
    return _run_qwen3_family(
        model_path='BAAI/bge-reasoner-embed-qwen3-8b-0923',
        cache_model_name='bge-reasoner',
        task=task, instructions=instructions,
        queries=queries, query_ids=query_ids, documents=documents, doc_ids=doc_ids,
        cache_dir=cache_dir, excluded_ids=excluded_ids, long_context=long_context, **kwargs)


@torch.no_grad()
def retrieval_diver_1020(queries, query_ids, documents, doc_ids, task, model_id,
                         instructions, cache_dir, excluded_ids, long_context, **kwargs):
    return _run_qwen3_family(
        model_path='AQ-MedAI/Diver-Retriever-4B-1020',
        cache_model_name='diver-1020',
        task=task, instructions=instructions,
        queries=queries, query_ids=query_ids, documents=documents, doc_ids=doc_ids,
        cache_dir=cache_dir, excluded_ids=excluded_ids, long_context=long_context, **kwargs)


@torch.no_grad()
def retrieval_inf_retriever_pro(queries, query_ids, documents, doc_ids, task, model_id,
                                instructions, cache_dir, excluded_ids, long_context, **kwargs):
    return _run_qwen3_family(
        model_path='infly/inf-retriever-v1-pro',
        cache_model_name='inf-retriever-pro',
        task=task, instructions=instructions,
        queries=queries, query_ids=query_ids, documents=documents, doc_ids=doc_ids,
        cache_dir=cache_dir, excluded_ids=excluded_ids, long_context=long_context, **kwargs)


@torch.no_grad()
def retrieval_rtriever_4b(queries, query_ids, documents, doc_ids, task, model_id,
                          instructions, cache_dir, excluded_ids, long_context, **kwargs):
    """RTriever-4B (yale-nlp/RTriever-4B) — a 4B Qwen3-based dense retriever
    specialized for reasoning-intensive retrieval."""
    return _run_qwen3_family(
        model_path='yale-nlp/RTriever-4B',
        cache_model_name='rtriever-4b',
        task=task, instructions=instructions,
        queries=queries, query_ids=query_ids, documents=documents, doc_ids=doc_ids,
        cache_dir=cache_dir, excluded_ids=excluded_ids, long_context=long_context, **kwargs)


@torch.no_grad()
def retrieval_embeddinggemma(queries, query_ids, documents, doc_ids, task, model_id,
                             instructions, cache_dir, excluded_ids, long_context, **kwargs):
    """EmbeddingGemma-300M uses ``encode_query`` / ``encode_document`` which
    apply the ``task: search result | query: ...`` / ``title: none | text: ...``
    templates internally (per the model card)."""
    from sentence_transformers import SentenceTransformer
    model_cache_folder = kwargs.get('model_cache_folder', None)
    batch_size = kwargs.get('batch_size', 64)
    st_kwargs = {"trust_remote_code": True,
                 "model_kwargs": {"torch_dtype": torch.bfloat16, "device_map": "auto"}}
    if model_cache_folder is not None:
        st_kwargs["cache_folder"] = model_cache_folder
    model = SentenceTransformer('google/embeddinggemma-300m', **st_kwargs)
    model.max_seq_length = 2048

    cache_doc_emb_dir = os.path.join(cache_dir, 'doc_emb', 'embeddinggemma', task, f"long_{long_context}")
    os.makedirs(cache_doc_emb_dir, exist_ok=True)
    cur_cache_file = os.path.join(cache_doc_emb_dir, '0.npy')

    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file, allow_pickle=True)
    else:
        doc_emb = model.encode_document(
            documents, batch_size=batch_size,
            convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True,
        )
        torch.cuda.empty_cache()
        np.save(cur_cache_file, doc_emb)
    print("Shape of doc emb", doc_emb.shape)

    query_emb = model.encode_query(
        queries, batch_size=batch_size,
        convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True,
    )
    print("Shape of query emb", query_emb.shape)

    scores = cosine_similarity(query_emb, doc_emb).tolist()
    return get_scores(query_ids=query_ids, doc_ids=doc_ids, scores=scores, excluded_ids=excluded_ids)


RETRIEVAL_FUNCS = {
    'bm25': retrieval_bm25,
    'inst-xl': retrieval_instructor,
    'gte-qwen2': retrieval_gte_qwen2,
    'qwen3-embed': retrieval_qwen3_embedding,
    'openai': retrieval_openai,
    'grit': retrieval_grit,
    'reasonir': retrieval_reasonir,
    'diver-retriever': retrieval_qwen3_diver,
    'bge-reasoner': retrieval_bge_reasoner,
    'diver-1020': retrieval_diver_1020,
    'inf-retriever-pro': retrieval_inf_retriever_pro,
    'embeddinggemma': retrieval_embeddinggemma,
    'rtriever-4b': retrieval_rtriever_4b,
}

# calculate_retrieval_metrics() moved to retrieval/metrics.py so callers can
# import it without pulling in the retriever backends. Re-export for back-compat.
from metrics import calculate_retrieval_metrics  # noqa: E402, F401
