import os
import multiprocessing as mp
import argparse
import json
import sys as _sys_bp
from tqdm import tqdm
# `retrievers` pulls heavy backends (torch, cohere, sentence_transformers, ...).
# Import it lazily — only if we actually need to run retrieval. Evaluating a
# pre-existing score.json should not require the retriever backends.
from metrics import calculate_retrieval_metrics

_bp = os.path.dirname(os.path.abspath(__file__))
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS


def _load_bright_orig(config, task, cache_dir):
    """Load examples/documents from upstream `xlangai/BRIGHT` HF dataset.

    Returns list[dict] matching BRIGHT-PRO's load_bright_pro shape so the rest
    of run.py can stay agnostic to the source.

    Many array tasks hit the dataset builder's NFS lock file at the same time,
    which fails sporadically with EBADF/ESTALE ("Stale file handle"). Retry a
    few times with backoff before giving up.
    """
    import time, random
    from datasets import load_dataset
    last_err = None
    for attempt in range(8):
        try:
            ds = load_dataset("xlangai/BRIGHT", config, cache_dir=cache_dir)[task]
            return [dict(row) for row in ds]
        except OSError as e:
            last_err = e
            # 11 = EAGAIN (lock busy), 116 = ESTALE (NFS stale handle)
            if e.errno in (11, 116) or "Stale file handle" in str(e):
                wait = random.uniform(2.0, 6.0) * (attempt + 1)
                print(f"_load_bright_orig: NFS lock contention (attempt {attempt+1}), retry in {wait:.1f}s — {e}")
                time.sleep(wait)
                continue
            raise
    raise last_err


if __name__=='__main__':
    # Ensure CUDA is not initialized in forked subprocesses
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser()
    # Default BRIGHT-PRO task list (7 SE subsets). With --dataset bright we
    # accept any string so caller can pick original-BRIGHT-only tasks
    # (aops, leetcode, theoremqa_*, pony) without us hard-coding them here.
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='bright-pro',
                        choices=['bright-pro', 'bright'],
                        help="Source dataset: 'bright-pro' = yale-nlp/Bright-Pro on HF; "
                             "'bright' = upstream xlangai/BRIGHT via HF.")
    parser.add_argument('--max_queries', type=int, default=-1,
                        help='Truncate examples to first N queries (smoke test). '
                             'Default -1 = use all.')
    # Retrievers reported in the paper.
    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25', 'inst-xl', 'gte-qwen2', 'qwen3-embed',
                                 'openai', 'grit', 'reasonir', 'diver-retriever',
                                 'bge-reasoner', 'diver-1020', 'inf-retriever-pro',
                                 'embeddinggemma',
                                 'rtriever-4b'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--reasoning', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    parser.add_argument('--model_cache_folder', type=str, default=None)
    parser.add_argument('--topk_save', type=int, default=200,
                        help='truncate score.json to top-K scored docs per query (0 = keep all). '
                             'Default 200 is enough for NDCG@100 / α-nDCG@25 and keeps score.json ~100x smaller.')
    # Sharded doc encoding — run N independent encode jobs and a final
    # merge-and-score job (which sees no shard args and concatenates from disk).
    parser.add_argument('--shard_id', type=int, default=-1,
                        help='Shard index for parallel doc encoding (default -1 = full mode).')
    parser.add_argument('--n_shards', type=int, default=-1,
                        help='Total number of shards; partner of --shard_id.')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,f"{args.task}_{args.model}")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    score_file_path = os.path.join(args.output_dir,f'score.json')

    if args.input_file is not None:
        with open(args.input_file) as f:
            examples = json.load(f)
    elif args.dataset == 'bright':
        examples = _load_bright_orig('examples', args.task, args.cache_dir)
    else:
        if args.task not in SE_TASKS:
            raise ValueError(f"task {args.task!r} not in BRIGHT-PRO SE_TASKS={SE_TASKS}")
        examples = load_bright_pro('examples', args.task)

    if args.max_queries > 0:
        examples = examples[: args.max_queries]
        print(f"--max_queries={args.max_queries} -> truncated to {len(examples)} examples")

    if args.dataset == 'bright':
        doc_pairs = _load_bright_orig('documents', args.task, args.cache_dir)
    else:
        doc_pairs = load_bright_pro('documents', args.task)

    doc_ids = []
    documents = []

    for dp in doc_pairs:
        doc_ids.append(dp['id'])
        documents.append(dp['content'])
    


    if not os.path.isfile(score_file_path):
        with open(os.path.join(args.config_dir,args.model,f"{args.task}.json")) as f:
            config = json.load(f)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        queries = []
        query_ids = []
        for e in examples:
            queries.append(e["query"])
            query_ids.append(e['id'])
        # `excluded_ids` was removed from BRIGHT-PRO examples (always empty)
        # but is non-empty for some queries in upstream xlangai/BRIGHT — pass
        # the per-query lists through so retrievers / get_scores can drop them.
        if args.dataset == 'bright':
            excluded_ids = {str(e['id']): list(e.get('excluded_ids', []) or []) for e in examples}
        else:
            excluded_ids = {}
        assert len(queries)==len(query_ids), f"{len(queries)}, {len(query_ids)}"
        doc_ids_dir = os.path.join(args.cache_dir, 'doc_ids')
        os.makedirs(doc_ids_dir, exist_ok=True)
        doc_ids_path = os.path.join(doc_ids_dir, f"{args.task}_{args.long_context}.json")
        # Concurrent (model × task) jobs share this cache. Use a write-temp +
        # atomic-rename pattern so peers never see a partial file. On read,
        # tolerate a partial file (mid-rename window) by retrying briefly.
        cached_doc_ids = None
        if os.path.isfile(doc_ids_path):
            import time
            for _ in range(5):
                try:
                    with open(doc_ids_path) as f:
                        cached_doc_ids = json.load(f)
                    break
                except (json.JSONDecodeError, ValueError):
                    time.sleep(0.5)
        if cached_doc_ids is not None:
            for id1, id2 in zip(cached_doc_ids, doc_ids):
                assert id1 == id2
        else:
            tmp_path = f"{doc_ids_path}.{os.getpid()}.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(doc_ids, f, indent=2)
            os.replace(tmp_path, doc_ids_path)
        assert len(doc_ids)==len(documents), f"{len(doc_ids)}, {len(documents)}"

        print(f"{len(queries)} queries")
        print(f"{len(documents)} documents")
        if args.debug:
            documents = documents[:30]
            doc_paths = doc_ids[:30]
        kwargs = {}
        if args.query_max_length>0:
            kwargs = {'query_max_length': args.query_max_length}
        if args.doc_max_length>0:
            kwargs.update({'doc_max_length': args.doc_max_length})
        if args.encode_batch_size>0:
            kwargs.update({'batch_size': args.encode_batch_size})
        if args.key is not None:
            kwargs.update({'key': args.key})
        if args.ignore_cache:
            kwargs.update({'ignore_cache': args.ignore_cache})
        if args.model_cache_folder is not None:
            kwargs.update({'model_cache_folder': args.model_cache_folder})
        if args.shard_id >= 0 and args.n_shards > 0:
            kwargs.update({'shard_id': args.shard_id, 'n_shards': args.n_shards})
        from retrievers import RETRIEVAL_FUNCS  # lazy: avoids loading backends for eval-only runs
        scores = RETRIEVAL_FUNCS[args.model](
            queries=queries, query_ids=query_ids, documents=documents, excluded_ids=excluded_ids,
            instructions=config['instructions_long'] if args.long_context else config['instructions'],
            doc_ids=doc_ids, task=args.task, cache_dir=args.cache_dir, long_context=args.long_context,
            model_id=args.model, checkpoint= args.checkpoint, **kwargs
        )
        if scores is None:
            # Shard-encode mode: doc_emb shard saved, exit before scoring.
            print(f"shard {args.shard_id}/{args.n_shards} done — exiting before query scoring")
            _sys_bp.exit(0)
        # Truncate to top-K per query for disk compactness. The full score
        # vector can be 10-20 MB per query; top-200 is sufficient for NDCG@100
        # and α-nDCG@25 and brings each score.json well under 1 MB.
        if args.topk_save and args.topk_save > 0:
            scores = {
                str(qid): dict(sorted(d.items(), key=lambda kv: -kv[1])[: args.topk_save])
                for qid, d in scores.items()
            }
        os.makedirs(args.output_dir, exist_ok=True)
        with open(score_file_path,'w') as f:
            json.dump(scores,f,indent=2)
    else:
        with open(score_file_path) as f:
            scores = json.load(f)
        print(score_file_path,'exists')
    # `gold_ids_long` was removed — same content as `gold_ids` across all SE tasks.
    key = 'gold_ids'
    ground_truth = {}
    for e in tqdm(examples):
        ground_truth[e['id']] = {gid: 1 for gid in e[key]}

    print(args.output_dir)
    # Paper appendix tables (model_performance_arecall/recall/ndcg) report @25;
    # we keep the full k ladder {1, 5, 10, 25, 50, 100} so downstream tables can
    # pick whichever they need.
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth,
                                          k_values=[1, 5, 10, 25, 50, 100])
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {results_path}')
