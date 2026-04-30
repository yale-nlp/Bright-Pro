#!/usr/bin/env python3
"""OpenAI Batch API runner for BRIGHT-PRO static retrieval.

Batch API is ~50% the price of the real-time Embeddings endpoint, at the cost
of a 24-hour turnaround. For our ~104M token workload this saves ~$7 on
text-embedding-3-large and ~$1 on -small.

Workflow (three stages, resumable via state file):
    submit    — build per-chunk JSONL, upload, create batch, save batch_ids
    poll      — report batch status; exit non-zero if any batch errored
    aggregate — download results, compute cosine-sim scores, write
                retrieval/outputs/<task>_<model>/score.json & results.json

State file: retrieval/outputs/.openai_batch_<model>.json  — maps
  {(task, chunk_idx): {jsonl_file_id, batch_id, output_file_id, status}}

Runs on login node; no SLURM. Requires OPENAI_API_KEY in env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
from openai import OpenAI

HERE = Path(__file__).resolve().parent
BPRO = HERE.parent
sys.path.insert(0, str(BPRO))
from bright_pro_data import load_bright_pro  # noqa: E402
from metrics import calculate_retrieval_metrics  # noqa: E402

TASKS = ['biology', 'earth_science', 'economics', 'psychology', 'robotics',
         'stackoverflow', 'sustainable_living']
CHUNK_MAX = 45_000  # requests per chunk (batch API ≤ 50k)
CUT_TOKENS = 6000
OUT_DIR = HERE / 'outputs'
BATCH_TMP = HERE / 'outputs' / '_batch_tmp'
BATCH_TMP.mkdir(parents=True, exist_ok=True)

enc = tiktoken.get_encoding('cl100k_base')

def cut(text: str) -> str:
    ids = enc.encode(text)
    return enc.decode(ids[:CUT_TOKENS]) if len(ids) > CUT_TOKENS else text


def instruct_query(task: str, q: str) -> str:
    # Match retrieval/configs/openai/<task>.json instruction
    return f"Instruct: Given a {task} post, retrieve relevant passages that help answer the post\nQuery: {q}"


def state_path(model: str) -> Path:
    return OUT_DIR / f'.openai_batch_{model}.json'


def load_state(model: str) -> dict:
    p = state_path(model)
    return json.loads(p.read_text()) if p.is_file() else {}


def save_state(model: str, state: dict) -> None:
    state_path(model).write_text(json.dumps(state, indent=2))


def build_task_jsonl(task: str, model: str) -> list[tuple[Path, int, int]]:
    """Write per-chunk JSONL for a task. Returns [(path, n_docs, n_queries_in_chunk), ...]."""
    docs = load_bright_pro('documents', task)
    exs = load_bright_pro('examples', task)
    embed_name = {'openai': 'text-embedding-3-large'}[model]

    # Records: list of (custom_id, text). Docs first, then queries.
    records: list[tuple[str, str]] = []
    for i, d in enumerate(docs):
        records.append((f"{task}|doc|{i}|{d['id']}", cut(d['content'])))
    for i, e in enumerate(exs):
        records.append((f"{task}|qry|{i}|{e['id']}", cut(instruct_query(task, e['query']))))

    # Split into chunks
    chunks = []
    for c_idx in range(0, len(records), CHUNK_MAX):
        chunk = records[c_idx:c_idx + CHUNK_MAX]
        path = BATCH_TMP / f'{model}_{task}_c{c_idx // CHUNK_MAX:02d}.jsonl'
        with path.open('w') as f:
            for cid, txt in chunk:
                f.write(json.dumps({
                    'custom_id': cid,
                    'method': 'POST',
                    'url': '/v1/embeddings',
                    'body': {'model': embed_name, 'input': txt},
                }, ensure_ascii=False) + '\n')
        chunks.append((path, len(chunk), c_idx // CHUNK_MAX))
    return chunks


def cmd_submit(model: str) -> None:
    state = load_state(model)
    client = OpenAI()
    for task in TASKS:
        chunks = build_task_jsonl(task, model)
        print(f"[{task}] built {len(chunks)} chunk(s) for {sum(n for _, n, _ in chunks)} requests")
        for path, n, c_idx in chunks:
            key = f'{task}|{c_idx}'
            if key in state and state[key].get('batch_id'):
                print(f"  skip {key} (already submitted: {state[key]['batch_id']})")
                continue
            with path.open('rb') as f:
                f_obj = client.files.create(file=f, purpose='batch')
            b = client.batches.create(
                input_file_id=f_obj.id,
                endpoint='/v1/embeddings',
                completion_window='24h',
                metadata={'project': 'brightpro', 'task': task, 'model': model, 'chunk': str(c_idx)},
            )
            state[key] = {
                'jsonl_file_id': f_obj.id,
                'batch_id': b.id,
                'n_requests': n,
                'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'status': b.status,
            }
            save_state(model, state)
            print(f"  submitted {key}: file={f_obj.id} batch={b.id} status={b.status}")


def cmd_poll(model: str, verbose=True) -> bool:
    """Returns True if all batches are terminal (completed/failed/expired/cancelled)."""
    state = load_state(model)
    client = OpenAI()
    all_terminal = True
    by_status: dict = {}
    for key, v in state.items():
        if not v.get('batch_id'):
            continue
        b = client.batches.retrieve(v['batch_id'])
        v['status'] = b.status
        if b.status == 'completed':
            v['output_file_id'] = b.output_file_id
        elif b.status in ('failed', 'expired', 'cancelled'):
            v['error_file_id'] = getattr(b, 'error_file_id', None)
        else:
            all_terminal = False
        by_status.setdefault(b.status, []).append(key)
    save_state(model, state)
    if verbose:
        for s, keys in sorted(by_status.items()):
            print(f"  {s}: {len(keys)}  {keys[:3]}{'…' if len(keys) > 3 else ''}")
    return all_terminal


def cmd_aggregate(model: str) -> None:
    """Download all completed outputs, compute scores per task, save results."""
    state = load_state(model)
    client = OpenAI()

    # For each task, gather (custom_id -> embedding) across its chunks
    for task in TASKS:
        task_chunks = [(k, v) for k, v in state.items() if k.startswith(f'{task}|')]
        if not task_chunks or not all(v.get('status') == 'completed' for _, v in task_chunks):
            print(f"  {task}: not all chunks completed, skipping")
            continue
        # Build emb dict
        cid_to_emb: dict[str, list[float]] = {}
        for key, v in task_chunks:
            out_path = BATCH_TMP / f'{model}_{key.replace("|", "_")}.out.jsonl'
            if not out_path.is_file():
                content = client.files.content(v['output_file_id']).read()
                out_path.write_bytes(content)
            with out_path.open() as f:
                for line in f:
                    row = json.loads(line)
                    cid = row['custom_id']
                    if row.get('error'):
                        print(f"  ERROR {cid}: {row['error']}")
                        continue
                    emb = row['response']['body']['data'][0]['embedding']
                    cid_to_emb[cid] = emb

        # Separate docs / queries
        docs = load_bright_pro('documents', task)
        exs = load_bright_pro('examples', task)
        # Determine embedding dim from first available record
        any_emb = next(iter(cid_to_emb.values()))
        emb_dim = len(any_emb)
        # Fill missing embeddings (rare API failures) with zero vectors so
        # those docs never match any query (cos sim = 0).
        n_missing_docs = 0
        n_missing_qry = 0
        doc_emb_rows = []
        for i, d in enumerate(docs):
            cid = f"{task}|doc|{i}|{d['id']}"
            if cid in cid_to_emb:
                doc_emb_rows.append(cid_to_emb[cid])
            else:
                doc_emb_rows.append([0.0] * emb_dim)
                n_missing_docs += 1
        q_emb_rows = []
        for i, e in enumerate(exs):
            cid = f"{task}|qry|{i}|{e['id']}"
            if cid in cid_to_emb:
                q_emb_rows.append(cid_to_emb[cid])
            else:
                q_emb_rows.append([0.0] * emb_dim)
                n_missing_qry += 1
        if n_missing_docs or n_missing_qry:
            print(f"  [{task}] WARNING: missing {n_missing_docs} doc emb + {n_missing_qry} qry emb (filled with zeros)")
        doc_emb = np.array(doc_emb_rows, dtype=np.float32)
        q_emb = np.array(q_emb_rows, dtype=np.float32)
        # Normalize (OpenAI embeddings are already L2-normalized, but be safe)
        doc_emb /= np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-12
        q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12

        # Cosine similarity + score dict per query, top-200 per query
        sim = q_emb @ doc_emb.T
        scores: dict[str, dict[str, float]] = {}
        doc_ids = [d['id'] for d in docs]
        top_k = 200
        for qi, e in enumerate(exs):
            qid = str(e['id'])
            idx = np.argsort(-sim[qi])[:top_k]
            scores[qid] = {doc_ids[i]: float(sim[qi, i]) for i in idx}

        out_dir = OUT_DIR / f'{task}_{model}'
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'score.json').write_text(json.dumps(scores, indent=2))

        # Metrics
        qrels = {str(e['id']): {gid: 1 for gid in e['gold_ids']} for e in exs}
        metrics = calculate_retrieval_metrics(results=scores, qrels=qrels,
                                              k_values=[1, 5, 10, 25, 50, 100])
        (out_dir / 'results.json').write_text(json.dumps(metrics, indent=2))
        print(f"  {task}: NDCG@25={metrics['NDCG@25']*100:.1f}  Recall@25={metrics['Recall@25']*100:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['openai'], required=True)
    ap.add_argument('--mode', choices=['submit', 'poll', 'aggregate', 'all'], required=True)
    args = ap.parse_args()

    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY not set')

    if args.mode == 'submit':
        cmd_submit(args.model)
    elif args.mode == 'poll':
        done = cmd_poll(args.model)
        sys.exit(0 if done else 2)
    elif args.mode == 'aggregate':
        cmd_aggregate(args.model)
    else:  # all — helpful for small runs
        cmd_submit(args.model)
        while not cmd_poll(args.model, verbose=True):
            print('sleeping 5 min...')
            time.sleep(300)
        cmd_aggregate(args.model)


if __name__ == '__main__':
    main()
