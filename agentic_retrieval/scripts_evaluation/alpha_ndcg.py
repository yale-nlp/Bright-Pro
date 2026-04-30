import argparse
import json
import os as _os_bp
import sys as _sys_bp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable

_bp = _os_bp.path.dirname(_os_bp.path.abspath(__file__))
while _bp != "/" and not _os_bp.path.isfile(_os_bp.path.join(_bp, "bright_pro_data.py")):
    _bp = _os_bp.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS, build_doc_to_aspect_id, build_aspect_weights


def parse_round_docids(run_json: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extract per-round retrieved doc ids from a run JSON produced in fixed_turn_runs.

    Expected structure (observed): a top-level list field `result` containing items with
    - type == 'tool_result'
    - round: int (1-based)
    - docids: List[str]

    Returns a map from round number to the list of doc ids in retrieval order.
    If multiple tool_results exist for the same round, the latest one wins.
    """
    rounds: Dict[int, List[str]] = {}
    for item in run_json.get('result', []):
        if not isinstance(item, dict):
            continue
        if item.get('type') != 'tool_result':
            continue
        rnd = item.get('round')
        docids = item.get('docids')
        if isinstance(rnd, int) and isinstance(docids, list):
            rounds[rnd] = [str(d) for d in docids]
    return rounds


def cumulative_unique(seqs: Iterable[List[str]]) -> List[str]:
    """
    Given a sequence of ranked lists (rounds), return the cumulative unique list
    preserving original order across rounds.
    """
    seen: set = set()
    out: List[str] = []
    for seq in seqs:
        for d in seq:
            if d in seen:
                continue
            seen.add(d)
            out.append(d)
    return out


def build_aspect_maps(task: str, cache_dir: str) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Build maps:
    - doc_id_to_aspect_id: map from document id (string) to its aspect id (string)
    - aspect_id_to_weight: map from aspect id (string) to its weight (float)

    Notes:
    - Documents split has fields: id (string), content (string), aspect (string)
    - Aspects split has fields: id (string), weight (float)
    """
    # doc -> aspect lookup derives from aspects' supporting_docs (authoritative)
    doc_id_to_aspect_id: Dict[str, str] = build_doc_to_aspect_id(task)
    # Stored weights are raw Likert {1,2,3}; normalize per-query (Σ=1) here.
    aspect_id_to_weight: Dict[str, float] = build_aspect_weights(task)
    return doc_id_to_aspect_id, aspect_id_to_weight


def _get_aspect_weight(aspect_id_to_weight: Dict[str, float], aspect_id: str) -> float:
    """Strict lookup: raise on missing aspect instead of silently using 1.0.

    Callers should pass the map returned by
    `bright_pro_data.build_aspect_weights(task)`, which covers every aspect of
    the task (normalized so Σ = 1 per query)."""
    if aspect_id not in aspect_id_to_weight:
        raise KeyError(
            f"aspect weight missing for {aspect_id!r}; pass the full map from "
            f"`bright_pro_data.build_aspect_weights(task)`"
        )
    return float(aspect_id_to_weight[aspect_id])


def compute_alpha_dcg_at_k(
    ranked_doc_ids: List[str],
    gold_doc_ids: List[str],
    doc_id_to_aspect_id: Dict[str, str],
    aspect_id_to_weight: Dict[str, float],
    alpha: float,
    k: int,
) -> float:
    """
    Compute weighted alpha-DCG@k for a single query.

    Assumptions:
    - Only gold documents have aspects; each gold document has exactly one aspect.
    - Relevance to the document's aspect is 1, other aspects 0.

    gain_i = weight(a) * (1 - alpha)^{n_a(i-1)} / log2(i+1)
    if ranked_doc_ids[i-1] is a gold doc with aspect a; else 0.
    """
    from math import log2

    gold_set = set(str(x) for x in gold_doc_ids)
    aspect_seen_counts: Dict[str, int] = {}
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id not in gold_set:
            continue
        aspect_id = doc_id_to_aspect_id.get(doc_id)
        if aspect_id is None:
            continue
        weight = _get_aspect_weight(aspect_id_to_weight, aspect_id)
        prev_count = aspect_seen_counts.get(aspect_id, 0)
        novelty = (1.0 - alpha) ** prev_count
        gain = weight * novelty / log2(i + 1)
        dcg += gain
        aspect_seen_counts[aspect_id] = prev_count + 1
    return dcg


def compute_alpha_idcg_at_k(
    gold_doc_ids: List[str],
    doc_id_to_aspect_id: Dict[str, str],
    aspect_id_to_weight: Dict[str, float],
    alpha: float,
    k: int,
) -> float:
    """
    Compute the ideal weighted alpha-DCG@k (IDCG) for a query given gold documents.

    Greedy diversification across aspects maximizes alpha-DCG with per-aspect novelty.
    """
    from math import log2

    # Group gold documents by aspect
    aspect_to_docs: Dict[str, List[str]] = {}
    for doc_id in (str(x) for x in gold_doc_ids):
        aspect_id = doc_id_to_aspect_id.get(doc_id)
        if aspect_id is None:
            continue
        aspect_to_docs.setdefault(aspect_id, []).append(doc_id)

    remaining: Dict[str, int] = {a: len(docs) for a, docs in aspect_to_docs.items()}
    seen: Dict[str, int] = {a: 0 for a in aspect_to_docs.keys()}

    idcg = 0.0
    for i in range(1, k + 1):
        best_aspect: Optional[str] = None
        best_gain = 0.0
        for a, rem in remaining.items():
            if rem <= 0:
                continue
            weight = _get_aspect_weight(aspect_id_to_weight, a)
            novelty = (1.0 - alpha) ** seen[a]
            gain = weight * novelty / log2(i + 1)
            if gain > best_gain:
                best_gain = gain
                best_aspect = a
        if best_aspect is None:
            break
        idcg += best_gain
        remaining[best_aspect] -= 1
        seen[best_aspect] += 1

    return idcg


KNOWN_TASKS = SE_TASKS


def infer_task_from_path(path: Path) -> Optional[str]:
    """
    Infer task name from a run file path like .../fixed_turn_runs/<model>/<benchmark>/<task>/run_*.json
    Falls back to None if no known task segment is found.
    """
    for part in path.parts:
        if part in KNOWN_TASKS:
            return part
    return None


def load_examples_index(task: str, cache_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Load examples for a task and build indices:
    - qid_to_gold: mapping from query id (string) to list of gold doc ids (strings)
    - doc_id_to_aspect_id: as produced by build_aspect_maps
    Returns (qid_to_gold, doc_id_to_aspect_id)
    """
    examples = load_bright_pro('examples', task)
    field = 'gold_ids'
    qid_to_gold: Dict[str, List[str]] = {}
    for ex in examples:
        qid_str = str(ex['id'])
        gold_list = [str(x) for x in ex.get(field, [])]
        qid_to_gold[qid_str] = gold_list
    doc_id_to_aspect_id, _ = build_aspect_maps(task, cache_dir)
    return qid_to_gold, doc_id_to_aspect_id


def evaluate_run(
    run_path: Path,
    qid_to_gold: Dict[str, List[str]],
    doc_id_to_aspect_id: Dict[str, str],
    aspect_id_to_weight: Dict[str, float],
    alpha: float,
) -> Dict[str, Optional[float]]:
    """
    Compute alpha-nDCG@5 (round1), @10 (rounds1+2), @15 (rounds1+2+3) for a single run file.
    Returns dict with keys 'r1_ndcg@5', 'r2_ndcg@10', 'r3_ndcg@15'. Missing rounds yield None.
    """
    data = json.loads(run_path.read_text())
    qid = str(data.get('query_id')) if data.get('query_id') is not None else None
    if qid is None or qid not in qid_to_gold:
        return {'r1_ndcg@5': None, 'r2_ndcg@10': None, 'r3_ndcg@15': None}

    gold = qid_to_gold[qid]
    if not gold:
        return {'r1_ndcg@5': None, 'r2_ndcg@10': None, 'r3_ndcg@15': None}

    rounds = parse_round_docids(data)
    r1 = rounds.get(1, [])
    r2 = rounds.get(2, [])
    r3 = rounds.get(3, [])

    res: Dict[str, Optional[float]] = {'r1_ndcg@5': None, 'r2_ndcg@10': None, 'r3_ndcg@15': None}

    # Precompute IDCGs
    idcg5 = compute_alpha_idcg_at_k(gold, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=5)
    idcg10 = compute_alpha_idcg_at_k(gold, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=10)
    idcg15 = compute_alpha_idcg_at_k(gold, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=15)

    if r1:
        ranked1 = cumulative_unique([r1])
        dcg1 = compute_alpha_dcg_at_k(ranked1, gold, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=5)
        res['r1_ndcg@5'] = (dcg1 / idcg5) if idcg5 > 0 else None

    if r1 or r2:
        ranked12 = cumulative_unique([r1, r2])
        dcg2 = compute_alpha_dcg_at_k(ranked12, gold, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=10)
        res['r2_ndcg@10'] = (dcg2 / idcg10) if idcg10 > 0 else None

    if r1 or r2 or r3:
        ranked123 = cumulative_unique([r1, r2, r3])
        dcg3 = compute_alpha_dcg_at_k(ranked123, gold, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=15)
        res['r3_ndcg@15'] = (dcg3 / idcg15) if idcg15 > 0 else None

    return res


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-run metrics into per-(model, task) averages.
    Each result must contain keys: model, task, metrics dict with r1_ndcg@5, r2_ndcg@10, r3_ndcg@15.
    """
    agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        key = (r['model'], r['task'])
        if key not in agg:
            agg[key] = {
                'model': r['model'],
                'task': r['task'],
                'count': 0,
                'r1_sum': 0.0, 'r1_count': 0,
                'r2_sum': 0.0, 'r2_count': 0,
                'r3_sum': 0.0, 'r3_count': 0,
            }
        a = agg[key]
        a['count'] += 1
        m = r['metrics']
        if m.get('r1_ndcg@5') is not None:
            a['r1_sum'] += m['r1_ndcg@5']
            a['r1_count'] += 1
        if m.get('r2_ndcg@10') is not None:
            a['r2_sum'] += m['r2_ndcg@10']
            a['r2_count'] += 1
        if m.get('r3_ndcg@15') is not None:
            a['r3_sum'] += m['r3_ndcg@15']
            a['r3_count'] += 1

    # finalize averages
    finalized: Dict[str, Any] = {}
    for (model, task), a in agg.items():
        finalized[f'{model}::{task}'] = {
            'model': model,
            'task': task,
            'queries': a['count'],
            'avg_r1_ndcg@5': (a['r1_sum'] / a['r1_count']) if a['r1_count'] > 0 else None,
            'avg_r2_ndcg@10': (a['r2_sum'] / a['r2_count']) if a['r2_count'] > 0 else None,
            'avg_r3_ndcg@15': (a['r3_sum'] / a['r3_count']) if a['r3_count'] > 0 else None,
            'support': {
                'r1': a['r1_count'], 'r2': a['r2_count'], 'r3': a['r3_count'],
            }
        }
    return finalized


def main():
    parser = argparse.ArgumentParser(description='Evaluate weighted alpha-nDCG across rounds in fixed_turn_runs')
    parser.add_argument('--fixed_dir', type=str, default='fixed_turn_runs')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--cache_dir', type=str, default='../retrieval/cache')
    parser.add_argument('--benchmark', type=str, default=None, help='optional benchmark filter (e.g., grit)')
    parser.add_argument('--save_json', type=str, default=None, help='path to write all results and aggregates as JSON')
    parser.add_argument('--excel_out', type=str, default=None, help='path to write an Excel file with model x task matrices per metric')
    args = parser.parse_args()

    fixed_root = Path(args.fixed_dir)
    if not fixed_root.exists():
        print(f"fixed_dir not found: {fixed_root}")
        return

    # Discover run files: fixed_turn_runs/<model>/<benchmark>/<task>/run_*.json
    run_files: List[Path] = []
    for model_dir in fixed_root.iterdir():
        if not model_dir.is_dir():
            continue
        for bench_dir in model_dir.iterdir():
            if not bench_dir.is_dir():
                continue
            if args.benchmark and bench_dir.name != args.benchmark:
                continue
            for task_dir in bench_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                # collect JSON files in task dir
                for f in task_dir.glob('*.json'):
                    run_files.append(f)

    if not run_files:
        print(json.dumps({'error': 'no run files found', 'fixed_dir': str(fixed_root)}))
        return

    # Prepare caches per task
    task_cache: Dict[str, Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, float]]] = {}

    all_results: List[Dict[str, Any]] = []

    for run_path in sorted(run_files):
        task = infer_task_from_path(run_path)
        if task is None:
            continue
        # parents: [run_file, task_dir, retriever_dir, generator_dir, fixed_turn_runs, ...]
        generator = run_path.parents[2].name if len(run_path.parents) >= 3 else 'unknown'
        retriever = run_path.parents[1].name if len(run_path.parents) >= 2 else 'unknown'
        # Save model as retriever label (e.g., bge, bm25, grit, ...)
        model = retriever
        task_key = task
        if task_key not in task_cache:
            qid_to_gold, doc_id_to_aspect_id = load_examples_index(task_key, args.cache_dir)
            _, aspect_id_to_weight = build_aspect_maps(task_key, args.cache_dir)
            task_cache[task_key] = (qid_to_gold, doc_id_to_aspect_id, aspect_id_to_weight)

        qid_to_gold, doc_id_to_aspect_id, aspect_id_to_weight = task_cache[task_key]

        metrics = evaluate_run(run_path, qid_to_gold, doc_id_to_aspect_id, aspect_id_to_weight, alpha=args.alpha)
        all_results.append({
            'file': str(run_path),
            'model': model,
            'generator': generator,
            'task': task,
            'metrics': metrics,
        })

    # Print per-run results
    for r in all_results:
        print(json.dumps(r, ensure_ascii=False))

    # Aggregate per (model, task)
    aggregates = aggregate_metrics(all_results)
    for key, val in sorted(aggregates.items()):
        print(json.dumps({'aggregate': val}, ensure_ascii=False))

    if args.save_json:
        payload = {
            'alpha': args.alpha,
            'fixed_dir': str(fixed_root),
            'results': all_results,
            'aggregates': aggregates,
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # Optional Excel output: each sheet is a metric; rows=models, columns=tasks
    if args.excel_out:
        try:
            import pandas as pd  # type: ignore
            models = sorted({v['model'] for v in aggregates.values()})
            tasks = sorted({v['task'] for v in aggregates.values()})

            def build_matrix(metric_key: str) -> 'pd.DataFrame':
                rows = []
                for m in models:
                    row = {'model': m}
                    for t in tasks:
                        key = f'{m}::{t}'
                        val = aggregates.get(key, {}).get(metric_key)
                        row[t] = val
                    rows.append(row)
                df = pd.DataFrame(rows)
                if not df.empty:
                    df = df.set_index('model')
                return df

            df_r1 = build_matrix('avg_r1_ndcg@5')
            df_r2 = build_matrix('avg_r2_ndcg@10')
            df_r3 = build_matrix('avg_r3_ndcg@15')

            out_path = Path(args.excel_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(out_path) as writer:
                df_r1.to_excel(writer, sheet_name='r1_ndcg@5')
                df_r2.to_excel(writer, sheet_name='r2_ndcg@10')
                df_r3.to_excel(writer, sheet_name='r3_ndcg@15')
        except Exception as e:
            print(json.dumps({'excel_error': str(e), 'hint': 'Install pandas (and openpyxl) or use --save_json'}, ensure_ascii=False))


if __name__ == '__main__':
    main()


