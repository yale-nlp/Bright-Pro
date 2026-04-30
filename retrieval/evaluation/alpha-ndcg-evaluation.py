import argparse
import json
import os as _os_bp
import sys as _sys_bp
from pathlib import Path
from typing import Dict, List, Tuple, Optional

_bp = _os_bp.path.dirname(_os_bp.path.abspath(__file__))
while _bp != "/" and not _os_bp.path.isfile(_os_bp.path.join(_bp, "bright_pro_data.py")):
    _bp = _os_bp.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS, build_doc_to_aspect_id, build_aspect_weights

# Optional dependency for Excel export
try:
	import pandas as pd
except ImportError:
	pd = None


def parse_score_entry(entry):
	"""
	Normalize a score.json entry to a ranked list of document ids (strings).

	The repository uses two possible formats in different places:
	- Mapping: {doc_id: score, ...} per query id
	- Ranked list: [doc_id, doc_id, ...] per query id

	This function returns a list of doc ids sorted from high to low score
	when a mapping is given, or the list as-is when already ranked.
	"""
	if isinstance(entry, dict):
		# Sort by score descending, keep only doc ids
		return [doc_id for doc_id, _ in sorted(entry.items(), key=lambda x: x[1], reverse=True)]
	elif isinstance(entry, list):
		# Already a ranked list
		return entry
	else:
		raise ValueError(f"Unsupported score entry type: {type(entry)}")


def build_aspect_maps(task: str, cache_dir: str) -> Tuple[Dict[str, str], Dict[str, float]]:
	"""
	Build maps:
	- doc_id_to_aspect_id: map from document id (string) to its aspect id (string)
	- aspect_id_to_weight: map from aspect id (string) to its weight (float)

	Notes:
	- Documents split has fields: id (string), content (string), aspect (string)
	- Aspects split has fields: id (string), weight (float)
	"""
	# doc -> aspect lookup is derived from each aspect's `supporting_docs`
	# (authoritative), not from a per-document field on the documents table.
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


def compute_alpha_dcg_at_k(ranked_doc_ids: List[str], gold_doc_ids: List[str], doc_id_to_aspect_id: Dict[str, str], aspect_id_to_weight: Dict[str, float], alpha: float, k: int) -> float:
	"""
	Compute weighted alpha-DCG@k for a single query.

	Assumptions given by user:
	- Only gold documents have aspects; each gold document has exactly one aspect.
	- Relevance to the document's aspect is 1, other aspects 0.

	We interpret this as: For each aspect a present among the gold documents, we
	set the aspect gain weight to weight(a) from the aspects table. The net gain
	at rank i is the weighted novelty contribution of the retrieved document's
	aspect if that document is gold.

	alpha-DCG definition adapted with weights and binary relevance:
	For ranks i = 1..k, if ranked_doc_ids[i-1] is a gold doc with aspect a,
	gain_i = weight(a) * (1 - alpha)^{n_a(i-1)} / log2(i+1), where n_a(i-1) is the
	number of previously retrieved gold documents up to rank i-1 whose aspect is a.
	Otherwise gain_i = 0.

	Returns the unnormalized alpha-DCG@k (DCG). The caller can divide by IDCG to get nDCG.
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
			print(f"Missing aspect for doc {doc_id}")
			# If missing aspect, skip as only gold docs are assumed to have aspect
			continue
		weight = _get_aspect_weight(aspect_id_to_weight, aspect_id)
		prev_count = aspect_seen_counts.get(aspect_id, 0)
		novelty = (1.0 - alpha) ** prev_count
		gain = weight * novelty / log2(i + 1)
		dcg += gain
		aspect_seen_counts[aspect_id] = prev_count + 1
	return dcg


def compute_alpha_idcg_at_k(gold_doc_ids: List[str], doc_id_to_aspect_id: Dict[str, str], aspect_id_to_weight: Dict[str, float], alpha: float, k: int) -> float:
	"""
	Compute the ideal weighted alpha-DCG@k (IDCG) for a query given gold documents.

	We simulate the best possible ordering to maximize DCG under the same gain model:
	- As each gold document contributes only to its aspect, the optimal strategy is
	to diversify aspects early. We therefore greedily place one document from each
	distinct aspect before repeating aspects.
	- When repeating the same aspect, earlier occurrences have higher novelty.

	This greedy diversification strategy is optimal for the alpha-DCG objective with
	aspect weights when the only gains come from per-aspect novelty contributions.
	"""
	from math import log2

	# Group gold documents by aspect
	aspect_to_docs: Dict[str, List[str]] = {}
	for doc_id in (str(x) for x in gold_doc_ids):
		aspect_id = doc_id_to_aspect_id.get(doc_id)
		if aspect_id is None:
			# Skip gold without aspect; per problem statement this should not happen
			continue
		aspect_to_docs.setdefault(aspect_id, []).append(doc_id)

	# Prepare counts of remaining docs per aspect and seen counts for novelty
	remaining: Dict[str, int] = {a: len(docs) for a, docs in aspect_to_docs.items()}
	seen: Dict[str, int] = {a: 0 for a in aspect_to_docs.keys()}

	idcg = 0.0
	for i in range(1, k + 1):
		# Pick the aspect that yields maximum marginal gain at this rank
		best_aspect = None
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
		# Place a document of that aspect at this rank
		idcg += best_gain
		remaining[best_aspect] -= 1
		seen[best_aspect] += 1

	return idcg


def evaluate_file(score_file: Path, examples, doc_id_to_aspect_id: Dict[str, str], aspect_id_to_weight: Dict[str, float], alpha: float, k: int) -> Dict:
	"""
	Compute alpha-nDCG@k per query for a given score.json file and summarize.
	"""
	with open(score_file) as f:
		scores = json.load(f)

	field = 'gold_ids'
	ndcgs: List[float] = []
	n = 0

	for ex in examples:
		qid = str(ex['id']) if isinstance(ex['id'], (int,)) else str(ex['id'])
		if qid not in scores:
			continue
		ranked_list = parse_score_entry(scores[qid])
		gold_list = [str(x) for x in ex.get(field, [])]
		if not gold_list:
			continue

		dcg = compute_alpha_dcg_at_k(ranked_list, gold_list, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=k)
		idcg = compute_alpha_idcg_at_k(gold_list, doc_id_to_aspect_id, aspect_id_to_weight, alpha=alpha, k=k)
		if idcg > 0:
			ndcgs.append(dcg / idcg)
			n += 1

	avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
	return {
		'file': str(score_file),
		'num_evaluated_queries': n,
		'alpha': alpha,
		'k': k,
		'avg_alpha_ndcg': avg_ndcg
	}


def find_score_files(task: str, output_dir: str, explicit_score_file: str = None, process_all: bool = False) -> List[Path]:
	if explicit_score_file:
		return [Path(explicit_score_file)]
	outputs_dir = Path(output_dir)
	if task == 'all':
		# Scan all score.json across all tasks
		return sorted([p for p in outputs_dir.glob('*/score.json')])
	if process_all:
		return [p for p in outputs_dir.glob('*/score.json') if p.parent.name.startswith(f"{task}_") or p.parent.name == task]
	# default: first matching file for task
	files = list(outputs_dir.glob(f"{task}_*/score.json"))
	if not files:
		# also allow directory exactly named task (rare)
		files = list(outputs_dir.glob(f"{task}/score.json"))
	return files[:1]


KNOWN_TASKS = SE_TASKS


def infer_task_from_dirname(dirname: str) -> Optional[str]:
	"""Infer task from run directory name using known task prefixes.

	Handles tasks containing underscores by matching the longest known task
	which is a prefix of the directory name followed by '_' (or exact match).
	"""
	for t in sorted(KNOWN_TASKS, key=len, reverse=True):
		if dirname == t or dirname.startswith(t + '_'):
			return t
	return None


def infer_retriever_from_dirname(dirname: str) -> str:
	"""Infer retriever name from a run directory name.

	Expected pattern examples:
	- biology_bm25_long_False -> retriever "bm25"
	- earth_science_bge-reasoner_long_True -> retriever "bge-reasoner"
	- psychology_grit -> retriever "grit"

	When task prefix is unknown, returns the dirname unchanged.
	"""
	task = infer_task_from_dirname(dirname)
	if task is None:
		return dirname
	# Remove the task prefix and optional underscore
	rest = dirname[len(task):]
	if rest.startswith('_'):
		rest = rest[1:]
	# Strip trailing markers like _long_True / _long_False
	if '_long_' in rest:
		rest = rest.split('_long_')[0]
	return rest if rest else dirname


def main():
	parser = argparse.ArgumentParser(description='Evaluate weighted alpha-nDCG@K over score.json files')
	parser.add_argument('--task', type=str, default='biology',
					choices=KNOWN_TASKS + ['all'])
	parser.add_argument('--k', type=int, default=25)
	parser.add_argument('--alpha', type=float, default=0.5)
	parser.add_argument('--score_file', type=str, default=None)
	# Paths are relative to the current working directory (project root recommended)
	parser.add_argument('--output_dir', type=str, default='outputs')
	parser.add_argument('--cache_dir', type=str, default='cache')
	parser.add_argument('--all', action='store_true', help='process all score.json files for the task (ignored when --task all)')
	parser.add_argument('--save_json', type=str, default=None, help='path to write all results as a single JSON file')
	parser.add_argument('--save_excel', type=str, default=None, help='path to write retrievers x tasks matrix to an Excel .xlsx file')
	args = parser.parse_args()

	files = find_score_files(args.task, args.output_dir, args.score_file, args.all)
	if not files:
		print(f"No score.json files found for task '{args.task}'.")
		return

	results = []
	if args.task == 'all':
		# Build per-task datasets/maps lazily and cache
		cache_per_task: Dict[str, Tuple] = {}
		for sf in files:
			dirname = sf.parent.name
			task = infer_task_from_dirname(dirname)
			if task is None:
				# Skip unknown task folders
				continue
			if task not in cache_per_task:
				examples = load_bright_pro('examples', task)
				doc_id_to_aspect_id, aspect_id_to_weight = build_aspect_maps(task, args.cache_dir)
				cache_per_task[task] = (examples, doc_id_to_aspect_id, aspect_id_to_weight)
			examples, doc_id_to_aspect_id, aspect_id_to_weight = cache_per_task[task]
			res = evaluate_file(sf, examples, doc_id_to_aspect_id, aspect_id_to_weight, alpha=args.alpha, k=args.k)
			# annotate result with inferred task
			res['task'] = task
			results.append(res)
	else:
		# Single task mode
		examples = load_bright_pro('examples', args.task)
		doc_id_to_aspect_id, aspect_id_to_weight = build_aspect_maps(args.task, args.cache_dir)
		for sf in files:
			res = evaluate_file(sf, examples, doc_id_to_aspect_id, aspect_id_to_weight, alpha=args.alpha, k=args.k)
			res['task'] = args.task
			results.append(res)

	# Print summary
	for r in results:
		print(json.dumps(r, ensure_ascii=False))

	overall = None
	if len(results) > 1:
		avg = sum(r['avg_alpha_ndcg'] for r in results) / len(results)
		overall = {'mode': args.task, 'files': len(results), 'avg_alpha_ndcg': avg, 'alpha': args.alpha, 'k': args.k}
		print(json.dumps(overall, ensure_ascii=False))

	# Optional save to JSON
	if args.save_json:
		payload = {
			'mode': args.task,
			'alpha': args.alpha,
			'k': args.k,
			'files': len(results),
			'results': results
		}
		if overall is not None:
			payload['aggregate'] = overall
		with open(args.save_json, 'w') as f:
			json.dump(payload, f, ensure_ascii=False, indent=2)

	# Optional save to Excel (retrievers as rows, tasks as columns)
	if args.save_excel:
		if pd is None:
			print("pandas is not installed; cannot write Excel. Install pandas and openpyxl to enable --save_excel.")
		else:
			# Build matrix: retriever -> task -> avg_alpha_ndcg
			matrix = {}
			seen_tasks = set()
			for r in results:
				dirname = Path(r['file']).parent.name
				retriever = infer_retriever_from_dirname(dirname)
				task = r.get('task') or infer_task_from_dirname(dirname) or 'unknown'
				seen_tasks.add(task)
				matrix.setdefault(retriever, {})[task] = r.get('avg_alpha_ndcg', None)

			# Order columns by KNOWN_TASKS where applicable
			ordered_tasks = [t for t in KNOWN_TASKS if t in seen_tasks]
			# Include any extra tasks not in KNOWN_TASKS (unlikely)
			for t in sorted(seen_tasks):
				if t not in ordered_tasks:
					ordered_tasks.append(t)

			retrievers = sorted(matrix.keys())
			data = [[matrix.get(ret, {}).get(t, None) for t in ordered_tasks] for ret in retrievers]
			df = pd.DataFrame(data=data, index=retrievers, columns=ordered_tasks)
			# Write to Excel
			with pd.ExcelWriter(args.save_excel, engine='openpyxl') as writer:
				df.to_excel(writer, sheet_name='alpha_ndcg')


if __name__ == '__main__':
	main()


