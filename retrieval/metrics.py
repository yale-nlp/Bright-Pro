"""BEIR-style retrieval metrics (NDCG@k / Recall@k / MAP@k / P@k / MRR).

Lifted out of `retrievers.py` so callers (and tests) can import the metric
computation without pulling in every retriever backend (torch, cohere,
voyageai, gritlm, ...). Only `pytrec_eval` is needed here.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

import pytrec_eval


def calculate_retrieval_metrics(
    results: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    k_values: List[int] = [1, 5, 10, 25, 50, 100],
) -> Dict[str, float]:
    """Average NDCG@k / Recall@k / MAP@k / P@k / MRR across queries.

    Parameters
    ----------
    results : {qid: {doc_id: score}}
    qrels   : {qid: {doc_id: binary_relevance}}
    k_values: list of cut-offs

    Returns a flat dict with keys like 'NDCG@25', 'Recall@25', 'MAP@10', 'MRR'.
    """
    ndcg: Dict[str, float] = {}
    _map: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    precision: Dict[str, float] = {}
    mrr = {"MRR": 0.0}

    # pytrec_eval wants string keys both at outer and inner level.
    qrels = {str(qid): {str(pid): int(rel) for pid, rel in rels.items()}
             for qid, rels in qrels.items()}
    results = {str(qid): {str(pid): float(score) for pid, score in docs.items()}
               for qid, docs in results.items()}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join(map(str, k_values))
    ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
    recall_string = "recall." + ",".join(map(str, k_values))
    precision_string = "P." + ",".join(map(str, k_values))

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {map_string, ndcg_string, recall_string, precision_string, "recip_rank"},
    )
    scored = evaluator.evaluate(results)

    for qid in scored:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scored[qid][f"ndcg_cut_{k}"]
            _map[f"MAP@{k}"] += scored[qid][f"map_cut_{k}"]
            recall[f"Recall@{k}"] += scored[qid][f"recall_{k}"]
            precision[f"P@{k}"] += scored[qid][f"P_{k}"]
        mrr["MRR"] += scored[qid]["recip_rank"]

    n = max(1, len(scored))
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / n, 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / n, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / n, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / n, 5)
    mrr["MRR"] = round(mrr["MRR"] / n, 5)

    return {**ndcg, **_map, **recall, **precision, **mrr}
