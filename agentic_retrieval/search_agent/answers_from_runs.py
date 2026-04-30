import os
import json
import sys as _sys_bp
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Bootstrap sys.path so 'searcher.*' and 'bright_pro_data' resolve regardless
# of where this file is invoked from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTIC_ROOT = os.path.dirname(_HERE)        # agentic_retrieval/
if _AGENTIC_ROOT not in _sys_bp.path:
    _sys_bp.path.insert(0, _AGENTIC_ROOT)
_bp = _HERE
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in _sys_bp.path:
    _sys_bp.path.insert(0, _bp)

from searcher.searchers import SearcherType                # noqa: E402
from searcher.searchers.base import BaseSearcher            # noqa: E402
from openai import AzureOpenAI                              # noqa: E402
from bright_pro_data import load_bright_pro, SE_TASKS       # noqa: E402


def collect_runs(runs_root: Path) -> List[Path]:
    return sorted(runs_root.rglob("run_*.json"))


def extract_per_round_docids(result_items: List[dict]) -> Dict[int, List[str]]:
    round_to_docids: Dict[int, List[str]] = defaultdict(list)
    for item in result_items:
        if item.get("type") == "tool_result" and item.get("tool_name") == "search":
            rnd = item.get("round")
            output = item.get("output")
            try:
                parsed = json.loads(output) if isinstance(output, str) else output
            except Exception:
                parsed = None
            if rnd and isinstance(parsed, list):
                for r in parsed:
                    if isinstance(r, dict) and r.get("docid"):
                        round_to_docids[rnd].append(r["docid"])
    return round_to_docids


def cumulative_docids(round_to_docids: Dict[int, List[str]], upto: int) -> List[str]:
    seen = set()
    combined: List[str] = []
    for r in range(1, upto + 1):
        for d in round_to_docids.get(r, []):
            if d not in seen:
                seen.add(d)
                combined.append(d)
    return combined


def get_documents(searcher: BaseSearcher, docids: List[str]) -> List[Dict]:
    docs: List[Dict] = []
    for d in docids:
        got = searcher.get_document(d)
        if got:
            docs.append(got)
    return docs


def collect_items_upto_round(result_items: List[dict], upto: int) -> Dict[str, List[dict]]:
    """Collect tool_call and tool_result items up to a given round, preserving original structure."""
    calls: List[dict] = []
    results: List[dict] = []
    for item in result_items:
        itype = item.get("type")
        rnd = item.get("round")
        if itype in ("tool_call", "tool_result") and isinstance(rnd, int) and rnd <= upto:
            if itype == "tool_call":
                calls.append(item)
            else:
                results.append(item)
    return {"tool_calls": calls, "tool_results": results}


def make_prompt(query: str, docs: List[Dict]) -> str:
    """Build the fixed-round answer prompt (paper Figure 4).

    Uses the shared ``QUERY_TEMPLATE_ORACLE`` template so every fixed-round
    answer generation path (``openai_fixed_turn`` AND this post-hoc rerun
    path) uses identical wording. Evidence documents are rendered as
    ``[{docid}] {text}`` — matching the paper's citation convention where
    the model cites ``[20]`` with the actual doc id."""
    from prompts import QUERY_TEMPLATE_ORACLE

    lines = []
    for d in docs:
        docid = d.get("docid") or ""
        body = d.get("text") or d.get("snippet") or ""
        lines.append(f"[{docid}] {body}")
    evidence = "\n".join(lines)
    return QUERY_TEMPLATE_ORACLE.format(Question=query, EvidenceDocuments=evidence)


def generate_answer(client: AzureOpenAI, model: str, prompt: str, max_tokens: int = 2000) -> str:
    resp = client.responses.create(
        model=model,
        max_output_tokens=max_tokens,
        input=[{"role": "user", "content": prompt}],
    )
    texts: List[str] = []
    for out in getattr(resp, "output", []) or []:
        if getattr(out, "type", None) == "message":
            for piece in getattr(out, "content", []) or []:
                text_val = getattr(piece, "text", None)
                if text_val:
                    texts.append(text_val)
    return "\n".join(texts).strip()


def load_question_text(task: str, qid: str) -> str:
    table = load_bright_pro('examples', task)
    for row in table:
        if str(row['id']) == str(qid):
            return row['query']
    return ""


def process_run(client: AzureOpenAI, run_path: Path, model: str, searcher: BaseSearcher) -> Tuple[Path, Dict[int, str]]:
    with run_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result_items = data.get("result", [])
    qid = str(data.get("query_id"))
    task = run_path.parent.name
    query_text = load_question_text(task, qid)
    round_docs = extract_per_round_docids(result_items)

    answers: Dict[int, str] = {}
    max_round = min(3, max(round_docs.keys()) if round_docs else 0)
    for n in range(1, max_round + 1):
        docids = cumulative_docids(round_docs, n)
        docs = get_documents(searcher, docids)
        prompt = make_prompt(query_text, docs)
        answer = generate_answer(client, model=model, prompt=prompt)
        answers[n] = answer

    out_file = run_path.with_name(run_path.stem + "_answers_rounds.json")
    # Also include natural function-call format up to each round
    per_round_fc = {}
    for n in range(1, max_round + 1):
        per_round_fc[str(n)] = collect_items_upto_round(result_items, n)

    payload = {
        "query_id": qid,
        "task": task,
        "model": model,
        "rounds_available": sorted(list(round_docs.keys())),
        "answers": answers,
        "natural_function_calls": per_round_fc,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_file, answers


def init_searcher(cli_name: str, task: str, cache_dir: str | None = None) -> BaseSearcher:
    class Args:
        pass
    args = Args()
    args.task = task
    args.cache_dir = cache_dir or "cache"
    args.bm25_k1 = 0.9
    args.bm25_b = 0.4
    searcher_class = SearcherType.get_searcher_class(cli_name)
    return searcher_class(args)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate answers from prior runs for rounds 1..3")
    parser.add_argument("--runs-root", default="runs/gpt-5-mini-08-07", help="Root folder containing runs")
    parser.add_argument("--searcher", required=True, choices=SearcherType.get_choices(), help="Searcher type matching the run (e.g., bm25, bge)")
    parser.add_argument("--task", required=True, choices=['biology','earth_science','economics','psychology','robotics','stackoverflow','sustainable_living'])
    parser.add_argument("--model", default="gpt-5-mini-08-07", help="Answering model")
    parser.add_argument("--azure-endpoint", default=os.getenv("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--api-version", default=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"))
    parser.add_argument("--api-key", default=os.getenv("AZURE_OPENAI_API_KEY"),
                        help="Azure API key (or set AZURE_OPENAI_API_KEY)")
    parser.add_argument("--cache-dir", default="cache", help="HF cache dir")
    args = parser.parse_args()

    # Prefer public OpenAI when OPENAI_API_KEY is set; otherwise fall back to Azure.
    if os.getenv("OPENAI_API_KEY"):
        import openai as _openai
        client = _openai.OpenAI()
    else:
        if not args.api_key or not args.azure_endpoint:
            raise RuntimeError("Neither OPENAI_API_KEY nor (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT) set")
        client = AzureOpenAI(api_version=args.api_version, azure_endpoint=args.azure_endpoint, api_key=args.api_key)

    runs_root = Path(args.runs_root).expanduser().resolve() / args.searcher / args.task
    run_files = collect_runs(runs_root)
    if not run_files:
        print(f"No run_*.json found under {runs_root}")
        return

    searcher = init_searcher(args.searcher, task=args.task, cache_dir=args.cache_dir)

    for rp in run_files:
        try:
            out_path, answers = process_run(client, rp, model=args.model, searcher=searcher)
            print(f"Wrote {out_path} | rounds: {sorted(answers.keys())}")
        except Exception as e:
            print(f"Failed {rp}: {e}")


if __name__ == "__main__":
    main()


