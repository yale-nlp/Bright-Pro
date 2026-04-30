#!/usr/bin/env python3
"""LLM-as-Judge scorer — paper §"LLM-as-Judge for System Response Evaluation".

Given a query, the reference answer, and an agent's system answer, calls the
judge LLM and extracts two 1-5 Likert scores:
  * reasoning_completeness — does the system answer cover the reasoning aspects
    and evidence needed to answer the question?
  * overall_quality — overall answer quality relative to the reference.

Inputs:
  * --runs-dir      walks ``<dir>/<retriever>/<task>/run_*.json`` and grades each
                    ``final_answer`` item (fixed-round runs have one per round).
  * --answers-jsonl (alternative) each line a JSON object with keys
                    ``{task, qid, round, answer, retriever?}``; useful for
                    reliability tests where we want to score synthetic or
                    mocked answers.

Outputs:
  * --save-jsonl    append one line per (task, qid, round, rep) with the parsed
                    judge scores, justification, and raw response for audit.

Reliability mode:
  * --reps N        repeat each judgment N times (default 1). Sigma and mean
                    are reported at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
# Bootstrap bright_pro_data (walk up to BRIGHT-PRO/)
_bp = str(_HERE)
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in sys.path:
    sys.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from openai import AzureOpenAI, OpenAI  # noqa: E402


REPO_ROOT = Path(_bp).parent
load_dotenv(REPO_ROOT / ".env")
# Judge model used in the paper.
DEFAULT_DEPLOYMENT = os.environ.get("JUDGE_DEPLOYMENT", "gpt-5")


# -----------------------------------------------------------------------------
# Prompt (the verbatim text below is the judge prompt used in the paper).
# -----------------------------------------------------------------------------

JUDGE_INSTRUCTION = """\
You are an expert evaluator grading a research-assistant's answer.

For each example you receive:
  QUESTION          — a query from a specialized StackExchange community.
  REASONING_ASPECTS — a list of key sub-questions/premises a correct answer
                      must address. Each aspect has a short id (a1, a2, ...).
                      Treat the list as the authoritative rubric.
  REFERENCE_ANSWER  — a citation-grounded answer produced from the gold
                      evidence passages. This is the high-quality target.
  SYSTEM_ANSWER     — the answer produced by the system under evaluation.

Step 1 — score EACH aspect on a 3-point coverage scale:
  1.0 — fully addressed with specific, correctly-supported claims
  0.5 — partially addressed (mentioned but shallow, OR addressed with notable
        inaccuracies, OR right idea but missing critical detail)
  0.0 — not addressed, off-topic, or factually wrong
You MUST grade every aspect id given in REASONING_ASPECTS — do not skip any.

Step 2 — assign one holistic overall_quality score (1–5 integer) for
SYSTEM_ANSWER relative to REFERENCE_ANSWER (correctness, structure, citations,
coherence, no hallucinations).
  5 — matches or exceeds REFERENCE_ANSWER
  4 — slightly worse but still correct and well-structured
  3 — correct but less thorough or less clearly structured
  2 — partially correct with notable issues
  1 — mostly wrong or hallucinated

Return STRICTLY a single JSON object, no markdown fences, no prose outside the
object:
{"aspect_scores": {"a1": <0|0.5|1>, "a2": <0|0.5|1>, ...}, "overall_quality": <int 1-5>, "justification": "<1-2 sentence rationale>"}
"""


JUDGE_USER_TEMPLATE = """\
QUESTION:
{question}

REASONING_ASPECTS:
{aspects_block}

REFERENCE_ANSWER:
{reference_answer}

SYSTEM_ANSWER:
{system_answer}
"""


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

@dataclass
class JudgeItem:
    task: str
    qid: str
    round: Optional[int]
    retriever: Optional[str]
    system_answer: str


def _iter_task_dirs(runs_dir: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (retriever, task_dir). Supports both layouts:
      - runs_dir/<retriever>/<task>/
      - runs_dir/<retriever>/<benchmark>/<task>/  (e.g. bright-pro)
    """
    for retriever_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for child in sorted(p for p in retriever_dir.iterdir() if p.is_dir()):
            if child.name in SE_TASKS:
                yield retriever_dir.name, child
            else:
                # benchmark level (e.g. bright-pro)
                for task_dir in sorted(p for p in child.iterdir() if p.is_dir()):
                    if task_dir.name in SE_TASKS:
                        yield retriever_dir.name, task_dir


def _iter_runs_dir(runs_dir: Path) -> Iterable[JudgeItem]:
    """Walk runs and yield one JudgeItem per final answer.

    fixed-round agentic (fixed-turn): emits ``type: final_answer`` items, one per round.
    adaptive-round agentic (adaptive):  emits a single ``type: output_text`` item (round=None).
    """
    for retriever_name, task_dir in _iter_task_dirs(runs_dir):
        task = task_dir.name
        for rf in sorted(task_dir.glob("run_*.json")):
            try:
                data = json.loads(rf.read_text())
            except Exception as e:
                print(f"  skip malformed {rf}: {e}", file=sys.stderr)
                continue
            qid = str(data.get("query_id") or "")
            if not qid:
                continue
            yielded_any = False
            # fixed-round agentic: per-round final_answer
            for item in data.get("result") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "final_answer":
                    continue
                ans = item.get("output") or ""
                if not ans.strip():
                    continue
                yielded_any = True
                yield JudgeItem(
                    task=task, qid=qid,
                    round=item.get("round"),
                    retriever=retriever_name,
                    system_answer=ans,
                )
            if yielded_any:
                continue
            # adaptive-round agentic: single output_text (last one wins)
            adaptive_text = None
            for item in data.get("result") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "output_text":
                    t = item.get("output") or item.get("text") or ""
                    if t.strip():
                        adaptive_text = t
            if adaptive_text:
                yield JudgeItem(
                    task=task, qid=qid,
                    round=None,
                    retriever=retriever_name,
                    system_answer=adaptive_text,
                )


def _iter_answers_jsonl(path: Path) -> Iterable[JudgeItem]:
    """Each line: {task, qid, round?, retriever?, answer}."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        yield JudgeItem(
            task=row["task"], qid=str(row["qid"]),
            round=row.get("round"), retriever=row.get("retriever"),
            system_answer=row["answer"],
        )


# -----------------------------------------------------------------------------
# Context lookup: question + aspects + reference answer from yale-nlp/Bright-Pro
# -----------------------------------------------------------------------------

_context_cache: Dict[Tuple[str, str], Dict] = {}


def _load_context(task: str, qid: str) -> Dict:
    key = (task, qid)
    if key in _context_cache:
        return _context_cache[key]

    ex_list = load_bright_pro("examples", task)
    asp_list = load_bright_pro("aspects", task)

    ex = next((e for e in ex_list if str(e["id"]) == qid), None)
    if ex is None:
        raise KeyError(f"no example for (task={task}, qid={qid})")

    # Aspects belonging to this query — id pattern `{task}-{qid}-a{k}` or legacy.
    suffix_re = re.compile(r"-(?:a\d+|aspect-\d+)$")
    my_asps = [
        a for a in asp_list
        if suffix_re.sub("", str(a["id"])) == f"{task}-{qid}"
    ]
    # Stable order by the trailing k
    def _k(a):
        m = re.search(r"-(?:a|aspect-)(\d+)$", str(a["id"]))
        return int(m.group(1)) if m else 0
    my_asps.sort(key=_k)

    # Per-aspect weights: prefer raw weight on the aspect record (Likert {1,2,3}).
    # Normalized later by the consumer.
    weights: List[float] = []
    for a in my_asps:
        w = a.get("weight")
        try:
            weights.append(float(w) if w is not None else 1.0)
        except Exception:
            weights.append(1.0)

    ctx = {
        "question": ex["query"],
        "reference_answer": ex.get("reference_answer") or "",
        "aspects": [{"id": a["id"], "content": a["content"]} for a in my_asps],
        "aspect_weights": weights,
    }
    _context_cache[key] = ctx
    return ctx


def _fmt_aspects(aspects: List[Dict]) -> str:
    """Render aspects as 'a1: <content>' lines for the per-aspect rubric."""
    if not aspects:
        return "(no aspects annotated for this query)"
    return "\n".join(f"  a{i+1}: {a['content']}" for i, a in enumerate(aspects))


# -----------------------------------------------------------------------------
# Judge call
# -----------------------------------------------------------------------------

def make_client(model: str = ""):
    """Pick the right client based on available keys.

    - if OPENAI_API_KEY is set → direct OpenAI (uses the Responses API)
    - else AzureOpenAI
    """
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def _call_judge(client, deployment: str, user_content: str,
                max_output_tokens: int = 4000) -> Tuple[Optional[Dict], str, Dict]:
    """Returns (parsed_dict_or_None, raw_response_text, usage_dict)."""
    raw = ""
    usage: Dict[str, int] = {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}

    # Responses API (OpenAI gpt-5 family / Azure)
    resp = client.responses.create(
        model=deployment,
        input=[
            {"role": "system", "content": JUDGE_INSTRUCTION},
            {"role": "user", "content": user_content},
        ],
        max_output_tokens=max_output_tokens,
    )
    text_parts: List[str] = []
    for out in getattr(resp, "output", []) or []:
        if getattr(out, "type", None) == "message":
            for piece in getattr(out, "content", []) or []:
                t = getattr(piece, "text", None)
                if t:
                    text_parts.append(t)
    raw = "\n".join(text_parts).strip()
    u = getattr(resp, "usage", None)
    if u is not None:
        usage["input"] = int(getattr(u, "input_tokens", 0) or 0)
        usage["output"] = int(getattr(u, "output_tokens", 0) or 0)
        details = getattr(u, "input_tokens_details", None)
        if details is not None:
            usage["cached_input"] = int(getattr(details, "cached_tokens", 0) or 0)
        out_det = getattr(u, "output_tokens_details", None)
        if out_det is not None:
            usage["reasoning"] = int(getattr(out_det, "reasoning_tokens", 0) or 0)

    # Strip potential code fences
    stripped = raw
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)

    # Attempt JSON parse
    parsed: Optional[Dict] = None
    try:
        parsed = json.loads(stripped)
    except Exception:
        # Greedier match — capture the outermost JSON object containing
        # overall_quality. Use balanced-brace search rather than [^{}] which
        # can't span the nested aspect_scores dict.
        m = re.search(r"\{.*?\"overall_quality\".*\}", stripped, re.DOTALL)
        if m:
            for candidate in (m.group(0), m.group(0) + "}"):
                try:
                    parsed = json.loads(candidate)
                    break
                except Exception:
                    parsed = None

    # Validate shape: aspect_scores dict + integer overall_quality
    if parsed is not None:
        oq = parsed.get("overall_quality")
        ascores = parsed.get("aspect_scores")
        ok_oq = isinstance(oq, (int, float)) and 1 <= oq <= 5
        ok_as = isinstance(ascores, dict) and len(ascores) > 0 and all(
            isinstance(v, (int, float)) and v in (0, 0.5, 1)
            for v in ascores.values()
        )
        if not (ok_oq and ok_as):
            parsed = None

    return parsed, raw, usage


def judge_item(client, deployment: str, item: JudgeItem,
               max_retries: int = 3) -> Dict:
    """Grade one (task, qid, round) — returns a result row (one rep)."""
    ctx = _load_context(item.task, item.qid)
    user_content = JUDGE_USER_TEMPLATE.format(
        question=ctx["question"],
        aspects_block=_fmt_aspects(ctx["aspects"]),
        reference_answer=ctx["reference_answer"] or "(no reference answer)",
        system_answer=item.system_answer,
    )

    attempts = []
    cum_usage: Dict[str, int] = {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
    for attempt in range(max_retries + 1):
        t0 = time.time()
        parsed, raw, usage = _call_judge(client, deployment, user_content)
        for k, v in usage.items():
            cum_usage[k] = cum_usage.get(k, 0) + int(v)
        attempts.append({
            "attempt": attempt,
            "ok": parsed is not None,
            "elapsed_s": round(time.time() - t0, 2),
            "usage": usage,
        })
        if parsed is not None:
            # Aggregate per-aspect → coverage (uniform mean) and weighted coverage
            ascores_raw = parsed.get("aspect_scores") or {}
            n_aspects = len(ctx["aspects"])
            weights = ctx["aspect_weights"] or [1.0] * n_aspects
            ordered: List[float] = []
            for i in range(n_aspects):
                key = f"a{i+1}"
                v = ascores_raw.get(key)
                if v is None:  # tolerate alt keys (full aspect ids)
                    v = ascores_raw.get(ctx["aspects"][i]["id"])
                ordered.append(float(v) if v is not None else 0.0)
            coverage = sum(ordered) / n_aspects if n_aspects else 0.0
            wsum = sum(weights[:n_aspects]) or 1.0
            weighted_cov = sum(w * s for w, s in zip(weights, ordered)) / wsum
            # Backwards-compat reasoning_completeness on a 1-5 scale
            rc = int(round(weighted_cov * 4 + 1))  # 0→1, 1→5
            return {
                "task": item.task, "qid": item.qid, "round": item.round,
                "retriever": item.retriever,
                "aspect_scores": {f"a{i+1}": ordered[i] for i in range(n_aspects)},
                "aspect_coverage": round(coverage, 4),
                "weighted_aspect_coverage": round(weighted_cov, 4),
                "reasoning_completeness": rc,
                "overall_quality": int(parsed["overall_quality"]),
                "justification": parsed.get("justification") or "",
                "attempts": attempts,
                "usage": cum_usage,
                "model": deployment,
                "raw": raw,
                "status": "ok",
            }
        time.sleep(1)
    return {
        "task": item.task, "qid": item.qid, "round": item.round,
        "retriever": item.retriever,
        "aspect_scores": None,
        "aspect_coverage": None, "weighted_aspect_coverage": None,
        "reasoning_completeness": None, "overall_quality": None,
        "justification": "", "attempts": attempts, "raw": raw,
        "usage": cum_usage,
        "model": deployment,
        "status": "failed",
    }


# -----------------------------------------------------------------------------
# Reliability summary
# -----------------------------------------------------------------------------

def _reliability_summary(results: List[Dict]) -> Dict:
    """Group results by (task, qid, round, retriever) and compute mean/σ of the
    two scores across repeats."""
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in results:
        if r["status"] != "ok":
            continue
        key = (r["task"], r["qid"], r.get("round"), r.get("retriever"))
        groups[key].append(r)

    rc_sigmas, oq_sigmas = [], []
    per_group = []
    for key, rows in groups.items():
        if len(rows) <= 1:
            continue
        rcs = [r["reasoning_completeness"] for r in rows]
        oqs = [r["overall_quality"] for r in rows]
        rc_mean, rc_sigma = mean(rcs), pstdev(rcs)
        oq_mean, oq_sigma = mean(oqs), pstdev(oqs)
        rc_sigmas.append(rc_sigma)
        oq_sigmas.append(oq_sigma)
        per_group.append({
            "key": {"task": key[0], "qid": key[1], "round": key[2], "retriever": key[3]},
            "n_reps": len(rows),
            "reasoning_completeness": {"values": rcs, "mean": rc_mean, "sigma": rc_sigma},
            "overall_quality": {"values": oqs, "mean": oq_mean, "sigma": oq_sigma},
        })

    return {
        "n_groups_with_repeats": len(per_group),
        "avg_sigma_reasoning_completeness": mean(rc_sigmas) if rc_sigmas else 0.0,
        "avg_sigma_overall_quality": mean(oq_sigmas) if oq_sigmas else 0.0,
        "max_sigma_reasoning_completeness": max(rc_sigmas) if rc_sigmas else 0.0,
        "max_sigma_overall_quality": max(oq_sigmas) if oq_sigmas else 0.0,
        "per_group": per_group,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="LLM-as-Judge scorer (GPT-5).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs-dir", type=Path,
                     help="path to runs/ dir (runs/<retriever>/<task>/run_*.json)")
    src.add_argument("--answers-jsonl", type=Path,
                     help="JSONL: {task, qid, round?, retriever?, answer} per line")
    p.add_argument("--save-jsonl", type=Path, required=True)
    p.add_argument("--reps", type=int, default=1,
                   help="number of independent judge calls per item (reliability)")
    p.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    p.add_argument("--limit", type=int, default=0,
                   help="cap total number of items graded (for smoke testing)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip items already present in --save-jsonl (match on task+qid+round+rep)")
    p.add_argument("--num-threads", type=int, default=1,
                   help="parallel judge calls (each request is independent)")
    p.add_argument("--retrievers", nargs="+", default=None,
                   help="only judge these retriever subdirectories under --runs-dir")
    args = p.parse_args()

    # Collect items
    if args.runs_dir:
        items = list(_iter_runs_dir(args.runs_dir))
        if args.retrievers:
            allow = set(args.retrievers)
            items = [it for it in items if it.retriever in allow]
    else:
        items = list(_iter_answers_jsonl(args.answers_jsonl))
    if args.limit:
        items = items[: args.limit]
    print(f"[judge] {len(items)} items × {args.reps} reps = {len(items) * args.reps} calls")

    # Determine done-set for resume
    done: set = set()
    if args.skip_existing and args.save_jsonl.exists():
        for line in args.save_jsonl.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task"], r["qid"], r.get("round"), r.get("retriever"), r.get("rep", 0)))
            except Exception:
                pass

    client = make_client(args.deployment)
    all_results: List[Dict] = []
    args.save_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Build the work queue (skipping already-done items)
    jobs: List[Tuple[int, JudgeItem, int]] = []
    for i, it in enumerate(items):
        for rep in range(args.reps):
            key = (it.task, it.qid, it.round, it.retriever, rep)
            if key in done:
                continue
            jobs.append((i, it, rep))
    print(f"[judge] {len(jobs)} pending after skip-existing  threads={args.num_threads}")

    write_lock = threading.Lock()
    progress = {"n": 0}

    def _run(i: int, it: JudgeItem, rep: int) -> Dict:
        r = judge_item(client, args.deployment, it)
        r["rep"] = rep
        return r

    with args.save_jsonl.open("a", encoding="utf-8") as out_f:
        if args.num_threads <= 1:
            for i, it, rep in jobs:
                r = _run(i, it, rep)
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                out_f.flush()
                all_results.append(r)
                progress["n"] += 1
                if progress["n"] % 10 == 0 or progress["n"] == len(jobs):
                    print(f"  [{progress['n']}/{len(jobs)}] {it.task}/qid={it.qid} round={it.round} status={r['status']}")
        else:
            with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
                futs = {ex.submit(_run, i, it, rep): (i, it, rep) for i, it, rep in jobs}
                for fut in as_completed(futs):
                    i, it, rep = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        print(f"  [err] {it.task}/qid={it.qid} round={it.round}: {e}", file=sys.stderr)
                        continue
                    with write_lock:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        out_f.flush()
                        all_results.append(r)
                        progress["n"] += 1
                        if progress["n"] % 10 == 0 or progress["n"] == len(jobs):
                            print(f"  [{progress['n']}/{len(jobs)}] {it.task}/qid={it.qid} round={it.round} status={r['status']}")

    n_ok = sum(1 for r in all_results if r["status"] == "ok")
    print(f"\n[judge] {n_ok}/{len(all_results)} succeeded")

    # Cost summary (this run only — does not include prior --skip-existing rows)
    PRICING = {
        "gpt-5":              {"input": 1.25, "cached": 0.125, "output": 10.00},
        "gpt-5-mini":         {"input": 0.25, "cached": 0.025, "output":  2.00},
    }
    tot = {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}
    for r in all_results:
        u = r.get("usage") or {}
        for k in tot:
            tot[k] += int(u.get(k, 0))
    price = PRICING.get(args.deployment, PRICING["gpt-5"])
    in_cost = (tot["input"] - tot["cached_input"]) / 1e6 * price["input"]
    cached_cost = tot["cached_input"] / 1e6 * price["cached"]
    out_cost = tot["output"] / 1e6 * price["output"]
    total = in_cost + cached_cost + out_cost
    print(f"\n[cost] model={args.deployment} this-run-only ({len(all_results)} rows)")
    print(f"  input  = {tot['input']:>10d} tokens (${in_cost:.4f})")
    print(f"  cached = {tot['cached_input']:>10d} tokens (${cached_cost:.4f})")
    print(f"  output = {tot['output']:>10d} tokens (${out_cost:.4f}, of which reasoning={tot['reasoning']})")
    print(f"  TOTAL  = ${total:.4f}")
    cost_path = args.save_jsonl.with_name(args.save_jsonl.stem + "__cost.json")
    cost_path.write_text(json.dumps({
        "model": args.deployment, "rows": len(all_results), "succeeded": n_ok,
        "tokens": tot, "pricing_per_1M": price, "cost_usd": {
            "input": round(in_cost, 4), "cached": round(cached_cost, 4),
            "output": round(out_cost, 4), "total": round(total, 4),
        },
    }, indent=2))
    print(f"  saved: {cost_path}")

    if args.reps > 1:
        summary = _reliability_summary(all_results)
        summary_path = args.save_jsonl.with_name(args.save_jsonl.stem + "__reliability.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n[reliability] n_groups_with_repeats={summary['n_groups_with_repeats']}")
        print(f"  avg σ(reasoning_completeness) = {summary['avg_sigma_reasoning_completeness']:.3f}")
        print(f"  avg σ(overall_quality)        = {summary['avg_sigma_overall_quality']:.3f}")
        print(f"  max σ(reasoning_completeness) = {summary['max_sigma_reasoning_completeness']:.3f}")
        print(f"  max σ(overall_quality)        = {summary['max_sigma_overall_quality']:.3f}")
        print(f"  full report: {summary_path}")


if __name__ == "__main__":
    main()
