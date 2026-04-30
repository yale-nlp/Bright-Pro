import os
import json
from datetime import datetime
import argparse
from pathlib import Path
import csv
import sys
from tqdm import tqdm
from prompts import format_query, QUERY_TEMPLATE_ORACLE
from concurrent.futures import ThreadPoolExecutor, as_completed
_bp = os.path.dirname(os.path.abspath(__file__))
while _bp != "/" and not os.path.isfile(os.path.join(_bp, "bright_pro_data.py")):
    _bp = os.path.dirname(_bp)
if _bp not in sys.path:
    sys.path.insert(0, _bp)
from bright_pro_data import load_bright_pro, SE_TASKS
import openai
from openai import AzureOpenAI


import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from searcher.searchers import SearcherType
from utils import extract_retrieved_docids_from_result
from transformers import AutoTokenizer


class SearchToolHandler:
    def __init__(self, searcher, snippet_max_tokens: int | None = None, k: int = 5, include_get_document: bool = False):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document
        
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        if self.include_get_document:
            tools.append({
                "type": "function",
                "name": "get_document",
                "description": self.searcher.get_document_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {
                            "type": "string",
                            "description": "Document ID to retrieve"
                        }
                    },
                    "required": ["docid"]
                }
            })
        
        return tools
    
    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)
        
        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[:self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]
        
        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({
                    "docid": cand["docid"],
                    "snippet": cand["snippet"]
                })
            else:
                results.append({
                    "docid": cand["docid"],
                    "score": cand["score"],
                    "snippet": cand["snippet"]
                })
        
        return json.dumps(results, indent=2)
    
    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


def build_request(
    query: str,
    model: str,
    max_tokens: int,
    tool_handler: SearchToolHandler,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
    query_template: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
    formatted_query = format_query(query, query_template)
    
    input_messages = [{"role": "user", "content": formatted_query}]
    
    body = {
        "model": model,
        "max_output_tokens": max_tokens,
        "input": input_messages,
        "tools": tool_handler.get_tool_definitions(),
        "tool_choice": "auto",
        # "truncation": "auto",
    }
    
    if not model.lower().startswith("o"):
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
    
    if system_prompt:
        body["instructions"] = system_prompt
    
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}
    
    return body


def _extract_text_from_response(response) -> str:
    
    """Extract concatenated text content from a Responses API response."""
    try:
        outputs = getattr(response, 'output', []) or []
        chunks = []
        for out in outputs:
            if getattr(out, 'type', None) == 'message':
                content = getattr(out, 'content', []) or []
                for piece in content:
                    text_val = getattr(piece, 'text', None)
                    if text_val:
                        chunks.append(text_val)
        return "\n".join(chunks).strip()
    except Exception:
        print(response)
        return ""


def _collect_docs_upto_round(combined_output: list[dict], upto_round: int) -> list[dict]:
    """Collect unique docids and snippets from search tool results up to a given round."""
    seen = set()
    docs: list[dict] = []
    for item in combined_output:
        if item.get("type") == "tool_result" and item.get("tool_name") == "search":
            round_idx = item.get("round")
            if round_idx is None or round_idx > upto_round:
                continue
            try:
                parsed = json.loads(item.get("output") or "[]")
                if isinstance(parsed, list):
                    for sr in parsed:
                        if isinstance(sr, dict):
                            docid = sr.get("docid")
                            snippet = sr.get("snippet")
                            if docid and docid not in seen:
                                seen.add(docid)
                                docs.append({"docid": docid, "snippet": snippet})
            except Exception:
                continue
    return docs


def _make_oracle_prompt(question: str, docs: list[dict]) -> str:
    lines = []
    for d in docs:
        docid = d.get("docid")
        snippet = d.get("snippet") or ""
        lines.append(f"[{docid}] {snippet}")
    evidence = "\n".join(lines)
    return QUERY_TEMPLATE_ORACLE.format(Question=question, EvidenceDocuments=evidence)


def run_conversation_with_tools(client, initial_request: dict, tool_handler: SearchToolHandler, max_iterations: int = 100, *, finalize_rounds: list[int] | None = None, final_answer_max_tokens: int = 10000, raw_question: str | None = None, temperature: float | None = None, top_p: float | None = None):
    
    input_items = initial_request["input"].copy()
    global_max_tokens = initial_request['max_output_tokens']
    
    cumulative_usage = {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0
    }
    
    combined_output = []  # normalized list of dicts: {type, tool_name?, arguments?, output?}
    tool_outputs = {}
    all_responses = []
    previous_response_id = None
    # vLLM /v1/responses does not persist server-side state — when
    # AGENT_USE_FULL_HISTORY=1 we replay the full input list each round
    # instead of using previous_response_id.
    use_full_history = os.environ.get("AGENT_USE_FULL_HISTORY", "0") == "1"

    finalize_rounds = sorted(set(finalize_rounds or []))
    stop_after_round = max(finalize_rounds) if finalize_rounds else None
    reached_max_iterations = True   # cleared once we break early
    for iteration in range(max_iterations):
        # Per-round budget: each iteration gets a full `global_max_tokens` slice
        # (not cumulative). Loop termination is governed by max_iterations and
        # the absence of pending function_calls; not by a shared token budget.
        request = {
            "model": initial_request["model"],
            "tools": initial_request.get("tools"),
            "tool_choice": initial_request.get("tool_choice", "auto"),
            "max_output_tokens": global_max_tokens,
            "input": input_items,
        }
        if previous_response_id and not use_full_history:
            request["previous_response_id"] = previous_response_id
        if "instructions" in initial_request:
            request["instructions"] = initial_request["instructions"]
        if "reasoning" in initial_request:
            request["reasoning"] = initial_request["reasoning"]
        if "temperature" in initial_request:
            request["temperature"] = initial_request["temperature"]
        if "top_p" in initial_request:
            request["top_p"] = initial_request["top_p"]
        
        response = client.responses.create(**request)

        # # Save raw response to JSON for inspection
        # try:
        #     raw_out_dir = os.path.join("runs", "raw_chat_responses")
        #     os.makedirs(raw_out_dir, exist_ok=True)
        #     ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        #     raw_filename = os.path.join(
        #         raw_out_dir, f"response_{ts}_iter{iteration+1}.json"
        #     )
        #     try:
        #         raw_data = json.loads(response.model_dump_json())
        #     except Exception:
        #         raw_data = json.loads(response.json())
        #     with open(raw_filename, "w", encoding="utf-8") as rf:
        #         json.dump(raw_data, rf, indent=2, ensure_ascii=False, default=str)
        # except Exception as _e:
        #     print(f"Warning: failed to save raw response JSON: {_e}")
        
        all_responses.append(response)
        previous_response_id = getattr(response, "id", None)
        
        # Accumulate usage if available
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            # Responses API typically exposes input_tokens/output_tokens/total_tokens
            in_tokens = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
            out_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))
            total_tokens = getattr(usage, 'total_tokens', in_tokens + out_tokens)
            cumulative_usage["input_tokens"] += in_tokens
            cumulative_usage["output_tokens"] += out_tokens
            cumulative_usage["total_tokens"] += total_tokens
        
        # Normalize outputs and detect function calls
        function_calls = []
        outputs = getattr(response, 'output', []) or []
        for out in outputs:
            try:
                out_type = getattr(out, 'type', None)
                if out_type == 'message':
                    text_chunks = []
                    content = getattr(out, 'content', []) or []
                    for piece in content:
                        text_val = getattr(piece, 'text', None)
                        if text_val:
                            text_chunks.append(text_val)
                    if text_chunks:
                        msg_text = "\n".join(text_chunks)
                        combined_output.append({
                            "type": "output_text",
                            "tool_name": None,
                            "arguments": None,
                            "output": msg_text,
                        })
                        if use_full_history:
                            input_items.append({"role": "assistant", "content": msg_text})
                elif out_type == 'reasoning':
                    combined_output.append({
                        "type": "output_reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": out.summary,
                    })
                elif out_type == 'function_call':
                    function_calls.append(out)
                    # Also log the tool call for persistence
                    combined_output.append({
                        "type": "tool_call",
                        "tool_name": getattr(out, 'name', None),
                        "arguments": getattr(out, 'arguments', None),
                        "call_id": getattr(out, 'call_id', None),
                        "round": iteration + 1,
                    })
                    if use_full_history:
                        input_items.append({
                            "type": "function_call",
                            "name": getattr(out, 'name', None),
                            "arguments": getattr(out, 'arguments', None),
                            "call_id": getattr(out, 'call_id', None),
                        })
            except Exception:
                continue
        
        if not function_calls:
            # No tool calls; optionally still produce final answers for requested rounds,
            # but avoid duplicating rounds already appended earlier.
            if finalize_rounds and raw_question:
                existing_final_rounds = {
                    item.get("round")
                    for item in combined_output
                    if item.get("type") == "final_answer"
                }
                for n in finalize_rounds:
                    if n in existing_final_rounds:
                        continue
                    docs_upto_n = _collect_docs_upto_round(combined_output, n)
                    oracle_prompt = _make_oracle_prompt(raw_question, docs_upto_n)
                    final_req = {
                        "model": initial_request["model"],
                        "max_output_tokens": final_answer_max_tokens,
                        "input": [{"role": "user", "content": oracle_prompt}],
                    }
                    if temperature is not None:
                        final_req["temperature"] = temperature
                    if top_p is not None:
                        final_req["top_p"] = top_p
                    final_resp = client.responses.create(**final_req)
                    final_text = _extract_text_from_response(final_resp)
                    combined_output.append({
                        "type": "final_answer",
                        "tool_name": None,
                        "arguments": None,
                        "output": final_text,
                        "round": n,
                    })
            return response, combined_output, cumulative_usage, tool_outputs
        
        # Execute tools and prepare next input as function_call_output items
        next_input = []
        for fc in function_calls:
            call_id = getattr(fc, 'call_id', None)
            name = getattr(fc, 'name', None)
            args_raw = getattr(fc, 'arguments', '{}')
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except Exception:
                args = {}
            try:
                result = tool_handler.execute_tool(name, args)
                tool_outputs[call_id] = {
                    "output": result,
                    "status": "completed",
                    "error": None
                }
                # Append a tool_result entry including parsed metadata for persistence
                tool_result_entry = {
                    "type": "tool_result",
                    "tool_name": name,
                    "arguments": args_raw,
                    "output": result,
                    "call_id": call_id,
                    "round": iteration + 1,
                }
                if name == "search":
                    try:
                        sr = json.loads(result)
                        if isinstance(sr, list):
                            docids = [r.get("docid") for r in sr if isinstance(r, dict) and r.get("docid")]
                            if docids:
                                tool_result_entry["docids"] = docids
                    except Exception:
                        pass
                combined_output.append(tool_result_entry)
                next_input.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result,
                })
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                tool_outputs[call_id] = {
                    "output": None,
                    "status": "error",
                    "error": error_msg
                }
                combined_output.append({
                    "type": "tool_result",
                    "tool_name": name,
                    "arguments": args_raw,
                    "output": error_msg,
                    "call_id": call_id,
                    "round": iteration + 1,
                })
                next_input.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": error_msg,
                })
        
        if use_full_history:
            input_items.extend(next_input)
        else:
            input_items = next_input

        # After executing tools for this round, optionally produce a final answer at this round
        current_round = iteration + 1
        if finalize_rounds and current_round in finalize_rounds and raw_question:
            docs_upto_now = _collect_docs_upto_round(combined_output, current_round)
            oracle_prompt = _make_oracle_prompt(raw_question, docs_upto_now)
            final_req = {
                "model": initial_request["model"],
                "max_output_tokens": final_answer_max_tokens,
                "input": [{"role": "user", "content": oracle_prompt}],
            }
            if temperature is not None:
                final_req["temperature"] = temperature
            if top_p is not None:
                final_req["top_p"] = top_p
            final_resp = client.responses.create(**final_req)
            final_text = _extract_text_from_response(final_resp)
            combined_output.append({
                "type": "final_answer",
                "tool_name": None,
                "arguments": None,
                "output": final_text,
                "round": current_round,
            })

        # Stop after the highest requested finalize round
        if stop_after_round is not None and current_round >= stop_after_round:
            reached_max_iterations = False
            break
    
    if reached_max_iterations:
        print(f"Warning: Conversation hit max iterations ({max_iterations})")
    
    final_response = all_responses[-1] if all_responses else response
    return final_response, combined_output, cumulative_usage, tool_outputs


def _persist_response(out_dir: str, request_body: dict, response, combined_output=None, cumulative_usage=None, tool_outputs: dict = None, *, query_id: str | None = None):
    """Persist request & response JSON for later inspection."""
    
    os.makedirs(out_dir, exist_ok=True)
    
    # combined_output is already normalized and in order; persist as-is
    normalized_results: list[dict] = list(combined_output or [])
    retrieved_documents: list[dict] = []
    
    # Collect retrieved documents per round
    for item in normalized_results:
        # Also include search results as retrieved documents (using snippet/score when available)
        if item.get("type") == "tool_result" and item.get("tool_name") == "search":
            try:
                raw_output = item.get("output") or "[]"
                parsed = json.loads(raw_output)
                if isinstance(parsed, list):
                    for sr in parsed:
                        if isinstance(sr, dict) and sr.get("docid"):
                            doc_record = {
                                "docid": sr.get("docid"),
                                # Persist snippet/score inside document for visibility; may be None
                                "document": {
                                    "snippet": sr.get("snippet"),
                                    "score": sr.get("score"),
                                },
                                "round": item.get("round"),
                                "call_id": item.get("call_id"),
                            }
                            retrieved_documents.append(doc_record)
            except Exception:
                pass

    # Count unique rounds with retrievals
    retrieved_round_count = len({d.get("round") for d in retrieved_documents if d.get("round") is not None})
    
    normalized_usage = {
        "input_tokens": cumulative_usage["input_tokens"],
        "input_tokens_cached": cumulative_usage["input_tokens_details"].get("cached_tokens", 0),
        "output_tokens": cumulative_usage["output_tokens"],
        "included_reasoning_tokens": cumulative_usage["output_tokens_details"].get("reasoning_tokens", 0),
        "total_tokens": cumulative_usage["total_tokens"],
    }
    
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(out_dir, f"run_{ts}.json")

    retrieved_documents_id = [d.get("docid") for d in retrieved_documents]
    seen = set()
    unique_ids = []
    for docid in retrieved_documents_id:
        if docid not in seen:
            unique_ids.append(docid)
            seen.add(docid)

    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": request_body.get("model"),
                "reasoning": request_body.get("reasoning"),
                "output_dir": str(out_dir),
                "max_tokens": request_body.get("max_output_tokens"),
            },
            "query_id": query_id,
            "usage": normalized_usage,
            "retrieved_round_count": retrieved_round_count,
            "retrieved_documents_id": unique_ids,
            "status": getattr(response, "status", "completed"),
            "result": normalized_results,
        }, f, indent=2, default=str)
    
    print("Saved response to", filename)


def _process_dataset(tsv_path: str, client, args, tool_handler: SearchToolHandler):
    """Process a TSV file of (id \t query) pairs sequentially and persist responses."""
    
    # dataset_path = Path(tsv_path)
    # if not dataset_path.is_file():
    #     raise FileNotFoundError(f"TSV file not found: {tsv_path}")
    
    # Layout: <output_dir>/<retriever>/<benchmark>/<task>/run_*.json
    # Required by scripts_evaluation/alpha_ndcg.py which globs 3 levels deep.
    args.out_dir = Path(args.output_dir).expanduser().resolve() / args.searcher_type / "bright-pro" / args.task

    queries = []
    examples = load_bright_pro('examples', args.task)
    queries = [(str(ex['id']), ex['query']) for ex in examples]

    # Apply agentic sample filter if requested (fix 25 qids per task, seed=42).
    # args.agentic_sample_ids holds the dict (task -> [qid, ...]) loaded in main().
    sample_ids = getattr(args, '_sample_ids_for_task', None)
    if sample_ids is not None:
        want = set(sample_ids)
        queries = [(qid, qtext) for qid, qtext in queries if qid in want]
        print(f"  [sample] using fixed {len(queries)} of original pool for {args.task}")
    if getattr(args, 'max_queries_per_task', None):
        queries = queries[: args.max_queries_per_task]
        print(f"  [cap] --max-queries-per-task={args.max_queries_per_task} → {len(queries)} queries")
    
    processed_ids = set()
    if args.out_dir.exists():
        for json_path in args.out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue  # ignore corrupt files
    
    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]
    
    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from BRIGHT-PRO {args.task} ..."
    )
    
    import threading
    completed_lock = threading.Lock()
    completed_count = [0]
    
    def _handle_single_query(qid: str, qtext: str, pbar=None):
        request_body = build_request(
            query=qtext,
            model=args.model,
            max_tokens=args.max_tokens,
            tool_handler=tool_handler,
            system_prompt=args.system,
            reasoning_effort=args.reasoning_effort,
            query_template=args.query_template,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        try:
            response, combined_output, cumulative_usage, tool_outputs = run_conversation_with_tools(
                client,
                request_body,
                tool_handler,
                args.max_iterations,
                finalize_rounds=args.finalize_at_rounds,
                final_answer_max_tokens=args.final_answer_max_tokens,
                raw_question=qtext,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            
            status = getattr(response, "status", "completed")
            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])
            
            _persist_response(args.out_dir, request_body, response, combined_output, cumulative_usage, tool_outputs, query_id=qid)
            
        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)
    
    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            futures = [executor.submit(_handle_single_query, qid, qtext, pbar) for qid, qtext in remaining]
            
            for f in as_completed(futures):
                pbar.update(1)
                try:
                    f.result()
                except Exception as exc:
                    print(f"[thread-err] {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Call OpenAI Responses API with native function calling and local search.")
    _SE_TASKS = ['biology','earth_science','economics','psychology','robotics',
                 'stackoverflow','sustainable_living']
    parser.add_argument('--task', type=str, default=None, choices=_SE_TASKS,
                        help='Single task (legacy). Prefer --tasks.')
    parser.add_argument('--tasks', type=str, nargs='+', default=None, choices=_SE_TASKS,
                        help='Run over multiple tasks within ONE Python process — '
                             'model loaded once, searcher.set_task() used per task. '
                             'Default: all 7 SE tasks if neither --task nor --tasks given.')
    parser.add_argument('--use-agentic-sample', action='store_true',
                        help='Restrict each task to the fixed 25 qids per '
                             'agentic_sample_ids.json (seed=42).')
    parser.add_argument('--max-queries-per-task', type=int, default=None,
                        help='Cap queries per task (after --use-agentic-sample). '
                             'Useful for smoke tests — e.g. 3 means 3×N_tasks total.')
    parser.add_argument("--query", default=None, help="(legacy single-query mode; not used by Bright-Pro agentic eval)")
    parser.add_argument("--model", default="gpt-5-mini-08-07", help="Model name (default: %(default)s)")
    parser.add_argument("--max_tokens", type=int, default=30000, help="Max tokens to generate (default: %(default)s)")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Optional reasoning effort level (for reasoning-capable models)",
    )
    parser.add_argument("--output-dir", default="fixed_turn_runs/gpt-5-mini-08-07", help="Directory to store logs (default: %(default)s)")
    parser.add_argument("--query-template", choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT", "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION", "QUERY_TEMPLATE_ORACLE"], default="QUERY_TEMPLATE_NO_GET_DOCUMENT", help="Specify the query template to use (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for the model (default: use model defaults)")
    parser.add_argument("--top_p", type=float, default=None, help="Top P for the model (default: use model defaults)")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for dataset processing (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per query on transient errors (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of conversation rounds with function calls (default: %(default)s)",
    )
    parser.add_argument(
        "--finalize-at-rounds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="At these rounds, generate a final answer from retrieved snippets; stop after the highest round.",
    )
    parser.add_argument(
        "--final-answer-max-tokens",
        type=int,
        default=10000,
        help="Max tokens for each round's final answer (default: %(default)s)",
    )
    
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    
    # Server configuration arguments
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=2048,
        help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 512).",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=None,
        help="Directory to cache models and datasets. If not provided, will use environment variables or default.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed number of search results to return for all queries in this session (default: 5).",
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="If set, register both the search tool and the get_document tool.",
    )
    # Enable get_document by default; users can opt-out by not using the tool or changing template
    parser.set_defaults(get_document=False)
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models. If not provided, will use environment variables or CLI login.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Hugging Face home directory for caching models and datasets. If not provided, will use environment variables or default.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory to cache models and datasets. If not provided, will use environment variables or default.",
    )
    
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()
    
    if args.hf_token:
        print(f"[hf-token] set from CLI")
        os.environ['HF_TOKEN'] = args.hf_token
        os.environ['HUGGINGFACE_HUB_TOKEN'] = args.hf_token
    
    if args.hf_home:
        print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
        os.environ['HF_HOME'] = args.hf_home
    
    # Prefer OpenAI (public) when OPENAI_API_KEY is set, fall back to Azure.
    if os.getenv("OPENAI_API_KEY"):
        client = openai.OpenAI()
        print(f"[client] using OpenAI (public) with model={args.model}")
    elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        print(f"[client] using Azure OpenAI with model={args.model}")
    else:
        raise RuntimeError("Neither OPENAI_API_KEY nor (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT) set")

    # Determine task list
    if args.tasks:
        tasks_to_run = list(args.tasks)
    elif args.task:
        tasks_to_run = [args.task]
    else:
        tasks_to_run = _SE_TASKS
    print(f"[tasks] will run {len(tasks_to_run)}: {tasks_to_run}")

    # Load agentic sample ids (fixed 25 per task, seed=42) if requested
    sample_ids_map = {}
    if args.use_agentic_sample:
        _bpro_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sample_path = os.path.join(_bpro_root, 'agentic_sample_ids.json')
        if not os.path.isfile(sample_path):
            raise RuntimeError(f"agentic_sample_ids.json not found at {sample_path}; run "
                               f"agentic_retrieval/scripts_evaluation/sample_agentic_qids.py first")
        with open(sample_path) as f:
            sample_ids_map = json.load(f)['tasks']
        print(f"[sample] agentic 25-qid/task enabled, "
              f"{sum(len(v) for v in sample_ids_map.values())} total qids")

    # Initial task — set on first task; searcher.__init__ may use it
    args.task = tasks_to_run[0]
    searcher = searcher_class(args)    # MODEL LOADED ONCE HERE

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document
    )

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")
    print(f"Search agent ({searcher.search_type}) ready; tools={', '.join(tools_registered)} "
          f"snippet_max_tokens={args.snippet_max_tokens} k={args.k}")

    # Loop over tasks — model stays loaded; only doc embeddings + instructions
    # swap per task via searcher.set_task(). Each task's run_*.json flushes
    # incrementally, so Ctrl-C / SLURM timeout loses nothing.
    for task in tasks_to_run:
        print(f"\n========== TASK: {task} ==========")
        if getattr(searcher, 'task', None) != task:
            searcher.set_task(task)
        args.task = task
        args._sample_ids_for_task = sample_ids_map.get(task)
        _process_dataset(None, client, args, tool_handler)


if __name__ == "__main__":
    main()