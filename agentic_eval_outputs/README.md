# Bright-Pro Agentic Search — Raw Model Outputs

Raw agent traces for the agentic search evaluation results reported in the
Bright-Pro paper. Each setup pairs a **retriever** (used as an in-loop search
tool) with an **agent** (the LLM that issues queries and writes the final
answer).

## Layout

```
agentic_eval_outputs/
├── fix_round_gpt5mini/        ← paper Table 1 (main, fix-round.tex)
├── fix_round_qwen35/          ← appendix (fix-round-qwen35.tex)
├── adaptive_round_gpt5mini/   ← paper Table 2 (main, dynamic-round.tex)
└── adaptive_round_qwen35/     ← appendix
```

Each setup directory has 10 jsonl files, one per retriever:

```
<setup>/<retriever>.jsonl    # 175 lines = 7 tasks × 25 queries
```

## Setups

| Setup | Agent | Search loop | Top-k per round |
|---|---|---|---|
| `fix_round_gpt5mini`      | `gpt-5-mini`        | fixed 3 rounds, agent finalizes at rounds 1/2/3 | 5 |
| `fix_round_qwen35`        | `Qwen3.5-122B-A10B` | fixed 3 rounds, agent finalizes at rounds 1/2/3 | 5 |
| `adaptive_round_gpt5mini` | `gpt-5-mini`        | agent self-decides stopping (≤100 rounds)        | 5 |
| `adaptive_round_qwen35`   | `Qwen3.5-122B-A10B` | agent self-decides stopping (≤100 rounds)        | 5 |

## Retrievers (10) — paper names

`BM25`, `grit` (GritLM-7B), `inst-xl` (Instructor-XL), `gte-qwen2` (GTE-Qwen2-7B),
`qwen3-embed` (Qwen3-Embedding-8B), `reasonir` (ReasonIR-8B), `diver-retriever`
(DIVER-4B), `diver-1020` (DIVER-4B-1020), `bge-reasoner` (BGE-Reasoner-8B),
`RTriever-4B` (ours).

## Tasks (7)

`biology`, `earth_science`, `economics`, `psychology`, `robotics`,
`stackoverflow`, `sustainable_living`. Each task has 25 queries (the agentic
sample, see `agentic_sample_ids.json`, seed=42).

## Per-line schema

One JSON object per line. Lines are sorted by `(task, query_id)`.

```json
{
  "task": "biology",
  "metadata": {
    "model": "gpt-5-mini",
    "reasoning": {"effort": "medium", "summary": "detailed"},
    "output_dir": "<retriever>/<task>",
    "max_tokens": 30000
  },
  "query_id": "11",
  "usage": {"input_tokens": ..., "output_tokens": ..., "total_tokens": ...},
  "retrieved_round_count": 3,
  "retrieved_documents_id": ["biology-...", "..."],
  "status": "completed",
  "result": [
    {"type": "output_reasoning", "output": ["..."]},
    {"type": "tool_call", "tool_name": "search", "arguments": "...", "round": 1},
    {"type": "tool_result", "tool_name": "search", "output": "..."},
    ...
    {"type": "final_answer_round_1", "output": "..."},
    {"type": "final_answer_round_2", "output": "..."},
    {"type": "final_answer_round_3", "output": "..."}
  ]
}
```

For `adaptive_round_*`, `retrieved_round_count` varies per query (the agent
decides when to stop).

## Counts

Per setup: 10 retrievers × 7 tasks × 25 queries = **1,750** queries across
**10** jsonl files. Total across 4 setups: 7,000 query traces.
