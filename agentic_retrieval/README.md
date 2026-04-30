# Agentic retrieval

This subdir runs the **agentic** evaluation: the retriever is plugged into an
LLM agent that iteratively issues search queries, reads the top-k results, and
produces a citation-grounded final answer. We then judge the final answer with
an LLM-as-Judge against a reference answer constructed from the human-annotated
reasoning aspects.

The agentic eval uses a fixed sample of 175 queries (25 per task, seed=42)
stored in [`../agentic_sample_ids.json`](../agentic_sample_ids.json).

## Setup

The `brightpro-eval` conda env from the top-level [`environment.yml`](../environment.yml)
covers everything. Required env vars (in `.env` at repo root):

- `OPENAI_API_KEY` — for both the GPT-5-mini agent and the GPT-5 judge
  (alternatively set `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` +
  `AZURE_OPENAI_API_VERSION`)
- `HF_TOKEN` — required by some retriever model repos (set if you hit a 401)

## Document-embedding cache

Both the static-eval `run.py` and the in-loop `Qwen3FamilySearcher` look for
cached doc embeddings at:

```
$CACHE_DIR/doc_emb/<retriever>/<task>/long_False/0.npy
$CACHE_DIR/doc_emb/<retriever>/<task>/long_False_<bs>.npy   # gte-qwen2 layout
$CACHE_DIR/doc_emb/<retriever>/<task>/0.npy                 # fallback
```

`$CACHE_DIR` defaults to `./cache` and can be overridden with `--cache-dir`
or `BRIGHT_PRO_CACHE_DIR`. If the cache is missing, the searcher re-encodes
the full corpus (50–124 k docs) on first use — for an 8B Qwen3 model on a
single H200 this is ~1–2 h per task. Recommended workflow:

| Step | Command | Output |
|---|---|---|
| 1 | `cd retrieval && python run.py --task <t> --model <m>` per (retriever × task) | populates `$CACHE_DIR/doc_emb/...` and writes static-retrieval scores |
| 2 | `python search_agent/openai_fixed_turn.py --searcher-type <m> --tasks ... --use-agentic-sample` | fixed-round agentic runs |
| 3 | `python search_agent/openai_new.py --searcher-type <m> --tasks ... --use-agentic-sample` | adaptive-round agentic runs |

To force re-encoding (smoke-test the encoder path), pass
`--qwen3-ignore-cache` to any Qwen3-family searcher.

## Running the agent

### Fixed-round (R ∈ {1, 2, 3}, top-5 per round)

```bash
python search_agent/openai_fixed_turn.py \
  --searcher-type rtriever-4b \
  --tasks biology earth_science economics psychology robotics stackoverflow sustainable_living \
  --use-agentic-sample
```

### Adaptive-round (agent decides when to stop)

```bash
python search_agent/openai_new.py \
  --searcher-type rtriever-4b \
  --tasks biology earth_science economics psychology robotics stackoverflow sustainable_living \
  --use-agentic-sample
```

`--searcher-type` is one of:
`bm25, reasonir, diver-retriever, diver-1020, bge-reasoner, qwen3-embed, inst-xl, gte-qwen2, grit, rtriever-4b`.

`--task` (or `--tasks`) is one of the seven StackExchange domains
(see top-level README).

The default agent model is `gpt-5-mini-08-07`. To swap in
`Qwen3.5-122B-A10B`, use `qwen_new.py` with an OpenAI-compatible vLLM
endpoint via `OPENAI_BASE_URL`.

### Re-judging completed runs

```bash
python scripts_evaluation/judge.py \
  --runs-dir runs/gpt-5-mini-08-07 \
  --save-jsonl runs/gpt-5-mini-08-07/_judgements.jsonl \
  --deployment gpt-5
```

The judge appends one JSON line per (task, qid, round, rep) to the file
given by `--save-jsonl` and prints aggregate cost. Use `JUDGE_DEPLOYMENT=gpt-5`
(default) to reproduce the paper.

## Computing α-nDCG over fixed-round runs

```bash
python scripts_evaluation/alpha_ndcg.py \
  --fixed_dir fixed_turn_runs \
  --alpha 0.5
```
