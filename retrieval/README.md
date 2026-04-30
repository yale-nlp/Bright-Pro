# Static retrieval

This subdir runs the **static retrieval** evaluation: each query is embedded
once, all corpus documents are embedded once, and the retriever ranks the
corpus by cosine similarity. Scores are written to
`outputs/{task}_{model}/{results,score}.json`.

## Setup

See the top-level [`README.md`](../README.md) and [`environment.yml`](../environment.yml)
for the canonical `brightpro-eval` conda env. Two pins matter:

| Pin | Why |
|---|---|
| `torch>=2.7` (CUDA 12.x) | Required for Blackwell-class GPUs (SM 12.0). |
| `transformers==4.55.2` | Several retrievers (GritLM, GTE-Qwen2, ReasonIR, INF-Retriever-Pro) load via `trust_remote_code=True` and the bundled modeling files still use 4.x APIs (`DynamicCache.from/to_legacy_cache`, `is_causal` toggle, direct `rope_theta`). Under 5.x they silently produce near-random embeddings. |

API keys go in `.env` at the repo root (see `.env.example`):

- **OpenAI text-embedding-3-Large** (`--model openai`): set `OPENAI_API_KEY`.
  For bulk indexing prefer the Batch API (50% cheaper, 24 h SLA) — see
  `run_openai_batch.py`.
- **EmbeddingGemma-300M**: set `HF_TOKEN` after accepting the license at
  <https://huggingface.co/google/embeddinggemma-300m>.
- **BM25**: needs a Java runtime (Java 11+, via `pyserini` → `pyjnius`).
  Set `JAVA_HOME` and ensure `java`/`javac` are on `$PATH` before running BM25.

## Running

```bash
cd retrieval
python run.py --task biology --model rtriever-4b
```

`--task` is one of: `biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living`.

`--model` is one of:

```
bm25            inst-xl       gte-qwen2     qwen3-embed   openai
grit            reasonir      diver-retriever               diver-1020
bge-reasoner    inf-retriever-pro             embeddinggemma
rtriever-4b
```

Common flags:

- `--output_dir` (default `outputs`)
- `--cache_dir` (default `cache`) — caches doc embeddings on disk so subsequent
  `--model` runs over the same task are fast.
- `--encode_batch_size`, `--query_max_length`, `--doc_max_length`
- `--model_cache_folder` — local HF cache directory for model weights.

## Outputs

After a `run.py` invocation, `outputs/<task>_<model>/` contains:

| File | Contents |
|---|---|
| `score.json` | per-query map `{qid: {doc_id: score}}`, top-200 docs/query |
| `results.json` | standard BRIGHT metrics: NDCG@k, MAP@k, Recall@k, P@k for `k ∈ {1,5,10,25,50,100}`, plus MRR |

Aspect-aware metrics (α-nDCG@k and weighted A-Recall@k) are computed post-hoc
from `score.json`:

```bash
python evaluation/alpha-ndcg-evaluation.py \
    --task biology \
    --score_file outputs/biology_rtriever-4b/score.json \
    --k 25 --alpha 0.5

python evaluation/weighted_aspect_recall.py \
    --task biology \
    --score_file outputs/biology_rtriever-4b/score.json \
    --k 25
```
