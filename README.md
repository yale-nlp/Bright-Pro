# Bright-Pro

Evaluation code for the **Bright-Pro** benchmark (ACL 2026): an expert-annotated
extension of [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) for
reasoning-intensive retrieval, with both static and agentic-search evaluation
protocols.

- 📊 **Dataset**: [`yale-nlp/Bright-Pro`](https://huggingface.co/datasets/yale-nlp/Bright-Pro)
- 🤖 **Retriever**: [`yale-nlp/RTriever-4B`](https://huggingface.co/yale-nlp/RTriever-4B)
- 📁 **Collection**: [Bright-Pro on HuggingFace](https://huggingface.co/collections/yale-nlp/bright-pro-69f2bfe0ccf4baa9bfc50142)

## What's in this repo

```
.
├── bright_pro_data.py         # tiny loader; defaults to load_dataset("yale-nlp/Bright-Pro")
├── retrieval/                 # static retrieval (α-nDCG@k, A-Recall@k, NDCG@k, Recall@k)
│   ├── run.py                 # main entry — embed corpus, retrieve, score
│   ├── retrievers.py          # all retrievers reported in the paper
│   ├── metrics.py             # α-nDCG / A-Recall / NDCG / Recall implementations
│   ├── qwen3_embedding.py     # sentence-transformers wrapper for the Qwen3-Embedding family
│   └── configs/<retriever>/<task>.json   # per-(retriever, task) instructions
├── agentic_retrieval/         # agentic search loop + LLM-as-Judge
│   ├── search_agent/          # OpenAI / Qwen agents, fixed-round and adaptive-round
│   ├── searcher/              # in-loop search tool (one class per retriever family)
│   ├── scripts_evaluation/    # judge.py, α-nDCG aggregator, AER, ...
│   └── configs/<retriever>/<task>.json
└── agentic_sample_ids.json    # fixed 175-query subset (25 / task, seed=42) for agentic eval
```

## Retrievers reported in the paper

| Category | Retriever | HF id |
|---|---|---|
| Lexical | BM25 | — |
| General-purpose | GritLM-7B | `GritLM/GritLM-7B` |
| | Instructor-XL (1.5B) | `hkunlp/instructor-xl` |
| | GTE-Qwen2-7B-Instruct | `Alibaba-NLP/gte-Qwen2-7B-instruct` |
| | Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` |
| | EmbeddingGemma-300M (static-only) | `google/embeddinggemma-300m` |
| | OpenAI text-embedding-3-Large (static-only) | OpenAI API |
| Reasoning-intensive | ReasonIR-8B | `reasonir/ReasonIR-8B` |
| | DIVER-Retriever-4B | `AQ-MedAI/Diver-Retriever-4B` |
| | DIVER-Retriever-4B-1020 | `AQ-MedAI/Diver-Retriever-4B-1020` |
| | BGE-Reasoner-Embed-Qwen3-8B | `BAAI/bge-reasoner-embed-qwen3-8b-0923` |
| | INF-Retriever-v1-Pro (static-only) | `infly/inf-retriever-v1-pro` |
| | **RTriever-4B** (ours) | `yale-nlp/RTriever-4B` |

The three retrievers tagged "static-only" are evaluated only in the static
setting (paper Table 1); the agentic protocol (Tables 2, 3) uses the
remaining 10.

## Agents and judge

| Role | Model |
|---|---|
| Search agent (primary) | `gpt-5-mini-08-07` |
| Search agent (alternative) | `Qwen3.5-122B-A10B` |
| LLM-as-Judge | `gpt-5` |

## Setup

```bash
conda env create -f environment.yml
conda activate brightpro-eval

cp .env.example .env       # fill OPENAI_API_KEY (judge + GPT-5-mini agent),
                           # HF_TOKEN (gated EmbeddingGemma), and optionally
                           # AZURE_OPENAI_* if going via Azure
chmod 600 .env
```

The data loader defaults to streaming `yale-nlp/Bright-Pro` from HuggingFace.
For offline runs, point `BRIGHT_PRO_DATA_ROOT` at a local mirror of the dataset
(layout: `<root>/{examples,documents,aspects}/<task>.json`).

## Static retrieval

```bash
cd retrieval
python run.py --task biology --model rtriever-4b
# outputs/biology_rtriever-4b/score.json     <- {qid: {doc_id: score}}
# outputs/biology_rtriever-4b/results.json   <- standard NDCG/MAP/Recall/P/MRR at k ∈ {1,5,10,25,50,100}
```

`--model` can be any of the 13 retrievers above (see `run.py --help` for the
exact CLI names). Standard metrics (NDCG@k, Recall@k, MAP@k, MRR) are written
by `run.py`; aspect-aware metrics (α-nDCG@k and weighted A-Recall@k) are
computed post-hoc:

```bash
python evaluation/alpha-ndcg-evaluation.py \
    --task biology --score_file outputs/biology_rtriever-4b/score.json \
    --k 25 --alpha 0.5
```

## Agentic retrieval

The agentic protocol is run in two flavours over a fixed 175-query subset
(`agentic_sample_ids.json` — 25 queries / task, seed=42):

**Fixed-round** (R ∈ {1,2,3}, top-5 per round):
```bash
cd agentic_retrieval
python search_agent/openai_fixed_turn.py \
    --searcher-type rtriever-4b \
    --tasks biology earth_science economics psychology robotics stackoverflow sustainable_living \
    --use-agentic-sample
```

**Adaptive-round** (agent decides when to stop):
```bash
python search_agent/openai_new.py \
    --searcher-type rtriever-4b \
    --tasks biology earth_science economics psychology robotics stackoverflow sustainable_living \
    --use-agentic-sample
```

Switch the OpenAI agent for the Qwen3.5 agent by replacing
`openai_*.py` with `qwen_new.py` and supplying a vLLM endpoint via
`OPENAI_BASE_URL`.

## LLM-as-Judge

```bash
cd agentic_retrieval
python scripts_evaluation/judge.py \
    --runs-dir runs/gpt-5-mini-08-07 \
    --save-jsonl runs/gpt-5-mini-08-07/_judgements.jsonl \
    --deployment gpt-5
```

The judge prompt is the one used in the paper (the verbatim text is the
`JUDGE_INSTRUCTION` constant at the top of `judge.py`). For each system
answer the judge scores every reasoning aspect on `{0, 0.5, 1}` and assigns
an overall answer score on a 1–5 Likert. Aspect coverage is aggregated using
the paper's annotated aspect weights.

## License

Released under the [MIT License](LICENSE).
