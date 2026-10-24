# The Small Librarian: Characterizing and Repairing Memory Rot in Strictly On-Device Small-Model Agents

Code and data for the paper. Everything runs **strictly on-device** with
open-weight models via [Ollama](https://ollama.com) — no paid APIs.

## What's here

| Path | Contents |
|------|----------|
| `src/memory/` | Memory store + managers (`naive`, `rag_only`, `framework` = typed slots + deletion signals, `framework_bitemporal` = non-destructive deletion, `framework_typed_only`, `framework_compaction`) |
| `src/diagnostic/` | Controlled 80-turn injection stream with ground-truth operation labels; per-operation + rot-score metrics |
| `src/eval/` | EM / token-F1 scoring and runners |
| `src/models/` | Ollama client + embedder (`bge-small-en-v1.5`) |
| `experiments/` | `phase1_run.py` (LongMemEval conditions), `ablation_run.py` (4-condition ablation), `longmemeval.py`, `locomo.py` (second benchmark), `spike_adjudicator.py` (semantic-routing probe) |
| `data/` | `longmemeval_s.json`, `locomo10.json` |
| `tests/` | Unit tests for the bitemporal store/manager |
| `compute_cis.py` | Wilson confidence intervals for the tables |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy, requests, sentence-transformers
ollama serve &
# pull the five on-device models used in the paper:
ollama pull qwen3:4b && ollama pull qwen3:8b && ollama pull llama3.2:3b \
  && ollama pull llama3.1:8b && ollama pull qwen2.5:7b
# the retrieval embedder (bge-small-en-v1.5) is downloaded automatically by
# sentence-transformers on first run — no Ollama pull needed.
```

## Reproducing the results

```bash
# Phase 1 — LongMemEval-S QA across conditions (RAG vs managed store)
python experiments/phase1_run.py rag   qwen3:4b 20
python experiments/longmemeval.py --condition naive --model qwen3:4b --n 20

# Phase 2 / ablation — per-operation recall + rot on the injection stream
python experiments/ablation_run.py qwen3:4b --condition=all

# Second benchmark — LoCoMo
python experiments/locomo.py --condition rag       --model llama3.2:3b --convs 10
python experiments/locomo.py --condition framework --model llama3.2:3b --convs 2

# Confidence intervals
python compute_cis.py
```

## Tests

```bash
python -m unittest discover -s tests
```

## Note on the embedder

Retrieval uses `BAAI/bge-small-en-v1.5` (`src/models/embedder.py`), consistent
across every condition and benchmark reported in the paper.

## Citation

> Mudgal, P. *The Small Librarian: Characterizing and Repairing Memory Rot in
> Strictly On-Device Small-Model Agents.* 2026.
