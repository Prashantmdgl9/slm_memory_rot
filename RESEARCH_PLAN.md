# Research Plan: On-Device Memory Rot in Small-Model Agents

**Working title:** *The Small Librarian: Characterizing and Repairing Memory Rot in Strictly On-Device Small-Model Agents*

## Context (why we're doing this)

Agentic LLM systems degrade over long-horizon use: self-maintained memory accumulates
contradictions, stale facts, and duplicated/mis-consolidated entries — "memory rot."
The 2025→2026 literature (Mem0, Supersede/supersession, memory-drift, SSGM, RecMem,
MemTrace) has largely converged on this problem **but assumes a frontier-scale LLM is
the memory manager**. Nobody is systematically asking what happens when the "librarian"
is a **small model running strictly on-device** — which is exactly the regime that matters
for private, offline, zero-cost personal agents on a laptop or phone.

That gap is the thesis. The laptop-only / small-model constraint is not a
limitation to apologize for — it *is* the research premise. The deeper goal is to
genuinely understand how agent memory breaks and why, using this paper as the vehicle.

The paper in one line:
> *On-device small models can't be trusted as agent "librarians" — here is a diagnostic
> that shows exactly how their memory rots, why, and a strictly-local framework that
> repairs most of it.*

## Contributions (the full arc)

1. **An instrument** — an operation-isolating diagnostic that decomposes the memory
   pipeline (extract → decide write op → resolve conflict → retrieve) and measures
   *where* a small model breaks and *how the error compounds per operation*.
2. **A characterization ("the why")** — a fully on-device capacity sweep showing the rot
   *curve* and its *qualitative signature* (hypothesis: small models extract fine but
   rarely fire UPDATE/DELETE → monotonic accumulation → duplication + unresolved
   contradictions → QA collapse). The finding must be non-obvious (disproportionate /
   superlinear + operation-specific), not "small = worse."
3. **A framework ("the fix")** — one principle: *move compounding judgment out of the
   weak model into structure, retrieval, and deterministic rules; leave the small model
   only the sub-tasks it is reliably good at.* Ablated, measured, still 100% on-device.

## Hard constraints

- **Strict on-device for the system:** the librarian never calls a remote/frontier model — ever.
- **Strict on-device for evaluation too:** grading uses ground-truth exact-match/F1 where
  possible + the largest *local* model as LLM-judge for open-ended answers. No API in the
  grading loop.
- **Open-source only, zero paid-API spend.** No proprietary/paid model (Claude, OpenAI) is
  ever used, anywhere. Every model is open-weights.
- **The 32B+ reference tier is the only remote thing, and it is a measuring stick — not part
  of the system.** It runs via the **HuggingFace Inference API free tier** purely to place an
  upper-bound point on the capacity curve, computed once and cached. Never inside the deployed
  pipeline, never the judge. Every figure labels it as remote.
- **Hardware:** Apple Silicon, 16GB+ (via MLX and/or Ollama). On-device test models are
  **8–15B** (4-bit); 32B+ cannot fit locally and is remote reference only.

## Experimental design (the scientific core)

**Single independent variable = the manager model.** Everything else is held constant so
any rot difference is attributable to the model driving extraction + write-decisions:
- same local embedder (bge-small-en-v1.5 or all-MiniLM-L6-v2) across ALL conditions,
- same retriever, same vector store, same memory schema, same prompts,
- same data and same controlled-injection stream.

Only the model performing extraction and the ADD/UPDATE/DELETE/NOOP write-decisions changes.

The sweep has two axes: (1) the **within-family capacity ladder** (isolates size cleanly, spans
on-device 8–15B rungs + the remote 32B+ reference), and (2) a **cross-family check** at ~8B and
~14B (confirms the phenomenon generalizes beyond one family).

**Attribution by isolation** (this is what earns "how it works and the reasons"):
- *Extraction:* precision/recall of extracted atomic facts vs gold facts.
- *Write-decision:* given (existing store, new fact) where the correct op is known from the
  injection layer, measure per-op classification accuracy — the cleanest isolation.
- *Retrieval:* P@k / recall@k with the held-constant embedder (differences here come from
  *what is in the store*, i.e. downstream of write-decisions).
- *Compounding:* track store-health (contradiction count, duplication factor, staleness)
  vs #operations per model; fit slope; test for superlinearity.
- *Downstream:* per wrong QA answer, classify the root cause (never-extracted / not-retrieved
  / stale / duplicated-or-contradicted) → an attribution breakdown.

## Datasets

- **Primary: LongMemEval** (`LongMemEval_S`, ~115k-token histories) — has *knowledge-update*
  and *temporal-reasoning* question types, which are literally ground-truth supersession/rot
  events. Runs on the laptop.
- **Secondary: LoCoMo** — the benchmark Mem0 used; gives numbers that situate against that
  literature.
- **Controlled-injection layer** built on top — synthetic update/contradiction streams where
  the ground-truth supersession events and correct write-ops are known exactly, enabling clean
  causal attribution that natural datasets alone can't give.

## Models — all open-source, zero paid-API spend

Model choices are locked. All on-device models run via Ollama (4-bit quantization) on a 16GB
Apple Silicon Mac. The judge/reference runs remotely via HF Inference API free tier — it is
never part of the system, only used for grading and as a measuring-stick upper bound.

- **Primary capacity ladder — Qwen3 (same family, clean size variable):**
  - Qwen3-4B (on-device, ~2.5GB)
  - Qwen3-8B (on-device, ~5GB)
- **Cross-family robustness — Llama (confirms phenomenon is not Qwen-specific):**
  - Llama 3.2-3B (on-device, ~2GB)
  - Llama 3.1-8B (on-device, ~5GB)
- **Remote reference + judge:** Qwen3-32B (or largest available 32B+ open-weight) via HF
  Inference API free tier. Computed once and cached for the reference curve; used as LLM-judge
  for open-ended answer grading. Never inside the deployed pipeline. Always labeled "remote" on
  plots. No paid proprietary model (Claude/OpenAI) is ever used.
- **Embedder (constant across ALL conditions):** bge-small-en-v1.5 (sentence-transformers).
- Reliable structured output for op-classification via JSON-schema / grammar-constrained
  decoding (Ollama `format=json`, or outlines / llama.cpp GBNF).

Total local disk footprint: ~15GB across all four on-device models.

## Baselines (Mem0-style comparison set), run across the capacity sweep

1. **No-memory** (floor).
2. **Full-context** (where it fits; truncation behavior is itself a data point).
3. **RAG-only** (raw-chunk retrieval, no consolidation).
4. **Naive Mem0-style manager** (extract/store/update/retrieve) — the primary rot exhibit.
5. **Ours** — the structure-over-judgment framework.

## The framework (Act III) + ablations

Principle: constrain the small model to competent sub-tasks (short extraction, binary
entailment); offload compounding judgment to structure/rules/retrieval. Mechanisms, each
targeting a rot source and each independently ablatable:
- **Typed/schematized memory** (entity–attribute slots) → UPDATE becomes a deterministic key
  match → attacks "never fires UPDATE" + staleness.
- **Retrieval-gated writes** (retrieve near-duplicates before ADD; decide only among candidates)
  → attacks duplication blow-up.
- **Rule-based bi-temporal supersession** (timestamps + provenance + newest-valid-wins)
  → attacks stale-fact-wins contradictions.
- **Decomposed write-decision** (a chain of binary entailment checks instead of one 4-way call)
  → directly attacks write-decision accuracy.
- **Periodic cheap compaction** (batched dedup/merge over structured slots) → repairs
  accumulated rot.

Report: gap-closure toward the larger local model, per-mechanism ablation, and on-device
latency/cost per turn (to prove "cheap").

## Repository structure (all new; directory is currently empty)

```
memory/
  README.md
  pyproject.toml / requirements.txt
  configs/                 # model + experiment configs (yaml)
  src/
    models/                # local wrappers (Ollama/MLX) + HF remote-reference client + embedder; constrained decoding
    memory/
      store.py             # vector store + schema, held constant across conditions
      managers/            # baseline_naive.py, rag_only.py, full_context.py, ours_framework.py
    diagnostic/
      injection.py         # controlled update/contradiction stream + ground-truth op labels
      metrics.py           # op-accuracy, duplication, staleness, contradiction, slope, attribution
    eval/
      judge_local.py       # ground-truth match + local LLM-judge
      runners.py           # LongMemEval / LoCoMo harness
  data/                    # dataset loaders (LongMemEval, LoCoMo)
  experiments/             # scripts per phase; JSONL result logs
  analysis/                # pandas + matplotlib -> figures/tables
  paper/                   # draft, figures, reproducibility notes
```

## Phased execution (each phase is independently reviewable)

- **Phase 0 — Spike / de-risk:** environment; one small + one large model live via Ollama/MLX;
  load `LongMemEval_S`; run a trivial end-to-end (ingest a few sessions, answer one question).
  Goal: prove the pipeline runs on the laptop before building anything.
- **Phase 1 — Harness & baselines:** minimal instrumented memory manager with pluggable model;
  baselines 1–4; first QA numbers on `LongMemEval_S` with a small vs a large model; lock the
  constant embedder/retriever/store.
- **Phase 2 — Instrument + injection (Act I):** controlled injection with ground-truth op labels;
  implement all diagnostic metrics.
- **Phase 3 — Characterization (Act II):** run the capacity sweep; produce rot curves,
  operation breakdown, attribution, superlinearity test; **identify the dominant failure mode.**
  (Pivotal: the fix in Phase 4 is shaped by what this finds.)
- **Phase 4 — Framework (Act III):** implement mechanisms targeting the dominant failure; ablate
  each; measure gap-closure + on-device latency/cost.
- **Phase 5 — Write-up:** figures/tables, paper draft, reproducibility package.

## Verification (how we know each step works)

- **Phase 0:** end-to-end smoke test answers ≥1 LongMemEval question correctly with a local model.
- **Pipeline sanity / controls:** with the injection layer, a *large* local model must show
  measurably *less* rot than a *small* one under identical everything-else (if not, a confound
  exists — stop and fix). The framework must not significantly *hurt* the large model.
- **Instrument validity:** on synthetic injections with known correct ops, metrics recover the
  planted contradictions/updates; op-accuracy is 100% on trivially-easy planted cases.
- **Judge validity:** local judge agreement with exact-match on the gold-answer subset is reported
  (e.g. Cohen's κ); grading isn't trusted until agreement is acceptable.
- **Framework claim:** ablations show gap-closure is attributable to specific mechanisms and
  reproducible across ≥3 seeds / prompt paraphrases; latency/cost stays within an on-device budget.
- **Reproducibility:** every figure regenerable from logged JSONL via `analysis/`.

## Key risks & mitigations

- **Triviality ("small is obviously worse"):** the diagnostic is designed to surface a
  *qualitative, operation-specific, compounding* signature — that is the actual finding.
- **Small-model prompt fragility:** grammar/JSON-constrained decoding for op-classification;
  fixed prompts; report variance across seeds and paraphrases.
- **Compute:** `LongMemEval_S` (~115k tokens) fits 16GB; large models Q4-quantized; cache
  embeddings; run offline in batches.
- **Remote reference availability:** HF free serverless is rate-limited and the largest models
  (70B) aren't always served free — 32B is the reliable reference; 70B is optional and fetched
  in small batched bursts. Cache every remote response so the reference is computed once and
  never re-fetched; if a model is unavailable, fall back to the next-smaller open 32B+ checkpoint.
- **Confounds:** embedder/retriever/store/prompts/data all held constant; only the manager varies.

## Deliverables

1. Reproducible on-device experimental harness (the artifact).
2. The rot diagnostic + controlled-injection benchmark layer.
3. Characterization results (the "why") and the mitigation framework (the fix) with ablations.
4. Paper draft situating the work against the 2026 memory literature.

## Key references (2026 landscape)

- Mem0 (2025) — scalable long-term memory, LOCOMO: https://arxiv.org/abs/2504.19413
- Supersede — diagnosing/training the memory-update gap: https://arxiv.org/html/2606.27472v1
- Memory for Autonomous LLM Agents (survey): https://arxiv.org/html/2603.07670v1
- SSGM — governing evolving memory: https://arxiv.org/html/2603.11768v1
- RecMem — recurrence-based consolidation: https://arxiv.org/pdf/2605.16045
- MemTrace — error attribution in memory systems: https://arxiv.org/pdf/2605.28732
- LongMemEval (ICLR 2025): https://github.com/xiaowu0162/longmemeval
- LongMemEval-V2 (agentic): https://arxiv.org/html/2605.12493v1
