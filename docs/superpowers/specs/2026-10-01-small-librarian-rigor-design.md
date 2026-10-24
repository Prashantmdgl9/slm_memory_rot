# Small Librarian — Rigor & Novelty Revision (Design Spec)

**Date:** 2026-10-01
**Author:** Prashant Mudgal (with Claude Code)
**Context:** `paper/main.tex` ("The Small Librarian") was rejected by arXiv cs.CL
moderation as "not sufficient original or substantive scholarly research." The
author already has an *accepted* cs.CL paper (arXiv:2609.20824, "Do small
language models know what they don't know?"), so the rejection is about this
paper's substance, not author signals. This spec redesigns the paper to raise
both rigor and novelty.

**Hard rule:** every reported number comes from a real run on this machine. No
fabricated data, figures, or observations — ever. If a stage cannot run or
underdelivers, we report that and adjust scope.

## Constraints (unchanged from original paper)

- Strictly on-device; open-weight models via Ollama; **zero paid API** anywhere
  in the system or evaluation.
- Hardware: 16 GB Apple Silicon laptop, ~15 GB free disk (prune logs as we go).
- Models already pulled: `qwen3:4b`, `qwen3:8b`, `llama3.2:3b`, `llama3.1:8b`,
  `qwen2.5:7b`; embedder `mxbai-embed-large`.

## Contribution spine (approved)

- **C1 — Finding (headline):** memory rot is a *leading indicator*.
  Operation-level recall degrades measurably *before* downstream QA collapses;
  we quantify the lead time. Novel, testable.
- **C2 — Mechanism (why):** routing, not comprehension. Force-feeding the gold
  prior fact into the routing prompt recovers UPDATE recall ⇒ the bottleneck is
  store-state tracking, not reasoning. Makes the trivial deterministic fix
  *evidence for C2* rather than a standalone contribution.
- **C3 — Method (novel repair):** an **entropy-gated write router** built on the
  author's semantic-entropy result (arXiv:2609.20824): sample the
  ADD/UPDATE/DELETE decision, measure semantic-entropy agreement, trust the
  model when confident, fall back to deterministic slot logic when not.
  Generalizes past the hardcoded keyword list; ties the two papers into one line.
  Deterministic typed-slots demoted to an ablation baseline.

**Evidence structure (two layers):**
- *Controlled labeled stream* (per-turn op ground truth) → carries C1 + C2.
- *Recognized benchmarks* — LongMemEval-S (scaled n) + LoCoMo → ecological
  generalization + C3 method eval.

## Experiments

| ID | Experiment | Reuse | New | Compute |
|----|-----------|-------|-----|---------|
| E0 | Fix qwen3 thinking-mode confound; re-baseline | runners | ollama options | low |
| C2 | Mechanism probe: blind vs prior-fact-injected UPDATE routing | ops logging | probe harness | low, **decisive** |
| C1 | ~200-turn stream + QA probes every ~20 turns; lead-time cross-correlation | rot/ops metrics | long-stream gen, lead-time analysis | moderate |
| C3 | Entropy-gated router; ablation ladder Naive→+slots→+entropy→+entropy+slots | store, extract | router, N-sample sampler | **heaviest** (N× inference) |
| B1 | LongMemEval-S n=20→50 + update/temporal subset breakout | runner | subset tagging | moderate |
| B2 | LoCoMo loader + QA runner | QA harness | loader | moderate |

**Run order (cheapest + most decisive first):** E0 → C2 → C1 → C3 → B1/B2 →
re-plot/re-tabulate → rewrite paper. C2 is a genuine gate: if force-feeding does
not recover routing, C2 is false — stop and revise the story before C3.

## Paper restructuring (no runs)

- Foreground recognized-benchmark evidence; reframe synthetic stream as the
  controlled *diagnostic instrument*, not the headline.
- Cut the auto-generated-looking "Practical Implications" listicle; thin run-in
  headers; add public code-repo link.
- Reframe claims to match evidence; state power limits up front.

## Success criteria

- C2 probe yields a clear, reportable verdict (recovers or not).
- C1 produces a measured lead time with CIs across models.
- C3 entropy-gated router beats naive on op-recall and is competitive with /
  better than deterministic-slots without hardcoded rules — or we report where
  it doesn't.
- Two recognized benchmarks reported at n≥50 with tightened CIs.
- Every figure regenerable from logged JSONL.
