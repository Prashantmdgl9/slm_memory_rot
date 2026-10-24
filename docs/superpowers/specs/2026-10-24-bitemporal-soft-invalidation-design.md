# Design: Bitemporal Soft-Invalidation ("recoverable deletes")

Date: 2026-10-24
Status: approved — building TDD-first
Target: The Small Librarian paper — Candidate B (after A and C were empirically ruled out)

## Why (and why not A/C)

Candidates A (threshold slot-binding) and C (in-band LLM adjudicator) were both
disproven by spikes before any production code:
- A: no global cosine τ separates true synonyms (`employer/company`=0.71) from
  distinct head-noun-sharing attributes (`home_city/work_city`=0.81).
- C: on-device adjudication of "same attribute?" tops out at 60% (qwen2.5:7b);
  the 8B model is at chance and false-merges home/work city.

Both negatives **reinforce the paper's thesis** (SLMs can't make the attribute-
identity judgment in any framing) and will be reported (see Task 2).

B needs **zero SLM judgment and zero embedding geometry** — pure deterministic
bookkeeping — so none of those failure modes apply.

## The problem B fixes

Paper §5.3: deletion signals erased *correct* facts (Qwen3-8B lost city and car),
dropping QA 64%→50%. A hard `DELETE` is permanent data loss, so the delete
precision/recall tradeoff is a wash. B makes deletion **non-destructive and
recoverable**, so deletion recall can be pushed aggressively without the erasure
penalty.

## Goals / non-goals

- Goal: wrongly-deleted facts remain **recoverable**; deletion recall can rise
  without permanent data loss; the §5.3 QA drop is recovered.
- Goal: existing conditions (naive, +Slots, +Del) stay **byte-identical** — B is
  additive, a new condition `+SoftDel`.
- Non-goal: no change to UPDATE routing (that was A/C).
- Anti-goal: indiscriminately resurrecting *correctly*-deleted facts (precision
  cost) — the recovery path must be a ranked fallback, not a flat merge.

## Mechanism

Two new classes; nothing existing is modified.

### `BitemporalStore(MemoryStore)` — `src/memory/store.py` (or new file)

Each entry gains `valid: bool` and `invalidated_at: str`.

- `add(...)`: as parent, plus `valid=True, invalidated_at=None`.
- `delete(eid)`: **soft** — set `valid=False`, `invalidated_at=date`; append a
  `DELETE` tombstone to `op_log`; **do not remove the entry.**
- `__len__`, `all_entries()`: return **valid-only** — so the existing rot metric
  (`store_health_snapshot`) still sees a soft-deleted fact as correctly gone (no
  false "undeleted" rot).
- `search(query, k, include_invalid=False)`: valid entries by similarity; when
  `include_invalid`, append invalid entries **ranked strictly after** all valid
  hits (a lower tier, never interleaved).
- `recoverable()`: the invalidated entries (for the data-loss metric).

### `FrameworkBitemporalManager(FrameworkManager)` — new manager

- `__init__`: `self.store = BitemporalStore()`; everything else inherited.
- `_apply_fact`: **inherited unchanged** — parent's deletion branch already calls
  `self.store.delete(eid)`, which is now soft. (No override needed.)
- `query`: override to retrieve via a tiered policy. Factor retrieval into a pure,
  testable `_retrieve(question)`:
  1. `valid = store.search(q, k=15)`.
  2. If the top valid score is weak (`< FALLBACK_TAU`, default 0.5) or fewer than
     a few hits, pull `recovered = store.search(q, k=5, include_invalid=True)`
     invalidated candidates.
  3. Build memory text: valid facts first; recovered facts in a clearly-labeled
     "(possibly outdated)" section so a false-delete can still answer a query the
     valid store can't.

## Evaluation plan

New ablation condition **`+SoftDel`** (parallels `+Del`):

| Metric | Expectation |
|--------|-------------|
| DELETE recall / precision | recall can rise; precision as `+Del` |
| **Data-loss rate** (new): correct facts irrecoverably erased | → **0** by construction |
| QA EM on injection stream | recovers the §5.3 drop (Qwen3-8B city/car retrievable) |
| QA EM on LongMemEval-S + LoCoMo | ≥ `+Del` |
| Rot score | ≤ `+Del` (soft-deleted still counts as gone in effective store) |

Honest knob to report: `FALLBACK_TAU` trades recovery (helps false deletes)
against resurrection of correct deletes (precision). Sweep it.

## TDD increments

1. `BitemporalStore.delete` is non-destructive + recoverable; effective store
   (`len`, `all_entries`, default `search`) excludes it. — failing test first.
2. `search(include_invalid=True)` tiers valid strictly before invalid.
3. `FrameworkBitemporalManager._retrieve` recovers a soft-deleted fact only when
   no valid entry answers (fallback policy).

Then: wire `--condition softdel` into `ablation_run.py` / `longmemeval.py` /
`locomo.py`, run, regenerate affected tables (Task 4).

## Paper note (Task 2)

Add a short subsection: "What didn't work — semantic binding." Report A's
threshold-inseparability and C's 50–60% adjudication ceiling as evidence that
SLMs cannot perform attribute-identity judgment, motivating the judgment-free
design of B.
