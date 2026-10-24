# Design: Semantic Slot-Binding ("soft typed slots")

Date: 2026-10-24
Status: awaiting approval (nothing is built yet)
Target: The Small Librarian paper — new headline anti-rot mechanism (Task 3)

## Intent and why

Typed slots (current framework) route writes by an **exact, string-normalized
key**: `key in self._slots` (`src/memory/managers/framework.py:87`). This misses
**implicit UPDATE** — supersession where the attribute is the same but the
extracted key phrasing differs ("employer" vs "company" vs "workplace" vs
"where I work"). Each phrasing opens a *separate* slot, so the store accumulates
contradictory values for one real-world attribute. This is:

- the single biggest residual gap in the framework (typed slots only help when
  key extraction is already clean — which the synthetic 80-turn stream is, but
  real benchmarks are not), and
- the reason the paper needed a **Compaction** section — which then does nothing
  in every experiment (`+Co` == `+De` in Table 4).

Semantic slot-binding makes the slot keyspace a **vector space**: supersession
becomes a *collision in embedding space*, decided by arithmetic (cosine), never
by the small model.

## Goals

- Raise UPDATE recall on **real, varied-phrasing** data (LongMemEval-S,
  LoCoMo) above exact-match typed slots, without any model fine-tuning.
- **Subsume compaction**: synonym fragmentation cannot accumulate because
  binding happens at write-time, not in a later LLM pass.
- Stay strictly on-device, training-free, open-weights (locked constraints).

## Non-goals / anti-goals

- Not DELETE handling (that is Candidate B, out of scope here).
- Not a learned/fine-tuned router (that is what Supersede does; our whole point
  is training-free). If we end up needing fine-tuning, the mechanism has failed.
- Anti-goal: over-merging distinct attributes (e.g. `home_city` vs `work_city`)
  — a false merge destroys a correct fact. Precision of binding matters as much
  as recall.

## Mechanism

Change the routing decision in `_apply_fact`. On each extracted `(key, value)`:

1. Normalize `key` (unchanged: lowercase, spaces→underscores).
2. Embed the key string → `k_emb` (via existing `src.models.embedder.embed`).
3. Compare `k_emb` against the key-embedding of every existing slot; take the
   max cosine `s*` and its slot `c*`.
4. **Bind-or-add:**
   - `s* ≥ τ` → same attribute → deterministic **UPDATE** of slot `c*`.
   - `s* < τ`  → new attribute → **ADD** a new slot keyed by `key`.

No small-model call is added; the embedding model does the binding.

### State change

`self._slots` grows one field:
`{canonical_key: {"eid": str, "value": str, "key_emb": np.ndarray}}`.
The canonical key is the first phrasing seen; later synonyms bind onto it.

### Code shape (one new subclass, mirrors `framework_typed_only.py`)

`src/memory/managers/framework_semantic.py`:

```python
class FrameworkSemanticManager(FrameworkManager):
    TAU = 0.80   # binding threshold (ablated)

    def _bind(self, k_emb):
        # returns (canonical_key, cosine) of best existing slot, or (None, 0)
        ...

    def _apply_fact(self, key, value, date):
        key, value = normalize(key), value.strip()
        if not key or not value: return
        k_emb = embed([key])[0]
        ckey, s = self._bind(k_emb)
        if ckey is not None and s >= self.TAU:
            self.store.update(self._slots[ckey]["eid"], f"{ckey}: {value}")
            self._slots[ckey]["value"] = value
        else:
            eid = self.store.add(f"{key}: {value}", created_at=date)
            self._slots[key] = {"eid": eid, "value": value, "key_emb": k_emb}
```

(Deletion signals from the parent can be layered on top later, exactly as the
existing `framework.py` does, so this composes with Candidate B if we add it.)

## Evaluation plan

New ablation condition **`+SlotBind`** inserted between `+Slots` (exact) and
`+Del`:

| Condition | Routing |
|-----------|---------|
| Naive | model picks op |
| +Slots | exact string key match |
| **+SlotBind** | **embedding nearest-neighbor key match** |
| +Del | +SlotBind + deletion signals |

Metrics (all from the existing harness):
- **UPDATE recall** on the injection stream (expect ~parity with +Slots: the
  stream has clean keys, so this *shows the synthetic stream understated the
  problem* — a useful honest result).
- **UPDATE recall + QA EM** on **LongMemEval-S and LoCoMo** (expect +SlotBind >
  +Slots: real phrasing varies, so exact slots fragment and semantic binds).
- **Rot score** over turns (expect lower: fewer duplicate-attribute slots).
- **False-merge rate** (new): fraction of binds that merged two ground-truth-
  distinct attributes. Guards the anti-goal.
- **τ sweep** τ ∈ {0.6, 0.7, 0.8, 0.9}: the precision/recall knob → a clean
  ablation figure (over-merge vs fragment).

## Open decisions (your call)

1. **What to embed for binding** — the bare key (`"employer"`), or key+value
   (`"employer: BigBank"`)? Bare key binds on the *attribute* (what we want);
   key+value risks binding on value similarity. Default: **bare key**.
2. **τ default** — 0.80 as the headline, full sweep reported. Override?
3. **Scope** — ship A alone (your pick), or leave the hooks so B (soft-
   invalidation) can layer on later? Default: leave the hooks, build only A.

## What's left to the builder

Implement `framework_semantic.py` + `_bind`, wire a `--condition semantic`
option into `experiments/ablation_run.py`, `longmemeval.py`, `locomo.py`,
write a unit test for `_bind` (synonym binds, distinct-attr doesn't), run the
ablation, regenerate the affected tables/figures (Task 4). Built via TDD.
```
