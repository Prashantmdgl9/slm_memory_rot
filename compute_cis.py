"""
Compute 95% Wilson score confidence intervals on all existing n=20 results.

Covers:
  - Phase 1 QA EM (by model × condition)
  - Phase 2/3 op recall — UPDATE, DELETE (by model × condition)
  - 4-condition ablation UPDATE recall (by model × condition)

Output: results/confidence_intervals.json  +  printed table
"""
import json, math, os
from pathlib import Path

RESULTS = Path("results")
Z = 1.96  # 95% CI


# ── Wilson score CI ────────────────────────────────────────────────────────────
def wilson(k, n, z=Z):
    """Return (lo, mid, hi) Wilson score 95% CI for k successes in n trials."""
    if n == 0:
        return (None, None, None)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - margin), p, min(1.0, center + margin))


def fmt(ci):
    lo, mid, hi = ci
    if mid is None:
        return "    —    "
    return f"{mid:.0%} [{lo:.0%}–{hi:.0%}]"


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    out = []
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return out


def qa_ci(records):
    hits = [r["hit"] for r in records if "hit" in r]
    return wilson(sum(hits), len(hits))


def recall_ci(records, op):
    gt = [r for r in records if r.get("ground_truth_op") == op]
    hits = [r for r in gt if r.get("predicted_op") == op]
    return wilson(len(hits), len(gt))


def precision_ci(records, op):
    pred = [r for r in records if r.get("predicted_op") == op]
    hits = [r for r in pred if r.get("ground_truth_op") == op]
    return wilson(len(hits), len(pred))


# ── Models / conditions ────────────────────────────────────────────────────────
SMALL_MODELS = [
    ("qwen3:4b",    "qwen3-4b"),
    ("qwen3:8b",    "qwen3-8b"),
    ("llama3.2:3b", "llama3.2-3b"),
    ("llama3.1:8b", "llama3.1-8b"),
    ("qwen2.5:7b",  "qwen2.5-7b"),
]

P1_CONDITIONS = ["no_memory", "rag_only", "naive", "full_context"]

ABLATION_CONDITIONS = [
    ("naive",      "phase2"),
    ("+slots",     "ablation_typed_only"),
    ("+del",       "phase3"),
    ("+compact",   "ablation_compaction"),
]

all_cis = {}

# ── Phase 1 QA ────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("PHASE 1 — QA EM with 95% Wilson CIs  (n=20 per cell)")
print("="*80)
header = f"{'Model':14}" + "".join(f"  {c:>22}" for c in P1_CONDITIONS)
print(header)
print("─" * len(header))

p1_cis = {}
for model, slug in SMALL_MODELS:
    row = f"{model:14}"
    p1_cis[model] = {}
    for cond in P1_CONDITIONS:
        recs = load_jsonl(RESULTS / f"phase1_{cond}_{slug}_n20.jsonl")
        ci = qa_ci(recs)
        p1_cis[model][cond] = {"n": len(recs), "ci": ci}
        row += f"  {fmt(ci):>22}"
    print(row)

# Groq reference
groq_recs = load_jsonl(RESULTS / "phase1_naive_llama3.3-70b-groq_n20.jsonl")
groq_ci = qa_ci(groq_recs)
print(f"{'llama3.3-70b (groq)':14}  {'':22}  {fmt(groq_ci):>22}  [n={len(groq_recs)}]")

all_cis["phase1_qa"] = p1_cis

# ── Phase 2/3 op recall ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("PHASE 2/3 — UPDATE & DELETE recall with 95% Wilson CIs  (n=turns per op)")
print("="*80)

OP_CONDITIONS = [
    ("naive",   "phase2"),
    ("framework", "phase3"),
]
print(f"{'Model':14}  {'UPDATE naive':>16}  {'UPDATE +fw':>16}  {'DELETE naive':>16}  {'DELETE +fw':>16}")
print("─" * 90)

op_cis = {}
for model, slug in SMALL_MODELS:
    op_cis[model] = {}
    row_parts = [f"{model:14}"]
    for label, prefix in OP_CONDITIONS:
        recs = load_jsonl(RESULTS / f"{prefix}_{slug}_ops.jsonl")
        u_ci = recall_ci(recs, "UPDATE")
        d_ci = recall_ci(recs, "DELETE")
        op_cis[model][label] = {
            "update_recall": {"n": len([r for r in recs if r.get("ground_truth_op")=="UPDATE"]), "ci": u_ci},
            "delete_recall": {"n": len([r for r in recs if r.get("ground_truth_op")=="DELETE"]), "ci": d_ci},
        }
        row_parts.append(f"  {fmt(u_ci):>16}")
    for label, prefix in OP_CONDITIONS:
        recs = load_jsonl(RESULTS / f"{prefix}_{slug}_ops.jsonl")
        d_ci = recall_ci(recs, "DELETE")
        row_parts.append(f"  {fmt(d_ci):>16}")
    print("".join(row_parts))

all_cis["phase2_3_ops"] = op_cis

# ── 4-condition ablation UPDATE recall ────────────────────────────────────────
print("\n" + "="*80)
print("ABLATION — UPDATE recall with 95% Wilson CIs  (n=26 UPDATE turns)")
print("="*80)
cond_labels = [c for c, _ in ABLATION_CONDITIONS]
print(f"{'Model':14}" + "".join(f"  {c:>18}" for c in cond_labels))
print("─" * 90)

abl_cis = {}
for model, slug in SMALL_MODELS:
    abl_cis[model] = {}
    row = f"{model:14}"
    for label, prefix in ABLATION_CONDITIONS:
        recs = load_jsonl(RESULTS / f"{prefix}_{slug}_ops.jsonl")
        ci = recall_ci(recs, "UPDATE")
        abl_cis[model][label] = {"n": len([r for r in recs if r.get("ground_truth_op")=="UPDATE"]), "ci": ci}
        row += f"  {fmt(ci):>18}"
    print(row)

all_cis["ablation_update_recall"] = abl_cis

# ── Save ───────────────────────────────────────────────────────────────────────
def ci_to_dict(ci):
    lo, mid, hi = ci
    return {"lo": round(lo, 4) if lo is not None else None,
            "mid": round(mid, 4) if mid is not None else None,
            "hi": round(hi, 4) if hi is not None else None}

def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return ci_to_dict(obj)
    return obj

out_path = RESULTS / "confidence_intervals.json"
with open(out_path, "w") as f:
    json.dump(serialize(all_cis), f, indent=2)

print(f"\nSaved → {out_path}")
print("\nNote: CI width ~±15-20pp for n=20, ~±7pp for n=80 op turns.")
