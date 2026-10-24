"""
Generate all paper figures.

fig1  — 4-condition ablation UPDATE recall (main result)
fig2  — per-op recall: naive
fig3  — per-op recall: framework
fig4  — rot curves: naive vs framework
fig5  — Phase 1 LongMemEval QA: all 4 conditions
fig6  — DELETE recall vs precision tradeoff
fig7  — UPDATE recall with 95% CI error bars

Usage:
  python analysis/plot_results.py
"""
import json, os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.diagnostic.metrics import per_op_accuracy, rot_curve

MODELS = [
    ("qwen3:4b",    "qwen3-4b"),
    ("qwen3:8b",    "qwen3-8b"),
    ("llama3.2:3b", "llama3.2-3b"),
    ("llama3.1:8b", "llama3.1-8b"),
    ("qwen2.5:7b",  "qwen2.5-7b"),
]
MODEL_LABELS = [m[0] for m in MODELS]
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette
C_NAIVE    = "#4C72B0"
C_SLOTS    = "#55A868"
C_DEL      = "#DD8452"
C_COMPACT  = "#8172B2"
C_FW       = "#DD8452"
OP_COLORS  = {"ADD": "#4C72B0", "UPDATE": "#55A868", "DELETE": "#C44E52", "NOOP": "#8172B2"}

ABLATION_CONDITIONS = [
    ("Naive",        "phase2",              C_NAIVE),
    ("+Typed Slots", "ablation_typed_only", C_SLOTS),
    ("+Del Signals", "phase3",              C_DEL),
    ("+Compaction",  "ablation_compaction", C_COMPACT),
]

Z95 = 1.96


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    out = []
    try:
        for line in open(path):
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except: pass
    except FileNotFoundError: pass
    return out

def load_ops(prefix, slug):
    return load_jsonl(f"results/{prefix}_{slug}_ops.jsonl")

def load_health(prefix, slug):
    return load_jsonl(f"results/{prefix}_{slug}_health.jsonl")

def recall(ops, op):
    gt = [r for r in ops if r.get("ground_truth_op") == op]
    if not gt: return 0.0, 0
    return sum(1 for r in gt if r.get("predicted_op") == op) / len(gt), len(gt)

def precision(ops, op):
    pred = [r for r in ops if r.get("predicted_op") == op]
    if not pred: return 0.0, 0
    return sum(1 for r in pred if r.get("ground_truth_op") == op) / len(pred), len(pred)

def wilson_err(p, n):
    """Return (lo_err, hi_err) suitable for matplotlib errorbar."""
    if n == 0: return 0, 0
    denom = 1 + Z95**2 / n
    center = (p + Z95**2 / (2*n)) / denom
    margin = (Z95 / denom) * math.sqrt(p*(1-p)/n + Z95**2/(4*n**2))
    return center - max(0, center - margin), min(1, center + margin) - center

def add_value_labels(ax, bars, vals, fmt="{:.2f}", offset=0.01, fontsize=8, rotation=0):
    for bar, val in zip(bars, vals):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + offset,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=fontsize, rotation=rotation)


# ── Fig 1: 4-condition ablation UPDATE recall ──────────────────────────────────
def fig1_ablation_update_recall():
    n_models = len(MODELS)
    n_conds  = len(ABLATION_CONDITIONS)
    x = np.arange(n_models)
    w = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    offsets = np.linspace(-(n_conds-1)/2, (n_conds-1)/2, n_conds) * w

    for ci, (label, prefix, color) in enumerate(ABLATION_CONDITIONS):
        vals, errs_lo, errs_hi = [], [], []
        for _, slug in MODELS:
            ops = load_ops(prefix, slug)
            r, n = recall(ops, "UPDATE")
            vals.append(r)
            lo, hi = wilson_err(r, n)
            errs_lo.append(lo)
            errs_hi.append(hi)

        bars = ax.bar(x + offsets[ci], vals, w, label=label,
                      color=color, alpha=0.88)
        ax.errorbar(x + offsets[ci], vals,
                    yerr=[errs_lo, errs_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylabel("UPDATE Recall", fontsize=11)
    ax.set_title("UPDATE Recall: 4-Condition Ablation", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    path = f"{OUT_DIR}/fig1_ablation_update_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 2 & 3: Per-op recall grouped bars ─────────────────────────────────────
def fig_per_op(prefix, title, fname):
    ops_list = ["ADD", "UPDATE", "DELETE", "NOOP"]
    data = {op: [] for op in ops_list}
    for _, slug in MODELS:
        ops = load_ops(prefix, slug)
        acc = per_op_accuracy(ops) if ops else {}
        for op in ops_list:
            data[op].append(acc.get(op, {}).get("recall", 0) if acc else 0)

    x = np.arange(len(MODELS))
    w = 0.18
    offsets = np.linspace(-1.5, 1.5, 4) * w

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, op in enumerate(ops_list):
        bars = ax.bar(x + offsets[i], data[op], w,
                      label=op, color=OP_COLORS[op], alpha=0.88)
        add_value_labels(ax, bars, data[op], fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9, ncol=4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    path = f"{OUT_DIR}/{fname}"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Honest rot metric: divergence from ground-truth world state ────────────────
def store_divergence_curve(rows):
    """Point-in-time memory rot = number of attributes the store is currently
    WRONG about, relative to ground truth. Counts three failure modes:
      - missing:    a live fact the store never recorded (or wrongly deleted)
      - stale:      the store holds an older, superseded value
      - undeleted:  a negated fact the store still holds
    Unlike a raw stale-entry count, this correctly penalizes an empty store
    (NOOP collapse), which is missing every fact it should hold.
    Reconstructed from per-turn ground_truth_op / predicted_op / fact_key.
    Returns (turns, errors)."""
    gt_turn, gt_live = {}, {}   # ground truth: latest-value turn, is-live
    st_turn, st_live = {}, {}   # store belief: latest-value turn, is-present
    turns, errors = [], []
    for r in rows:
        t, fk = r["turn_id"], r["fact_key"]
        gop, pop = r["ground_truth_op"], r["predicted_op"]
        if fk:
            if gop in ("ADD", "UPDATE"): gt_turn[fk] = t; gt_live[fk] = True
            elif gop == "DELETE":        gt_live[fk] = False
            if pop in ("ADD", "UPDATE"): st_turn[fk] = t; st_live[fk] = True
            elif pop == "DELETE":        st_live[fk] = False
        err = 0
        for k in gt_live:
            if gt_live[k]:
                if not st_live.get(k, False):        err += 1   # missing
                elif st_turn.get(k) != gt_turn.get(k): err += 1  # stale
            else:
                if st_live.get(k, False):            err += 1   # undeleted
        turns.append(t); errors.append(err)
    return turns, errors


# ── Fig 4: Rot curves (per model) ──────────────────────────────────────────────
def fig4_rot_curves():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    axes = axes.flatten()

    for i, (label, slug) in enumerate(MODELS):
        ax = axes[i]
        on = load_ops("phase2", slug)
        of = load_ops("phase3", slug)

        if on:
            nx, ny = store_divergence_curve(on)
            ax.plot(nx, ny, color=C_NAIVE, linewidth=2, label="Naive",
                    marker="o", markersize=3)
        if of:
            fx, fy = store_divergence_curve(of)
            ax.plot(fx, fy, color=C_FW, linewidth=2, label="Framework",
                    marker="s", markersize=3)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Turn", fontsize=9)
        ax.set_ylabel("Incorrect facts in memory", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 80)
        ax.set_ylim(bottom=0)

    # hide unused subplot
    axes[-1].set_visible(False)
    fig.suptitle("Memory Rot: Incorrect Facts in Memory (Naive vs. Framework)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = f"{OUT_DIR}/fig4_rot_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 8: Aggregate before/after rot (LinkedIn hero panel) ────────────────────
def fig8_rot_before_after():
    """Single-panel mean rot across all models: the punchy before/after story."""
    checkpoints = list(range(0, 80, 5)) + [79]

    def mean_curve(prefix):
        per_turn = {t: [] for t in checkpoints}
        for _, slug in MODELS:
            rows = load_ops(prefix, slug)
            if not rows:
                continue
            turns, errs = store_divergence_curve(rows)
            lut = dict(zip(turns, errs))
            for t in checkpoints:
                if t in lut:
                    per_turn[t].append(lut[t])
        xs = [t for t in checkpoints if per_turn[t]]
        ys = [sum(per_turn[t]) / len(per_turn[t]) for t in xs]
        return xs, ys

    nx, ny = mean_curve("phase2")
    fx, fy = mean_curve("phase3")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(nx, ny, color=C_NAIVE, linewidth=3, marker="o", markersize=6,
            label="Before: Naive LLM memory manager")
    ax.plot(fx, fy, color=C_FW, linewidth=3, marker="s", markersize=6,
            label="After: The Small Librarian")
    ax.fill_between(nx, ny, fy, color=C_NAIVE, alpha=0.08)

    # Annotate the end-of-stream gap
    if nx and fx:
        ax.annotate(f"{ny[-1]:.0f}", (nx[-1], ny[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=12, fontweight="bold", color=C_NAIVE, va="center")
        ax.annotate(f"{fy[-1]:.0f}", (fx[-1], fy[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=12, fontweight="bold", color=C_FW, va="center")

    ax.set_xlabel("Conversation turn", fontsize=12)
    ax.set_ylabel("Incorrect facts in memory\n(stale + missing + undeleted)", fontsize=12)
    ax.set_title("Memory Rot Compounds Without Deterministic Routing\n"
                 "Mean across five on-device models (3–8B)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 82)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11, loc="upper left", frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig8_rot_before_after.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 5: Phase 1 LongMemEval QA ─────────────────────────────────────────────
def fig5_phase1_qa():
    P1_CONDITIONS = [
        ("No Memory",    "no_memory",    "#C44E52"),
        ("RAG Only",     "rag_only",     "#4C72B0"),
        ("Naive",        "naive",        "#8172B2"),
        ("Full Context", "full_context", "#CCB974"),
    ]

    n_models = len(MODELS)
    n_conds  = len(P1_CONDITIONS)
    x = np.arange(n_models)
    w = 0.18
    offsets = np.linspace(-(n_conds-1)/2, (n_conds-1)/2, n_conds) * w

    fig, ax = plt.subplots(figsize=(11, 5))

    for ci, (label, cond, color) in enumerate(P1_CONDITIONS):
        vals, errs_lo, errs_hi = [], [], []
        for _, slug in MODELS:
            recs = load_jsonl(f"results/phase1_{cond}_{slug}_n20.jsonl")
            hits = [r["hit"] for r in recs if "hit" in r]
            p = sum(hits) / len(hits) if hits else 0
            n = len(hits)
            vals.append(p)
            lo, hi = wilson_err(p, n)
            errs_lo.append(lo)
            errs_hi.append(hi)

        bars = ax.bar(x + offsets[ci], vals, w, label=label,
                      color=color, alpha=0.88)
        ax.errorbar(x + offsets[ci], vals,
                    yerr=[errs_lo, errs_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=10)
    ax.set_ylabel("QA Exact Match", fontsize=11)
    ax.set_title("Phase 1: LongMemEval-S QA (n=20, 95% CI)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, ncol=4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    path = f"{OUT_DIR}/fig5_phase1_qa.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 6: DELETE recall vs precision scatter ──────────────────────────────────
def fig6_delete_tradeoff():
    CONDITIONS = [
        ("Naive",        "phase2",              C_NAIVE,   "o"),
        ("+Typed Slots", "ablation_typed_only", C_SLOTS,   "s"),
        ("+Del Signals", "phase3",              C_DEL,     "^"),
        ("+Compaction",  "ablation_compaction", C_COMPACT, "D"),
    ]

    # Short labels for annotation
    MODEL_SHORT = {
        "qwen3-4b":    "Q3-4B",
        "qwen3-8b":    "Q3-8B",
        "llama3.2-3b": "L3.2-3B",
        "llama3.1-8b": "L3.1-8B",
        "qwen2.5-7b":  "Q2.5-7B",
    }

    fig, ax = plt.subplots(figsize=(8, 6.5))

    for label, prefix, color, marker in CONDITIONS:
        rec_vals, prec_vals, model_labels = [], [], []
        for _, slug in MODELS:
            ops = load_ops(prefix, slug)
            r, nr = recall(ops, "DELETE")
            p, np_ = precision(ops, "DELETE")
            if nr > 0:
                rec_vals.append(r)
                prec_vals.append(p)
                model_labels.append(MODEL_SHORT.get(slug, slug))

        if rec_vals:
            ax.scatter(rec_vals, prec_vals, label=label, color=color,
                       marker=marker, s=100, alpha=0.85, zorder=3)
            # Annotate each point with its model name
            for rx, py, mlabel in zip(rec_vals, prec_vals, model_labels):
                ax.annotate(mlabel, (rx, py),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=7, color=color, alpha=0.9)

    ax.set_xlabel("DELETE Recall", fontsize=11)
    ax.set_ylabel("DELETE Precision", fontsize=11)
    ax.set_title("DELETE Recall vs. Precision by Condition", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.axline((0, 0), (1, 1), color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label="precision = recall")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = f"{OUT_DIR}/fig6_delete_tradeoff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Fig 7: UPDATE recall with CI error bars per model ─────────────────────────
def fig7_update_recall_ci():
    CONDITIONS = [
        ("Naive",        "phase2",              C_NAIVE),
        ("+Typed Slots", "ablation_typed_only", C_SLOTS),
        ("+Del Signals", "phase3",              C_DEL),
        ("+Compaction",  "ablation_compaction", C_COMPACT),
    ]

    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 4.5), sharey=True)

    for ai, (model_label, slug) in enumerate(MODELS):
        ax = axes[ai]
        labels = [c[0] for c in CONDITIONS]
        vals, errs_lo, errs_hi = [], [], []
        for _, prefix, _ in CONDITIONS:
            ops = load_ops(prefix, slug)
            r, n = recall(ops, "UPDATE")
            vals.append(r)
            lo, hi = wilson_err(r, n)
            errs_lo.append(lo)
            errs_hi.append(hi)

        colors = [c[2] for c in CONDITIONS]
        y = np.arange(len(CONDITIONS))
        bars = ax.barh(y, vals, color=colors, alpha=0.88, height=0.6)
        ax.errorbar(vals, y,
                    xerr=[errs_lo, errs_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

        ax.set_yticks(y)
        ax.set_yticklabels(labels if ai == 0 else [], fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_title(model_label, fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.axvline(0, color="black", linewidth=0.5)

        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{val:.0%}", va="center", fontsize=8)

    fig.suptitle("UPDATE Recall with 95% CI — All Models & Conditions",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = f"{OUT_DIR}/fig7_update_recall_ci.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1_ablation_update_recall()
    fig_per_op("phase2", "Per-Op Recall: Naive Manager",     "fig2_per_op_naive.png")
    fig_per_op("phase3", "Per-Op Recall: Framework Manager", "fig3_per_op_fw.png")
    fig4_rot_curves()
    fig5_phase1_qa()
    fig6_delete_tradeoff()
    fig7_update_recall_ci()
    fig8_rot_before_after()
    print(f"\nAll 8 figures saved to ./{OUT_DIR}/")
