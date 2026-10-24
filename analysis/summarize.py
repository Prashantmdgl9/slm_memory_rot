"""
Print the full comparison table: naive vs framework, all 4 models.

Usage:
  python analysis/summarize.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.diagnostic.metrics import per_op_accuracy, rot_curve

MODELS = [
    ("qwen3:4b",    "qwen3-4b"),
    ("llama3.2:3b", "llama3.2-3b"),
    ("qwen3:8b",    "qwen3-8b"),
    ("llama3.1:8b", "llama3.1-8b"),
]


def load_results(prefix, model_slug):
    ops_path    = f"results/{prefix}_{model_slug}_ops.jsonl"
    health_path = f"results/{prefix}_{model_slug}_health.jsonl"
    qa_path     = f"results/{prefix}_{model_slug}_qa.jsonl"

    if not os.path.exists(ops_path):
        return None

    ops    = [json.loads(l) for l in open(ops_path)]
    health = [json.loads(l) for l in open(health_path)] if os.path.exists(health_path) else []
    qa     = [json.loads(l) for l in open(qa_path)]     if os.path.exists(qa_path)     else []

    acc   = per_op_accuracy(ops)
    curve = rot_curve(health)
    fh    = health[-1] if health else {}
    qa_em = sum(r["hit"] for r in qa) if qa else None
    qa_f1 = sum(r["f1"] for r in qa) / len(qa) if qa else None

    return {
        "acc":       acc,
        "curve":     curve,
        "rot_score": fh.get("rot_score", "—"),
        "entries":   fh.get("n_entries",  "—"),
        "qa_em":     qa_em,
        "qa_f1":     round(qa_f1, 3) if qa_f1 is not None else "—",
        "n_qa":      len(qa),
    }


def fmt(v, digits=3):
    if v is None or v == "—":
        return "  —  "
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def print_table(prefix, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    header = f"{'Model':20s} {'ADD-R':6s} {'UPD-R':6s} {'UPD-P':6s} {'DEL-R':6s} {'NOOP-R':7s} {'OvAcc':6s} {'Entries':8s} {'RotScr':7s} {'Slope':7s} {'QA-EM':8s} {'QA-F1':6s}"
    print(header)
    print("-" * 80)

    for model_name, model_slug in MODELS:
        r = load_results(prefix, model_slug)
        if r is None:
            print(f"  {model_name:18s}  (no results)")
            continue
        a = r["acc"]
        c = r["curve"]
        row = (
            f"{model_name:20s}"
            f" {fmt(a['ADD']['recall']):6s}"
            f" {fmt(a['UPDATE']['recall']):6s}"
            f" {fmt(a['UPDATE']['precision']):6s}"
            f" {fmt(a['DELETE']['recall']):6s}"
            f" {fmt(a['NOOP']['recall']):7s}"
            f" {fmt(a['overall_accuracy']):6s}"
            f" {str(r['entries']):8s}"
            f" {str(r['rot_score']):7s}"
            f" {fmt(c['slope']):7s}"
            f" {str(r['qa_em'])+'/' + str(r['n_qa']) if r['qa_em'] is not None else '—':8s}"
            f" {fmt(r['qa_f1']):6s}"
        )
        print(row)


def print_ablation():
    """Side-by-side naive vs framework for UPDATE recall (the key metric)."""
    print(f"\n{'='*60}")
    print(f"  UPDATE recall: naive → framework  (key ablation)")
    print(f"{'='*60}")
    print(f"{'Model':20s} {'Naive UPD-R':12s} {'FW UPD-R':10s} {'Delta':8s} {'Naive QA':10s} {'FW QA':8s}")
    print("-" * 60)
    for model_name, model_slug in MODELS:
        n = load_results("phase2", model_slug)
        f = load_results("phase3", model_slug)
        if n is None:
            continue
        n_upd = n["acc"]["UPDATE"]["recall"]
        f_upd = f["acc"]["UPDATE"]["recall"] if f else None
        delta = (f_upd - n_upd) if f_upd is not None else None
        n_qa  = f"{n['qa_em']}/{n['n_qa']}" if n["qa_em"] is not None else "—"
        f_qa  = f"{f['qa_em']}/{f['n_qa']}" if f and f["qa_em"] is not None else "—"
        print(
            f"{model_name:20s}"
            f" {fmt(n_upd):12s}"
            f" {fmt(f_upd) if f_upd is not None else '—':10s}"
            f" {('+' if delta and delta > 0 else '') + fmt(delta) if delta is not None else '—':8s}"
            f" {n_qa:10s}"
            f" {f_qa:8s}"
        )


if __name__ == "__main__":
    print_table("phase2", "NAIVE MANAGER — per-op accuracy across models")
    print_table("phase3", "FRAMEWORK MANAGER — per-op accuracy across models")
    print_ablation()
