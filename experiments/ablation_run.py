"""
Ablation study: 3-condition comparison on the controlled injection stream.

Conditions (run in order, one model at a time):
  naive          — baseline, no framework       (reuses phase2_* results)
  typed_only     — typed slots, no deletion signals
  framework_full — typed slots + deletion signals (reuses phase3_* results)

Usage:
  python experiments/ablation_run.py [model] [--condition typed_only|all]

  model:     Ollama model tag (default: qwen3:4b)
  --condition typed_only    run only typed_only condition
  --condition compaction    run only compaction condition
  --condition all           run all new conditions (skips existing results)

Output:
  results/ablation_typed_only_<model>_ops.jsonl
  results/ablation_typed_only_<model>_health.jsonl
  results/ablation_typed_only_<model>_qa.jsonl
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODEL     = next((a for a in sys.argv[1:] if not a.startswith("-")), "qwen3:4b")
CONDITION = "typed_only"
for a in sys.argv[1:]:
    if a.startswith("--condition="):
        CONDITION = a.split("=", 1)[1]

from src.diagnostic.injection import load_stream, qa_pairs, STREAM_STATS
from src.diagnostic.metrics   import per_op_accuracy, store_health_snapshot, rot_curve
from src.eval.metrics          import exact_match, token_f1

STREAM     = load_stream()
N_TURNS    = len(STREAM)
MODEL_SLUG = MODEL.replace(":", "-")
os.makedirs("results", exist_ok=True)


def run_condition(name, manager):
    prefix   = f"ablation_{name}_{MODEL_SLUG}"
    out_ops    = f"results/{prefix}_ops.jsonl"
    out_health = f"results/{prefix}_health.jsonl"
    out_qa     = f"results/{prefix}_qa.jsonl"

    # Skip if already complete
    if os.path.exists(out_ops):
        existing = [l for l in open(out_ops) if l.strip()]
        if len(existing) >= N_TURNS:
            print(f"  [{name}] already complete — skipping")
            return

    print(f"\n{'='*60}")
    print(f"  Condition : {name}")
    print(f"  Model     : {MODEL}")
    print(f"  Turns     : {N_TURNS}  ({STREAM_STATS})")
    print(f"  Output    : {prefix}_*.jsonl")
    print(f"{'='*60}")

    store      = manager.store
    op_records = []
    health_records = []
    call_times = []
    t_start    = time.time()

    f_ops    = open(out_ops,    "w")
    f_health = open(out_health, "w")

    for turn in STREAM:
        tid, gt = turn["turn_id"], turn["op"]

        t0 = time.time()
        manager.ingest("user", turn["content"], turn["date"])
        manager.ingest_session_end()
        elapsed = time.time() - t0
        call_times.append(elapsed)

        # Infer predicted op from what the store recorded
        if hasattr(store, "op_log"):
            log = list(store.op_log)
            store.op_log.clear()
            ops_fired = [{"op": e["op"], "content": e.get("content", "")[:80]} for e in log]
        else:
            ops_fired = []

        # Determine dominant predicted op
        pred = "NOOP"
        for prio in ("DELETE", "UPDATE", "ADD"):
            if any(o["op"] == prio for o in ops_fired):
                pred = prio
                break

        correct = (gt == pred)
        rec = {
            "turn_id": tid, "date": turn["date"], "content": turn["content"],
            "fact_key": turn["fact_key"], "ground_truth_op": gt,
            "predicted_op": pred, "correct": correct,
            "call_time_s": round(elapsed, 2), "ops_fired": ops_fired,
        }
        op_records.append(rec)
        f_ops.write(json.dumps(rec) + "\n"); f_ops.flush()

        if tid % 5 == 0 or tid == N_TURNS - 1:
            snap = store_health_snapshot(store.all_entries(), tid, STREAM)
            health_records.append(snap)
            f_health.write(json.dumps(snap) + "\n"); f_health.flush()

        status = "OK " if correct else "MISS"
        avg    = sum(call_times[-10:]) / len(call_times[-10:])
        pct    = 100 * (tid + 1) / N_TURNS
        print(f"  t{tid:03d} {gt:6s}→{pred:6s} {status} | {(turn['fact_key'] or 'noop'):25s} | {elapsed:.1f}s  [{pct:.0f}% | avg {avg:.1f}s]")

    f_ops.close(); f_health.close()

    # QA pass
    print(f"\n  Running QA pass...")
    f_qa = open(out_qa, "w")
    qa_records = []
    for qi, (question, gold, _) in enumerate(qa_pairs()):
        t0   = time.time()
        pred_ans = manager.query(question, "2025-05-15")
        qt   = time.time() - t0
        hit  = exact_match(gold, pred_ans)
        f1   = token_f1(gold, pred_ans)
        rec  = {"qi": qi, "question": question, "gold": gold, "pred": pred_ans,
                "hit": hit, "f1": f1, "query_time_s": round(qt, 2)}
        qa_records.append(rec)
        f_qa.write(json.dumps(rec) + "\n"); f_qa.flush()
        print(f"  [{qi+1}/{len(qa_pairs())}] {'OK' if hit else 'MISS'} f1={f1:.2f}  Q: {question[:55]}")
    f_qa.close()

    # Summary
    acc   = per_op_accuracy(op_records)
    curve = rot_curve(health_records)
    total = time.time() - t_start
    correct_count = sum(r["correct"] for r in op_records)
    qa_hit = sum(r["hit"] for r in qa_records)

    print(f"\n  {'─'*50}")
    print(f"  Total: {total/60:.1f} min  |  Overall acc: {correct_count/len(op_records):.0%}")
    print(f"  Per-op accuracy:")
    for op in ("ADD", "UPDATE", "DELETE", "NOOP"):
        a = acc[op]
        print(f"    {op:6s}  recall={a['recall']:.3f}  precision={a['precision']:.3f}")
    if health_records:
        h = health_records[-1]
        print(f"  Final store: entries={h.get('n_entries')}  rot={h.get('rot_score')}")
    print(f"  QA: {qa_hit}/{len(qa_records)} EM  f1={sum(r['f1'] for r in qa_records)/len(qa_records):.3f}")


def main():
    conditions = []
    if CONDITION in ("typed_only", "all"):
        from src.memory.managers.framework_typed_only import FrameworkTypedOnlyManager
        conditions.append(("typed_only", FrameworkTypedOnlyManager(MODEL)))
    if CONDITION in ("compaction", "all"):
        from src.memory.managers.framework_compaction import FrameworkCompactionManager
        conditions.append(("compaction", FrameworkCompactionManager(MODEL)))

    if CONDITION in ("softdel", "all"):
        from src.memory.managers.framework_bitemporal import FrameworkBitemporalManager
        conditions.append(("softdel", FrameworkBitemporalManager(MODEL)))

    if not conditions:
        print(f"Unknown condition: {CONDITION}")
        print("Usage: python ablation_run.py [model] [--condition=typed_only|compaction|softdel|all]")
        sys.exit(1)

    for name, manager in conditions:
        run_condition(name, manager)

    print("\nDone. Cross-reference with phase2_* (naive) and phase3_* (framework_full).")


if __name__ == "__main__":
    main()
