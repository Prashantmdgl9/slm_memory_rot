"""
Phase 3: Framework manager on the controlled injection stream.

Same 80-turn stream as Phase 2. Compare per-op accuracy and rot metrics
against the naive manager results from Phase 2.

Usage:
  python experiments/phase3_framework.py [model] [--dry-run]
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODEL   = next((a for a in sys.argv[1:] if not a.startswith("-")), "qwen3:4b")
DRY_RUN = "--dry-run" in sys.argv

from src.diagnostic.injection  import load_stream, qa_pairs, STREAM_STATS
from src.diagnostic.metrics    import per_op_accuracy, store_health_snapshot, rot_curve
from src.memory.managers.framework import FrameworkManager
from src.eval.metrics           import exact_match, token_f1

STREAM     = load_stream()
N_TURNS    = len(STREAM)
MODEL_SLUG = MODEL.replace(":", "-")
OUT_OPS    = f"results/phase3_{MODEL_SLUG}_ops.jsonl"
OUT_HEALTH = f"results/phase3_{MODEL_SLUG}_health.jsonl"
OUT_QA     = f"results/phase3_{MODEL_SLUG}_qa.jsonl"

os.makedirs("results", exist_ok=True)


def _predicted_op(log_before_len, store):
    new_ops = store.op_log[log_before_len:]
    if not new_ops:
        return "NOOP"
    for prio in ("DELETE", "UPDATE", "ADD"):
        if any(e["op"] == prio for e in new_ops):
            return prio
    return "NOOP"


def _eta(elapsed, done, total):
    if done == 0:
        return "?"
    rate = elapsed / done
    m, s = divmod(int(rate * (total - done)), 60)
    return f"{m}m{s:02d}s"


def main():
    print(f"=== Phase 3 Framework Diagnostic ===")
    print(f"Model   : {MODEL}")
    print(f"Turns   : {N_TURNS}  ({STREAM_STATS})")
    print(f"Outputs : {OUT_OPS}, {OUT_HEALTH}, {OUT_QA}\n")

    if DRY_RUN:
        print("Dry run — skipping model calls.")
        return

    manager = FrameworkManager(MODEL)
    store   = manager.store

    op_records     = []
    health_records = []
    call_times     = []
    t_start        = time.time()

    f_ops    = open(OUT_OPS,    "w")
    f_health = open(OUT_HEALTH, "w")

    print("Running injection stream (framework manager)...")

    for turn in STREAM:
        tid = turn["turn_id"]
        gt  = turn["op"]

        log_before_len = len(store.op_log)
        store.op_log.clear()

        t0 = time.time()
        manager.ingest("user", turn["content"], turn["date"])
        manager.ingest_session_end()
        elapsed_call = time.time() - t0
        call_times.append(elapsed_call)

        predicted = _predicted_op(0, store)
        correct   = (gt == predicted)

        op_rec = {
            "turn_id":         tid,
            "date":            turn["date"],
            "content":         turn["content"],
            "fact_key":        turn["fact_key"],
            "ground_truth_op": gt,
            "predicted_op":    predicted,
            "correct":         correct,
            "call_time_s":     round(elapsed_call, 2),
            "ops_fired":       [{"op": e["op"], "content": e["content"][:80]}
                                 for e in store.op_log],
            "slots":           dict(manager._slots),
        }
        op_records.append(op_rec)
        f_ops.write(json.dumps(op_rec) + "\n")
        f_ops.flush()

        if tid % 5 == 0 or tid == N_TURNS - 1:
            snap = store_health_snapshot(store.all_entries(), tid, STREAM)
            health_records.append(snap)
            f_health.write(json.dumps(snap) + "\n")
            f_health.flush()

        manager._buf_turns = []
        manager._buf_date  = None

        elapsed_total = time.time() - t_start
        avg = sum(call_times[-10:]) / len(call_times[-10:])
        eta = _eta(elapsed_total, tid + 1, N_TURNS)
        pct = 100 * (tid + 1) / N_TURNS
        status = "OK " if correct else "MISS"
        print(f"  [{tid+1:3d}/{N_TURNS} | {pct:5.1f}% | avg {avg:.1f}s | ETA {eta}]"
              f"  t{tid:03d} {gt:6s}→{predicted:6s} {status}"
              f" | {turn['fact_key'] or 'noop':25s} | {elapsed_call:.1f}s")

    f_ops.close()
    f_health.close()

    # ── QA pass ──────────────────────────────────────────────────────────
    print("\n\nRunning QA pass...")
    qa_records = []
    f_qa = open(OUT_QA, "w")

    for qi, (question, gold, _) in enumerate(qa_pairs()):
        t0   = time.time()
        pred = manager.query(question, "2025-05-15")
        qt   = time.time() - t0
        hit  = exact_match(gold, pred)
        f1   = token_f1(gold, pred)
        rec  = {"qi": qi, "question": question, "gold": gold, "pred": pred,
                "hit": hit, "f1": f1, "query_time_s": round(qt, 2)}
        qa_records.append(rec)
        f_qa.write(json.dumps(rec) + "\n")
        f_qa.flush()
        status = "OK " if hit else "MISS"
        print(f"  [{qi+1:2d}/14] {status} f1={f1:.2f}  Q: {question[:60]}")
        print(f"         gold: {gold!r}   pred: {pred[:80]!r}")

    f_qa.close()

    # ── Summary ───────────────────────────────────────────────────────────
    acc   = per_op_accuracy(op_records)
    curve = rot_curve(health_records)

    print(f"\n{'='*60}")
    print(f"Model: {MODEL}  (FRAMEWORK)  |  Total: {(time.time()-t_start)/60:.1f} min")
    print(f"\nPer-op accuracy:")
    for op in ("ADD", "UPDATE", "DELETE", "NOOP"):
        a = acc[op]
        print(f"  {op:6s}  p={a['precision']:.3f}  r={a['recall']:.3f}"
              f"  (tp={a['tp']} fp={a['fp']} fn={a['fn']})")
    print(f"  Overall: {acc['overall_accuracy']:.3f}")
    print(f"\nRot:  slope={curve['slope']}  r2={curve['r2']}")
    fh = health_records[-1] if health_records else {}
    print(f"Store: entries={fh.get('n_entries')}  "
          f"stale={fh.get('stale_count')}  dupes={fh.get('duplicate_count')}  "
          f"rot_score={fh.get('rot_score')}")
    qa_hit = sum(r["hit"] for r in qa_records)
    print(f"QA: {qa_hit}/14 EM  mean_f1={sum(r['f1'] for r in qa_records)/14:.3f}")
    print(f"{'='*60}")

    # Print active slots for inspection
    print(f"\nFinal slots ({len(manager._slots)}):")
    for k, v in sorted(manager._slots.items()):
        print(f"  {k:25s}: {v['value']}")


if __name__ == "__main__":
    main()
