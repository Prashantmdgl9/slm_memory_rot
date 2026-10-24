"""
Phase 2: Controlled-injection diagnostic.

Feeds the injection stream turn-by-turn to a memory manager, records
what op was predicted vs ground truth, and tracks store health per turn.

Usage:
  python experiments/phase2_injection.py [model] [--dry-run]

  model:    any Ollama model tag (default: qwen3:4b)
  --dry-run: skip Ollama calls, print stream stats only

Output:
  results/phase2_<model>_ops.jsonl      — per-turn op accuracy log
  results/phase2_<model>_health.jsonl   — per-turn store health snapshots
  results/phase2_<model>_qa.jsonl       — QA results after full ingestion
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MODEL   = next((a for a in sys.argv[1:] if not a.startswith("-")), "qwen3:4b")
DRY_RUN = "--dry-run" in sys.argv

from src.diagnostic.injection import load_stream, qa_pairs, STREAM_STATS
from src.diagnostic.metrics   import per_op_accuracy, store_health_snapshot, rot_curve
from src.memory.managers.naive import NaiveManager
from src.eval.metrics          import exact_match, token_f1

STREAM   = load_stream()
N_TURNS  = len(STREAM)
MODEL_SLUG = MODEL.replace(":", "-")
OUT_OPS    = f"results/phase2_{MODEL_SLUG}_ops.jsonl"
OUT_HEALTH = f"results/phase2_{MODEL_SLUG}_health.jsonl"
OUT_QA     = f"results/phase2_{MODEL_SLUG}_qa.jsonl"

os.makedirs("results", exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────

def _predicted_op(op_log_before, op_log_after):
    """
    Infer what op the manager predicted for a single turn.
    op_log_before / after = store.op_log snapshots.
    Returns the dominant op fired, or NOOP if nothing fired.
    """
    new_ops = op_log_after[len(op_log_before):]
    if not new_ops:
        return "NOOP"
    # Priority: DELETE > UPDATE > ADD  (so we don't double-count)
    for prio_op in ("DELETE", "UPDATE", "ADD"):
        if any(e["op"] == prio_op for e in new_ops):
            return prio_op
    return "NOOP"


def _eta(elapsed_total, done, total):
    if done == 0:
        return "?"
    rate = elapsed_total / done
    remaining = rate * (total - done)
    m, s = divmod(int(remaining), 60)
    return f"{m}m{s:02d}s"


def _progress(turn_idx, elapsed, call_times):
    avg = sum(call_times[-10:]) / len(call_times[-10:]) if call_times else 0
    eta = _eta(elapsed, turn_idx, N_TURNS)
    pct = 100 * turn_idx / N_TURNS
    print(f"\r  [{turn_idx:3d}/{N_TURNS} turns | {pct:5.1f}% | avg {avg:.1f}s/call | ETA {eta}]",
          end="", flush=True)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    print(f"=== Phase 2 Injection Diagnostic ===")
    print(f"Model   : {MODEL}")
    print(f"Turns   : {N_TURNS}  ({STREAM_STATS})")
    print(f"Dry run : {DRY_RUN}")
    print(f"Outputs : {OUT_OPS}, {OUT_HEALTH}, {OUT_QA}\n")

    if DRY_RUN:
        for turn in STREAM:
            print(f"  [{turn['turn_id']:3d}] {turn['op']:6s}  {turn['fact_key'] or 'n/a':25s}  {turn['content'][:70]}")
        print(f"\nStream stats: {STREAM_STATS}")
        return

    manager = NaiveManager(MODEL)
    store   = manager.store

    op_records     = []
    health_records = []
    call_times     = []
    t_start        = time.time()

    f_ops    = open(OUT_OPS,    "w")
    f_health = open(OUT_HEALTH, "w")

    print("Running injection stream...")

    for turn in STREAM:
        tid  = turn["turn_id"]
        gt   = turn["op"]

        # Snapshot op_log before this turn
        log_before = list(store.op_log)
        store.op_log.clear()

        t0 = time.time()
        manager.ingest("user", turn["content"], turn["date"])
        manager._flush()   # flush THIS turn now so op_log reflects its decision (per-turn diagnostic)
        elapsed_call = time.time() - t0
        call_times.append(elapsed_call)

        # Snapshot after
        log_after  = list(store.op_log)
        predicted  = _predicted_op([], log_after)

        correct = (gt == predicted)
        op_rec = {
            "turn_id":          tid,
            "date":             turn["date"],
            "content":          turn["content"],
            "fact_key":         turn["fact_key"],
            "ground_truth_op":  gt,
            "predicted_op":     predicted,
            "correct":          correct,
            "call_time_s":      round(elapsed_call, 2),
            "ops_fired":        [{"op": e["op"], "content": e["content"][:80]} for e in log_after],
        }
        op_records.append(op_rec)
        f_ops.write(json.dumps(op_rec) + "\n")
        f_ops.flush()

        # Store health snapshot every 5 turns
        if tid % 5 == 0 or tid == N_TURNS - 1:
            snap = store_health_snapshot(store.all_entries(), tid, STREAM)
            health_records.append(snap)
            f_health.write(json.dumps(snap) + "\n")
            f_health.flush()

        # Reset the per-turn buffer so next ingest starts fresh
        manager._buf_turns = []
        manager._buf_date  = None

        status = "OK " if correct else "MISS"
        elapsed_total = time.time() - t_start
        _progress(tid + 1, elapsed_total, call_times)
        print(f"  t{tid:03d} {gt:6s}→{predicted:6s} {status} | {turn['fact_key'] or 'noop':25s} | {elapsed_call:.1f}s")

    f_ops.close()
    f_health.close()

    # ── QA pass ──────────────────────────────────────────────────────────
    print("\n\nRunning QA pass over injected memory...")
    qa_records = []
    f_qa = open(OUT_QA, "w")

    pairs = qa_pairs()
    for qi, (question, gold, _) in enumerate(pairs):
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
        print(f"  [{qi+1:2d}/{len(pairs)}] {status} f1={f1:.2f} ({qt:.1f}s)  Q: {question[:60]}")
        print(f"         gold: {gold!r}")
        print(f"         pred: {pred[:80]!r}")

    f_qa.close()

    # ── Summary ───────────────────────────────────────────────────────────
    acc = per_op_accuracy(op_records)
    curve = rot_curve(health_records)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Model: {MODEL}  |  Total time: {total_time/60:.1f} min")
    print(f"\nPer-op accuracy:")
    for op in ("ADD", "UPDATE", "DELETE", "NOOP"):
        a = acc[op]
        print(f"  {op:6s}  precision={a['precision']:.3f}  recall={a['recall']:.3f}"
              f"  (tp={a['tp']} fp={a['fp']} fn={a['fn']})")
    print(f"  Overall accuracy: {acc['overall_accuracy']:.3f}")
    print(f"\nRot curve:  slope={curve['slope']}  r2={curve['r2']}")
    final_health = health_records[-1] if health_records else {}
    print(f"Final store health:  entries={final_health.get('n_entries')}  "
          f"stale={final_health.get('stale_count')}  "
          f"dupes={final_health.get('duplicate_count')}  "
          f"rot_score={final_health.get('rot_score')}")
    qa_hit = sum(r["hit"] for r in qa_records)
    print(f"\nQA:  {qa_hit}/{len(qa_records)} EM  "
          f"mean_f1={sum(r['f1'] for r in qa_records)/len(qa_records):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
