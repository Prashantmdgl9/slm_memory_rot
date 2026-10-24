import json, os, time
from src.eval.metrics import exact_match, token_f1

def _load_checkpoint(output_path):
    """Return set of already-completed indices from an existing output file."""
    done = {}
    if output_path and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done[r["idx"]] = r
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done

def run_longmemeval(manager, data_path, n=None, output_path=None, verbose=True):
    data = json.load(open(data_path))
    if n:
        data = data[:n]

    # Resume: load already-completed records; skip their indices
    done = _load_checkpoint(output_path)
    if done:
        print(f"Resuming: {len(done)}/{len(data)} already done, skipping indices {sorted(done)}")

    out_f = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_f = open(output_path, "a")  # append so completed records are preserved

    records = list(done.values())

    try:
        for i, ex in enumerate(data):
            if i in done:
                continue

            manager.reset()

            t_ingest = time.time()
            for sess, date in zip(ex["haystack_sessions"], ex["haystack_dates"]):
                for turn in sess:
                    manager.ingest(turn["role"], turn["content"], date)
                if hasattr(manager, "ingest_session_end"):
                    manager.ingest_session_end()
            ingest_time = time.time() - t_ingest

            t_query = time.time()
            pred = manager.query(ex["question"], ex["question_date"])
            query_time = time.time() - t_query

            gold = str(ex["answer"])
            hit = exact_match(gold, pred)
            f1 = token_f1(gold, pred)

            rec = {
                "idx": i,
                "question_type": ex.get("question_type", ""),
                "question": ex["question"],
                "gold": gold,
                "pred": pred,
                "hit": hit,
                "f1": f1,
                "ingest_time_s": round(ingest_time, 1),
                "query_time_s": round(query_time, 1),
            }
            records.append(rec)

            # Checkpoint immediately
            if out_f:
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

            if verbose:
                print(f"[{i+1}/{len(data)}] {rec['question_type']:30s} {'OK ' if hit else 'MISS'} "
                      f"f1={f1:.2f} ({query_time:.1f}s query, {ingest_time:.1f}s ingest)")
                print(f"    gold: {gold!r}")
                print(f"    pred: {pred[:120]!r}")
    finally:
        if out_f:
            out_f.close()

    total = len(records)
    correct = sum(r["hit"] for r in records)
    mean_f1 = sum(r["f1"] for r in records) / total if total else 0

    by_type = {}
    for r in records:
        qt = r["question_type"]
        by_type.setdefault(qt, []).append(r["hit"])

    print(f"\n=== {correct}/{total} EM ({correct/total:.1%}), mean F1={mean_f1:.3f} ===")
    for qt, hits in sorted(by_type.items()):
        print(f"  {qt:30s}: {sum(hits)}/{len(hits)} ({sum(hits)/len(hits):.1%})")

    return records
