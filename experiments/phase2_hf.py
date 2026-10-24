"""
Phase 2 injection diagnostic using the remote 32B reference model (HF Inference API).

Same 80-turn stream as phase2_injection.py — produces reference upper-bound numbers
for op-classification accuracy and store health on the capacity curve.

Usage:
  export HF_TOKEN=<your_token>
  python experiments/phase2_hf.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.diagnostic.injection import load_stream, qa_pairs, STREAM_STATS
from src.diagnostic.metrics   import per_op_accuracy, store_health_snapshot, rot_curve
from src.eval.metrics          import exact_match, token_f1
from src.memory.store          import MemoryStore
import src.models.hf_client as llm

MODEL_SLUG = "qwen3-32b-hf"
OUT_OPS    = f"results/phase2_{MODEL_SLUG}_ops.jsonl"
OUT_HEALTH = f"results/phase2_{MODEL_SLUG}_health.jsonl"
OUT_QA     = f"results/phase2_{MODEL_SLUG}_qa.jsonl"
os.makedirs("results", exist_ok=True)

STREAM  = load_stream()
N_TURNS = len(STREAM)
TOP_K   = 10
MAX_MEM = 30

EXTRACT_PROMPT = """\
You are a memory manager. Maintain a concise, accurate memory store for a user.

Current memory:
{memory}

New turn [{date}]: {content}

Extract facts about the user and decide memory operations. Output JSON only:
{{"operations": [{{"op": "ADD", "content": "<new fact>"}}, {{"op": "UPDATE", "id": "<id>", "content": "<corrected fact>"}}, {{"op": "DELETE", "id": "<id>"}}, {{"op": "NOOP"}}]}}

Rules:
- ADD only facts not already in memory
- UPDATE when new info supersedes an existing entry (use its exact id)
- DELETE when something is no longer true with no replacement
- NOOP if nothing worth remembering"""

QUERY_PROMPT = """\
Answer the question using the user's memory. Be concise.

Today's date: {date}
User's memory:
{memory}

Question: {question}
Answer with just the fact:"""


def mem_text(store):
    entries = store.all_entries()[-MAX_MEM:]
    return "\n".join(f"({eid}) [{e['created_at']}] {e['content']}" for eid, e in entries) or "(empty)"


def predicted_op(ops_fired):
    for prio in ("DELETE", "UPDATE", "ADD"):
        if any(o["op"] == prio for o in ops_fired):
            return prio
    return "NOOP"


def main():
    active = llm.active_model()
    print(f"=== Phase 2 Injection Diagnostic — Remote Reference ===")
    print(f"Model   : {active}  (slug: {MODEL_SLUG})")
    print(f"Turns   : {N_TURNS}  ({STREAM_STATS})")
    print(f"Outputs : {OUT_OPS}, {OUT_HEALTH}, {OUT_QA}\n")

    # Resume: reload completed turns and replay them into the store
    done_ids = set()
    op_records, health_records = [], []
    store = MemoryStore()

    if os.path.exists(OUT_OPS):
        with open(OUT_OPS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                op_records.append(rec)
                done_ids.add(rec["turn_id"])
                # Replay ops into store to restore its state
                for op in rec.get("ops_fired", []):
                    if op["op"] == "ADD":
                        store.add(op["content"], created_at=rec["date"])
                    elif op["op"] == "UPDATE":
                        store.update(op.get("id", ""), op.get("content", ""))
                    elif op["op"] == "DELETE":
                        store.delete(op.get("id", ""))

    if os.path.exists(OUT_HEALTH):
        with open(OUT_HEALTH) as f:
            for line in f:
                line = line.strip()
                if line:
                    health_records.append(json.loads(line))

    if done_ids:
        print(f"Resuming from turn {max(done_ids) + 1}  ({len(done_ids)} turns already done)\n")

    t_start = time.time()

    f_ops    = open(OUT_OPS,    "a")
    f_health = open(OUT_HEALTH, "a")

    for turn in STREAM:
        tid, gt = turn["turn_id"], turn["op"]
        if tid in done_ids:
            continue
        prompt  = EXTRACT_PROMPT.format(memory=mem_text(store), date=turn["date"], content=turn["content"])

        t0     = time.time()
        result = llm.generate_json(prompt, max_tokens=400)
        elapsed = time.time() - t0

        ops_fired = []
        for op in (result or {}).get("operations", []):
            if not isinstance(op, dict):
                continue
            ot = op.get("op", "NOOP")
            if ot == "ADD":
                c = str(op.get("content", "")).strip()
                if c:
                    store.add(c, created_at=turn["date"])
                    ops_fired.append({"op": "ADD", "content": c})
            elif ot == "UPDATE":
                store.update(op.get("id", ""), str(op.get("content", "")))
                ops_fired.append({"op": "UPDATE", "content": op.get("content", "")[:80]})
            elif ot == "DELETE":
                store.delete(op.get("id", ""))
                ops_fired.append({"op": "DELETE", "content": op.get("id", "")})

        pred    = predicted_op(ops_fired)
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
        pct    = 100 * (tid + 1) / N_TURNS
        print(f"  t{tid:03d} {gt:6s}→{pred:6s} {status} | {turn['fact_key'] or 'noop':25s} | {elapsed:.1f}s  [{pct:.0f}%]")

    f_ops.close(); f_health.close()

    # QA pass
    print("\nRunning QA pass...")
    f_qa = open(OUT_QA, "w")
    qa_records = []
    for qi, (question, gold, _) in enumerate(qa_pairs()):
        hits = store.search(question, k=TOP_K)
        mem  = "\n".join(f"- {h['content']}" for h in hits) or "(empty)"
        t0   = time.time()
        pred = llm.generate(QUERY_PROMPT.format(date="2025-05-15", memory=mem, question=question), max_tokens=200)
        qt   = time.time() - t0
        hit  = exact_match(gold, pred)
        f1   = token_f1(gold, pred)
        rec  = {"qi": qi, "question": question, "gold": gold, "pred": pred,
                "hit": hit, "f1": f1, "query_time_s": round(qt, 2)}
        qa_records.append(rec)
        f_qa.write(json.dumps(rec) + "\n"); f_qa.flush()
        print(f"  [{qi+1}/{len(qa_pairs())}] {'OK' if hit else 'MISS'} f1={f1:.2f}  Q: {question[:60]}")
    f_qa.close()

    acc   = per_op_accuracy(op_records)
    curve = rot_curve(health_records)
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Model: {active}  |  Total: {total_time/60:.1f} min")
    print(f"\nPer-op accuracy:")
    for op in ("ADD", "UPDATE", "DELETE", "NOOP"):
        a = acc[op]
        print(f"  {op:6s}  precision={a['precision']:.3f}  recall={a['recall']:.3f}")
    print(f"  Overall: {acc['overall_accuracy']:.3f}")
    final = health_records[-1] if health_records else {}
    print(f"Final store:  entries={final.get('n_entries')}  stale={final.get('stale_count')}  rot={final.get('rot_score')}")
    qa_hit = sum(r["hit"] for r in qa_records)
    print(f"QA:  {qa_hit}/{len(qa_records)} EM  mean_f1={sum(r['f1'] for r in qa_records)/len(qa_records):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
