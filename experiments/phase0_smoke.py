"""Phase 0 de-risk spike: prove the pipeline runs end-to-end on the laptop.

Ingest a few LongMemEval oracle questions -> embed turns -> retrieve top-k ->
answer with a local Ollama model -> check correctness.

Success criterion (RESEARCH_PLAN Phase 0): answer >= 1 question correctly.
This is a throwaway spike; the real harness is built in Phase 1 under src/.
"""
import json, re, sys, time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

OLLAMA = "http://localhost:11434/api/generate"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
N_Q = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TOP_K = 10

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def ollama(prompt):
    r = requests.post(OLLAMA, json={"model": MODEL, "prompt": prompt, "stream": False,
                                    "options": {"temperature": 0}}, timeout=180)
    return r.json()["response"].strip()

def main():
    data = json.load(open("data/longmemeval_oracle.json"))
    # pick the first N questions with reasonably short answers (better for auto exact-match)
    qs = [e for e in data if len(str(e["answer"])) <= 60][:N_Q]
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
    Q_INSTR = "Represent this sentence for searching relevant passages: "

    correct = 0
    for i, e in enumerate(qs):
        # flatten all turns of all haystack sessions into memory units
        turns = []
        for sess, date in zip(e["haystack_sessions"], e["haystack_dates"]):
            for t in sess:
                turns.append(f"[{date}] {t['role']}: {t['content']}")
        vecs = emb.encode(turns, normalize_embeddings=True, show_progress_bar=False)
        qv = emb.encode([Q_INSTR + e["question"]], normalize_embeddings=True)[0]
        scores = vecs @ qv
        top = np.argsort(-scores)[:TOP_K]
        context = "\n".join(turns[j] for j in sorted(top))

        prompt = (
            "You are answering a question using the user's past conversation history.\n"
            f"Today's date: {e['question_date']}\n\n"
            f"Relevant history:\n{context}\n\n"
            f"Question: {e['question']}\n"
            "Answer concisely with just the fact:"
        )
        t0 = time.time()
        ans = ollama(prompt)
        gold = str(e["answer"])
        hit = norm(gold) in norm(ans) or norm(ans) in norm(gold)
        correct += hit
        print(f"[{i+1}/{len(qs)}] {e['question_type']:26s} {'OK ' if hit else 'MISS'} "
              f"({time.time()-t0:.1f}s)")
        print(f"    Q:    {e['question']}")
        print(f"    gold: {gold!r}")
        print(f"    pred: {ans[:120]!r}")

    print(f"\n=== Phase 0 smoke: {correct}/{len(qs)} correct with {MODEL} "
          f"(criterion: >=1) -> {'PASS' if correct >= 1 else 'FAIL'} ===")

if __name__ == "__main__":
    main()
