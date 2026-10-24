"""
C2 Mechanism Probe: routing failure vs comprehension failure.

For each UPDATE turn, we ask the SAME extraction/routing prompt under two
conditions, holding everything else constant:

  BLIND  : the store shown to the model does NOT contain the prior fact for
           this slot (3 unrelated distractor entries instead). The model has
           no way to know a prior value exists.
  PRIMED : the store shown contains the gold prior fact for this slot (with an
           id) plus the same 2 distractors.

If PRIMED yields high UPDATE firing and BLIND does not, the model CAN recognize
supersession when the prior fact is in context -> the naive failure is
state-tracking / retrieval (getting the prior fact into context), NOT
comprehension. That is the C2 claim. If PRIMED does not recover UPDATE, C2 is
false and the story must change.

Usage: python experiments/c2_probe.py [model ...]
Writes: results/c2_probe.jsonl  and prints a per-model summary.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.diagnostic.injection import load_stream
from src.models import ollama_client as llm

MODELS = [a for a in sys.argv[1:] if not a.startswith("-")] or [
    "qwen3:4b", "qwen3:8b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"]

STREAM = load_stream()

# Single-turn extract/routing prompt (mirrors naive.py EXTRACT_PROMPT).
PROMPT = """\
You are a memory manager. Maintain a concise, accurate memory store for a user.

Current memory:
{memory}

New turn [{date}] user: {content}

Extract facts about the user and decide memory operations. Output JSON only:
{{"operations": [{{"op": "ADD", "content": "<new fact>"}}, {{"op": "UPDATE", "id": "<id>", "content": "<corrected fact>"}}, {{"op": "DELETE", "id": "<id>"}}, {{"op": "NOOP"}}]}}

Rules:
- ADD only facts not already in memory
- UPDATE when new info supersedes an existing entry (use its exact id from current memory)
- DELETE when something is no longer true with no replacement (use its exact id)
- NOOP if nothing worth remembering (small talk, greetings, filler)
- Keep facts short and specific"""

DISTRACTORS = [
    ("d1", "2024-01-10", "has a younger sister named Emma"),
    ("d2", "2024-01-10", "studied computer science at UCLA"),
]

def gold_prior(update_turn):
    """Most recent earlier ADD/UPDATE with the same fact_key -> its value."""
    k = update_turn["fact_key"]
    prior = None
    for t in STREAM:
        if t["turn_id"] >= update_turn["turn_id"]:
            break
        if t["fact_key"] == k and t["op"] in ("ADD", "UPDATE") and t["fact_value"]:
            prior = t
    return prior

def mem_block(entries):
    return "\n".join(f"({i}) [{d}] {c}" for i, d, c in entries) or "(empty)"

def fired_update(result):
    ops = (result or {}).get("operations", [])
    return any(isinstance(o, dict) and o.get("op") == "UPDATE" for o in ops)

def run_model(model, probes):
    rows = []
    for p in probes:
        t = p["turn"]
        prior = p["prior"]
        primed_entries = [("m1", prior["date"], prior["fact_value"])] + DISTRACTORS
        blind_entries = DISTRACTORS + [("d3", "2024-01-11", "commutes 30 minutes by train")]
        out = {}
        for cond, entries in (("blind", blind_entries), ("primed", primed_entries)):
            prompt = PROMPT.format(memory=mem_block(entries), date=t["date"], content=t["content"])
            res = llm.generate_json(model, prompt, think=False)
            out[cond] = fired_update(res)
        row = {"model": model, "turn_id": t["turn_id"], "fact_key": t["fact_key"],
               "content": t["content"], "prior_value": prior["fact_value"],
               "blind_update": out["blind"], "primed_update": out["primed"]}
        rows.append(row)
        print(f"  t{t['turn_id']:03d} {t['fact_key']:20s} blind={'U' if out['blind'] else '.'} "
              f"primed={'U' if out['primed'] else '.'}")
    return rows

def main():
    # Build probe set: UPDATE turns that have a recoverable same-key prior.
    probes = []
    for t in STREAM:
        if t["op"] != "UPDATE":
            continue
        prior = gold_prior(t)
        if prior:
            probes.append({"turn": t, "prior": prior})
    print(f"C2 probe set: {len(probes)} UPDATE turns with recoverable prior "
          f"(of {sum(1 for t in STREAM if t['op']=='UPDATE')} total UPDATE turns)\n")

    os.makedirs("results", exist_ok=True)
    f = open("results/c2_probe.jsonl", "w")
    summary = {}
    for model in MODELS:
        print(f"=== {model} ===")
        rows = run_model(model, probes)
        for r in rows:
            f.write(json.dumps(r) + "\n"); f.flush()
        n = len(rows)
        blind = sum(r["blind_update"] for r in rows)
        primed = sum(r["primed_update"] for r in rows)
        summary[model] = (blind, primed, n)
        print(f"  -> blind UPDATE-fired {blind}/{n} ({blind/n:.0%}) | "
              f"primed {primed}/{n} ({primed/n:.0%})\n")
    f.close()

    print("=" * 56)
    print(f"{'model':16s} {'blind':>8s} {'primed':>8s} {'delta(pp)':>10s}")
    for m, (b, p, n) in summary.items():
        print(f"{m:16s} {b/n:>7.0%} {p/n:>8.0%} {100*(p-b)/n:>9.0f}")
    print("=" * 56)
    print("C2 holds if primed >> blind (prior-in-context recovers UPDATE routing).")

if __name__ == "__main__":
    main()
