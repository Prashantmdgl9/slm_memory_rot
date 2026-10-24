"""
LoCoMo secondary benchmark (Maharana et al., ACL 2024).

Very long-term, two-speaker conversations (up to 32 sessions). We ingest each
conversation ONCE through a memory manager, then answer a stratified sample of
its questions. Mirrors experiments/longmemeval.py.

Usage:
  python experiments/locomo.py --condition naive     --model qwen3:4b
  python experiments/locomo.py --condition framework --model llama3.1:8b --qa-per-conv 20

Output:
  results/locomo_{condition}_{model_slug}.jsonl        — one record per question
  results/locomo_{condition}_{model_slug}_summary.txt

Categories: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop (answerable);
5 adversarial (no gold answer — skipped unless --include-adversarial).
"""
import sys, os, json, time, argparse, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.eval.metrics import exact_match, token_f1

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True,
                    choices=["naive", "framework", "rag", "softdel", "hf"],
                    help="Which memory manager to use")
parser.add_argument("--model", default="qwen3:4b",
                    help="Ollama model (ignored for --condition hf)")
parser.add_argument("--convs", type=int, default=10,
                    help="Number of conversations to evaluate (max 10)")
parser.add_argument("--qa-per-conv", type=int, default=20,
                    help="Questions per conversation (stratified by category)")
parser.add_argument("--include-adversarial", action="store_true",
                    help="Also score category-5 adversarial questions")
parser.add_argument("--session-timeout", type=float, default=90,
                    help="Per-session ingestion timeout in seconds (skip on exceeded)")
parser.add_argument("--max-sessions", type=int, default=0,
                    help="Cap sessions ingested per conversation (0 = all; use for fast smoke)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

# ── Manager selection ─────────────────────────────────────────────────────────
if args.condition == "naive":
    from src.memory.managers.naive import NaiveManager
    def make_manager(): return NaiveManager(args.model)
    model_label = args.model
elif args.condition == "framework":
    from src.memory.managers.framework import FrameworkManager
    def make_manager(): return FrameworkManager(args.model)
    model_label = args.model
elif args.condition == "rag":
    from src.memory.managers.rag_only import RagOnlyManager
    def make_manager(): return RagOnlyManager(args.model)
    model_label = args.model
elif args.condition == "softdel":
    from src.memory.managers.framework_bitemporal import FrameworkBitemporalManager
    def make_manager(): return FrameworkBitemporalManager(args.model)
    model_label = args.model
else:  # hf
    from src.memory.managers.framework_hf import HFFrameworkManager
    def make_manager(): return HFFrameworkManager()
    model_label = "hf-32b"

MODEL_SLUG = (args.condition + "_" + model_label).replace(":", "-").replace("/", "-")
OUT_PATH   = f"results/locomo_{MODEL_SLUG}.jsonl"
SUM_PATH   = f"results/locomo_{MODEL_SLUG}_summary.txt"
os.makedirs("results", exist_ok=True)

CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain",
             4: "single-hop", 5: "adversarial"}

# ── Data loading ──────────────────────────────────────────────────────────────
ALL_DATA = json.load(open("data/locomo10.json"))[:args.convs]


def session_order(conv):
    """Return session indices in numeric order (session_1, session_2, ...)."""
    idxs = sorted(int(m.group(1))
                  for k in conv
                  for m in [re.fullmatch(r"session_(\d+)", k)] if m)
    return idxs


def stratified_qa(qa, n, seed):
    """Sample n answerable QA pairs, round-robin across categories."""
    random.seed(seed)
    cats = [1, 2, 3, 4] + ([5] if args.include_adversarial else [])
    by_cat = {c: [q for q in qa if q.get("category") == c] for c in cats}
    for v in by_cat.values():
        random.shuffle(v)
    selected, i = [], 0
    while len(selected) < n and any(by_cat.values()):
        c = cats[i % len(cats)]
        if by_cat[c]:
            selected.append(by_cat[c].pop(0))
        i += 1
        if i > n * len(cats):
            break
    return selected[:n]


def gold_answer(q):
    """Category 5 has no 'answer' field — use adversarial_answer for reference."""
    return str(q.get("answer", q.get("adversarial_answer", "")))


def _eta(elapsed, done, total):
    if done == 0:
        return "?"
    rate = elapsed / done
    rem = int(rate * (total - done))
    return f"{rem//60}m{rem%60:02d}s"


def ingest_conversation(manager, conv, session_timeout):
    """Feed every turn (both speakers) once; return (sessions_ok, skipped, turns)."""
    ok = skipped = turns = 0
    sessions = session_order(conv)
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
    for si in sessions:
        date = conv.get(f"session_{si}_date_time", "")
        t0 = time.time()
        timed_out = False
        for turn in conv[f"session_{si}"]:
            if time.time() - t0 > session_timeout:
                timed_out = True
                print(f"    !! session {si} timeout (>{session_timeout}s) — skipped")
                break
            content = f"{turn['speaker']}: {turn['text']}"
            manager.ingest("user", content, date)
            turns += 1
        if timed_out:
            skipped += 1
        else:
            if hasattr(manager, "ingest_session_end"):
                manager.ingest_session_end()
            ok += 1
    return ok, skipped, turns


def latest_date(conv):
    idxs = session_order(conv)
    return conv.get(f"session_{idxs[-1]}_date_time", "") if idxs else ""


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"=== LoCoMo Benchmark ===")
print(f"Condition : {args.condition}  model: {model_label}")
print(f"Convs     : {len(ALL_DATA)}   qa/conv: {args.qa_per_conv}"
      f"   adversarial: {args.include_adversarial}")
print(f"Output    : {OUT_PATH}\n")

if args.dry_run:
    total = sum(len(stratified_qa(c["qa"], args.qa_per_conv, args.seed)) for c in ALL_DATA)
    print(f"Dry run — would evaluate {total} questions across {len(ALL_DATA)} convs.")
    sys.exit(0)

if args.condition == "hf":
    from src.models import hf_client as hf
    print(f"HF model  : {hf.active_model()}\n")

records  = []
t_global = time.time()
f_out    = open(OUT_PATH, "w")
q_done   = 0
q_total  = sum(len(stratified_qa(c["qa"], args.qa_per_conv, args.seed)) for c in ALL_DATA)

for ci, ex in enumerate(ALL_DATA):
    conv = ex["conversation"]
    sample_id = ex["sample_id"]
    qa_sample = stratified_qa(ex["qa"], args.qa_per_conv, args.seed)
    qdate = latest_date(conv)

    t_ing = time.time()
    manager = make_manager()
    sess_ok, sess_skip, n_turns = ingest_conversation(manager, conv, args.session_timeout)
    ing_s = time.time() - t_ing
    print(f"  [conv {ci+1}/{len(ALL_DATA)}] {sample_id}  ingested "
          f"{n_turns} turns / {sess_ok} sessions (skip {sess_skip}) in {ing_s:.0f}s")

    for q in qa_sample:
        t0   = time.time()
        pred = manager.query(q["question"], qdate)
        gold = gold_answer(q)
        em   = exact_match(gold, pred)
        f1   = token_f1(gold, pred)

        rec = {
            "sample_id":   sample_id,
            "question":    q["question"],
            "category":    q.get("category"),
            "gold":        gold,
            "pred":        pred,
            "em":          em,
            "f1":          round(f1, 3),
            "turns_ingested": n_turns,
            "time_s":      round(time.time() - t0, 1),
        }
        records.append(rec)
        f_out.write(json.dumps(rec) + "\n")
        f_out.flush()

        q_done += 1
        elapsed_total = time.time() - t_global
        status = "OK " if em else "MISS"
        print(
            f"    [{q_done:3d}/{q_total} | {100*q_done/q_total:5.1f}% "
            f"| ETA {_eta(elapsed_total, q_done, q_total)}]  "
            f"cat{q.get('category')} {CAT_NAMES.get(q.get('category','?'),''):11s} "
            f"{status} f1={f1:.2f}"
        )

f_out.close()

# ── Summary ───────────────────────────────────────────────────────────────────
total_em = sum(r["em"] for r in records)
mean_f1  = sum(r["f1"] for r in records) / len(records) if records else 0

from collections import defaultdict
by_cat = defaultdict(list)
for r in records:
    by_cat[r["category"]].append(r)

lines = [
    "=== LoCoMo Summary ===",
    f"Condition : {args.condition}  model: {model_label}",
    f"Questions : {len(records)}",
    f"EM        : {total_em}/{len(records)}  "
    f"({100*total_em/len(records):.1f}%)" if records else "EM: n/a",
    f"Mean F1   : {mean_f1:.3f}",
    "",
    f"{'Category':22s}  {'n':>4}  {'EM':>8}  {'F1':>6}",
    "-" * 48,
]
for c in sorted(by_cat):
    recs = by_cat[c]
    c_em = sum(r["em"] for r in recs)
    c_f1 = sum(r["f1"] for r in recs) / len(recs)
    lines.append(f"  {str(c)+' '+CAT_NAMES.get(c,''):20s}  {len(recs):4d}  "
                 f"{c_em:3d}/{len(recs):<3d}  {c_f1:.3f}")

for line in lines:
    print(line)

with open(SUM_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nSaved {OUT_PATH}")
print(f"Saved {SUM_PATH}")
