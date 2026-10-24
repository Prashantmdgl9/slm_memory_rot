"""
LongMemEval secondary validation.

Feeds LongMemEval_S sessions through a memory manager, then answers the
benchmark question. Compares naive / framework (on-device) / hf-framework (32B).

Usage:
  python experiments/longmemeval.py --condition naive      --model qwen3:4b
  python experiments/longmemeval.py --condition framework  --model qwen3:4b
  python experiments/longmemeval.py --condition hf                           # needs HF_TOKEN
  python experiments/longmemeval.py --condition naive      --model qwen3:4b --n 20
  python experiments/longmemeval.py --condition framework  --model llama3.1:8b --n 20

Output:
  results/longmemeval_{condition}_{model_slug}.jsonl   — one record per example
  results/longmemeval_{condition}_{model_slug}_summary.txt

Progress line:
  [3/20 | 15% | avg 48s | ETA 14m20s]  e47becba single-session-user  OK  f1=0.85 | 42s
"""
import sys, os, json, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.eval.metrics import exact_match, token_f1

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True,
                    choices=["naive", "framework", "hf"],
                    help="Which memory manager to use")
parser.add_argument("--model", default="qwen3:4b",
                    help="Ollama model (ignored for --condition hf)")
parser.add_argument("--n", type=int, default=20,
                    help="Number of examples to evaluate (stratified by question type)")
parser.add_argument("--session-timeout", type=float, default=90,
                    help="Per-session ingestion timeout in seconds (skip on exceeded)")
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
else:  # hf
    from src.memory.managers.framework_hf import HFFrameworkManager
    def make_manager(): return HFFrameworkManager()
    model_label = "hf-32b"

MODEL_SLUG = (args.condition + "_" + model_label).replace(":", "-").replace("/", "-")
OUT_PATH   = f"results/longmemeval_{MODEL_SLUG}.jsonl"
SUM_PATH   = f"results/longmemeval_{MODEL_SLUG}_summary.txt"
os.makedirs("results", exist_ok=True)

# ── Data loading & stratified sample ─────────────────────────────────────────
ALL_DATA = json.load(open("data/longmemeval_s.json"))

QTYPES = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

def stratified_sample(data, n, seed):
    random.seed(seed)
    by_type = {qt: [] for qt in QTYPES}
    for ex in data:
        qt = ex["question_type"]
        if qt in by_type:
            by_type[qt].append(ex)
    for v in by_type.values():
        random.shuffle(v)

    # Round-robin fill
    selected = []
    i = 0
    while len(selected) < n:
        qt = QTYPES[i % len(QTYPES)]
        if by_type[qt]:
            selected.append(by_type[qt].pop(0))
        i += 1
        if i > n * len(QTYPES):
            break  # safety
    return selected[:n]

EXAMPLES = stratified_sample(ALL_DATA, args.n, args.seed)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _eta(elapsed, done, total):
    if done == 0:
        return "?"
    rate = elapsed / done
    rem  = int(rate * (total - done))
    return f"{rem//60}m{rem%60:02d}s"


def ingest_example(manager, example, session_timeout):
    """Feed all sessions; return (n_sessions_ok, n_sessions_skipped, total_turns)."""
    ok = skipped = turns = 0
    sessions   = example["haystack_sessions"]
    dates      = example["haystack_dates"]

    for sess_idx, (session, date) in enumerate(zip(sessions, dates)):
        t0 = time.time()
        timed_out = False
        for turn in session:
            if time.time() - t0 > session_timeout:
                timed_out = True
                print(f"    !! session {sess_idx} timeout (>{session_timeout}s) — skipped")
                break
            if turn["role"] == "user":
                manager.ingest("user", turn["content"], date)
                turns += 1
        if timed_out:
            skipped += 1
        else:
            manager.ingest_session_end()
            ok += 1

    return ok, skipped, turns


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"=== LongMemEval Validation ===")
print(f"Condition : {args.condition}  model: {model_label}")
print(f"Examples  : {args.n}  (stratified across {len(QTYPES)} question types)")
print(f"Sess timeout: {args.session_timeout}s")
print(f"Output    : {OUT_PATH}\n")

if args.dry_run:
    print("Dry run — exiting.")
    sys.exit(0)

# Check HF token early
if args.condition == "hf":
    from src.models import hf_client as hf
    print(f"HF model  : {hf.active_model()}\n")

records   = []
t_global  = time.time()

f_out = open(OUT_PATH, "w")

for i, ex in enumerate(EXAMPLES):
    qid   = ex["question_id"]
    qtype = ex["question_type"]
    t0    = time.time()

    manager = make_manager()
    sess_ok, sess_skip, n_turns = ingest_example(manager, ex, args.session_timeout)

    pred = manager.query(ex["question"], ex["question_date"])
    gold = ex["answer"]

    em = exact_match(gold, pred)
    f1 = token_f1(gold, pred)

    elapsed = time.time() - t0
    elapsed_total = time.time() - t_global

    rec = {
        "question_id":      qid,
        "question_type":    qtype,
        "question":         ex["question"],
        "gold":             gold,
        "pred":             pred,
        "em":               em,
        "f1":               round(f1, 3),
        "sessions_total":   len(ex["haystack_sessions"]),
        "sessions_ok":      sess_ok,
        "sessions_skipped": sess_skip,
        "turns_ingested":   n_turns,
        "time_s":           round(elapsed, 1),
    }
    records.append(rec)
    f_out.write(json.dumps(rec) + "\n")
    f_out.flush()

    status = "OK " if em else "MISS"
    eta    = _eta(elapsed_total, i + 1, args.n)
    avg    = elapsed_total / (i + 1)
    pct    = 100 * (i + 1) / args.n

    print(
        f"  [{i+1:3d}/{args.n} | {pct:5.1f}% | avg {avg:.0f}s | ETA {eta}]"
        f"  {qid[:8]} {qtype:28s} {status} f1={f1:.2f}"
        f" | turns={n_turns} skip={sess_skip} | {elapsed:.0f}s"
    )
    print(f"    gold: {gold!r}")
    print(f"    pred: {pred[:100]!r}")

f_out.close()

# ── Summary ───────────────────────────────────────────────────────────────────
total_em = sum(r["em"] for r in records)
mean_f1  = sum(r["f1"] for r in records) / len(records) if records else 0

# Per-type breakdown
from collections import defaultdict
by_type = defaultdict(list)
for r in records:
    by_type[r["question_type"]].append(r)

lines = []
lines.append(f"=== LongMemEval Summary ===")
lines.append(f"Condition : {args.condition}  model: {model_label}")
lines.append(f"Examples  : {len(records)}")
lines.append(f"EM        : {total_em}/{len(records)}  ({100*total_em/len(records):.1f}%)")
lines.append(f"Mean F1   : {mean_f1:.3f}")
lines.append(f"")
lines.append(f"{'Question Type':30s}  {'n':>4}  {'EM':>6}  {'F1':>6}")
lines.append("-" * 55)
for qt in QTYPES:
    recs = by_type.get(qt, [])
    if not recs:
        continue
    qt_em = sum(r["em"] for r in recs)
    qt_f1 = sum(r["f1"] for r in recs) / len(recs)
    lines.append(f"  {qt:28s}  {len(recs):4d}  {qt_em:3d}/{len(recs):<3d}  {qt_f1:.3f}")

for line in lines:
    print(line)

with open(SUM_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nSaved {OUT_PATH}")
print(f"Saved {SUM_PATH}")
