"""
THROWAWAY SPIKE — do not keep. Proves (or disproves) whether a small on-device
model can adjudicate "same attribute?" for the ambiguous embedding band that
pure threshold binding cannot separate (Candidate C's in-band LLM call).

Run: python experiments/spike_adjudicator.py [model]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.embedder import embed
from src.models import ollama_client as llm

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:3b"
LO, HI = 0.70, 0.87   # the ambiguous band where the adjudicator fires

# ground truth: should a new value for one overwrite the other?
PAIRS = [
    # SAME attribute (true synonyms)
    ("employer", "company", True),
    ("job_title", "role", True),
    ("employer", "workplace", True),
    ("relationship_status", "marital_status", True),
    ("car", "vehicle", True),
    ("home_city", "hometown", True),
    ("employer", "where i work", True),
    ("salary", "income", True),
    ("diet", "dietary_preference", True),
    ("phone", "mobile_number", True),
    # DIFFERENT attribute (distinct, often share a head noun)
    ("home_city", "work_city", False),
    ("work_project", "side_project", False),
    ("home_city", "parents_city", False),
    ("hobby_guitar", "hobby_piano", False),
    ("sibling", "parent", False),
    ("employer", "job_title", False),
    ("city", "country", False),
    ("mother", "father", False),
    ("home_address", "work_address", False),
    ("pet_dog", "pet_cat", False),
]

PROMPT = """Two fields in a person's memory: A and B.

The TEST: could one single person ever have DIFFERENT values for A and B at the
same time? If yes, they are DIFFERENT attributes. If their value is always
identical for everyone, they are the SAME attribute.

Examples:
- "employer" vs "company" -> SAME (always the same value)
- "job title" vs "role" -> SAME
- "home city" vs "work city" -> DIFFERENT (can live and work in different cities)
- "mother" vs "father" -> DIFFERENT (two different people)
- "home address" vs "work address" -> DIFFERENT

Field A: "{a}"
Field B: "{b}"

First answer in one short clause: could a person have different values for A and B?
Then on the last line write exactly one word: SAME or DIFFERENT."""


def cos(a, b):
    ea, eb = embed([a.replace("_", " ")])[0], embed([b.replace("_", " ")])[0]
    return float(ea @ eb)


def adjudicate(a, b):
    out = (llm.generate(MODEL, PROMPT.format(a=a.replace("_", " "),
                                             b=b.replace("_", " "))) or "").lower()
    # parse the LAST verdict token (model reasons first, answers last)
    i_s, i_d = out.rfind("same"), out.rfind("different")
    if i_s == -1 and i_d == -1:
        return None
    return i_s > i_d


print(f"=== Adjudicator spike — model: {MODEL} ===")
print(f"Band of interest: {LO}–{HI} (where pure threshold fails)\n")
print(f"{'A':20s} {'B':16s} {'cos':>5s} {'band':>5s} {'gold':>5s} {'pred':>5s}  res")
print("-" * 70)

in_band = []
all_rows = []
for a, b, same in PAIRS:
    c = cos(a, b)
    band = LO <= c <= HI
    pred = adjudicate(a, b)
    ok = (pred == same) if pred is not None else False
    mark = "OK " if ok else ("??" if pred is None else "XX")
    row = (a, b, c, band, same, pred, ok)
    all_rows.append(row)
    if band:
        in_band.append(row)
    print(f"{a:20s} {b:16s} {c:5.2f} {'IN' if band else '  ':>5s} "
          f"{'SAME' if same else 'DIFF':>5s} "
          f"{('SAME' if pred else 'DIFF') if pred is not None else '????':>5s}  {mark}")

def acc(rows):
    scored = [r for r in rows if r[5] is not None]
    if not scored:
        return 0.0, 0, 0
    good = sum(1 for r in scored if r[5] == r[4])
    return good / len(scored), good, len(scored)

a_all, g_all, n_all = acc(all_rows)
a_band, g_band, n_band = acc(in_band)

print("\n--- RESULTS ---")
print(f"Overall:  {g_all}/{n_all}  ({a_all:.0%})")
print(f"IN-BAND:  {g_band}/{n_band}  ({a_band:.0%})   <-- the cases that matter")
# the two canonical breakers
for a, b, same in [("employer", "company", True), ("home_city", "work_city", False)]:
    r = next(r for r in all_rows if r[0] == a and r[1] == b)
    print(f"  critical {a}/{b}: gold={'SAME' if same else 'DIFF'} "
          f"pred={('SAME' if r[5] else 'DIFF') if r[5] is not None else '????'} "
          f"{'OK' if r[6] else 'FAIL'}")
