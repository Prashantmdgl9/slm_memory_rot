"""
Diagnostic metrics for the memory-rot paper.

Three layers:
  1. Per-op accuracy  — did the model fire the right op for each injection turn?
  2. Store health     — duplication, staleness, contradiction counts over time.
  3. Attribution      — for each wrong QA answer, which failure mode caused it?
"""

from collections import defaultdict


# ── 1. Per-op accuracy ────────────────────────────────────────────────────

def per_op_accuracy(records):
    """
    records: list of {"ground_truth_op": str, "predicted_op": str}
    Returns dict: op -> {tp, fp, fn, precision, recall, accuracy}
    """
    ops = ["ADD", "UPDATE", "DELETE", "NOOP"]
    counts = {op: {"tp": 0, "fp": 0, "fn": 0} for op in ops}
    for r in records:
        gt = r["ground_truth_op"]
        pred = r["predicted_op"]
        if gt == pred:
            counts[gt]["tp"] += 1
        else:
            counts[gt]["fn"] += 1
            counts[pred]["fp"] += 1

    result = {}
    for op in ops:
        tp = counts[op]["tp"]
        fp = counts[op]["fp"]
        fn = counts[op]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result[op] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
        }
    total = len(records)
    correct = sum(1 for r in records if r["ground_truth_op"] == r["predicted_op"])
    result["overall_accuracy"] = round(correct / total, 3) if total else 0.0
    return result


# ── 2. Store health ────────────────────────────────────────────────────────

def store_health_snapshot(store_entries, stream_up_to_turn, stream):
    """
    Compute a store-health snapshot after processing turns 0..turn_id.

    store_entries : list of (eid, {content, created_at}) from store.all_entries()
    stream_up_to_turn : index of last processed turn
    stream : full STREAM list from injection.py

    Returns dict with:
      - n_entries       : total entries in store
      - stale_count     : entries whose fact_key has since been overridden
      - duplicate_count : extra entries beyond one per fact_key
      - contradiction_pairs : pairs of entries for the same fact_key
    """
    # Build ground-truth "what should be in store right now"
    # (latest value per fact_key among all turns up to stream_up_to_turn)
    current_facts = {}  # fact_key -> fact_value (None means deleted)
    for turn in stream[:stream_up_to_turn + 1]:
        fk = turn["fact_key"]
        if fk is None:
            continue
        if turn["op"] in ("ADD", "UPDATE"):
            current_facts[fk] = turn["fact_value"]
        elif turn["op"] == "DELETE":
            current_facts[fk] = None  # should not be in store

    live_keys = {k for k, v in current_facts.items() if v is not None}
    dead_keys = {k for k, v in current_facts.items() if v is None}

    n_entries = len(store_entries)

    # We can't perfectly map store entries to fact_keys without an NLU pass,
    # so we use a keyword-overlap heuristic for health metrics.
    # This is approximate but sufficient for the rot-curve paper figures.
    def _matches_key(content, fact_key):
        keywords = {
            "job_title":           ["engineer", "cto", "vp", "officer", "job", "role", "title"],
            "city":                ["chicago", "seattle", "austin", "city", "moved", "live"],
            "relationship_status": ["single", "dating", "engaged", "married", "divorced", "separated"],
            "pet":                 ["cat", "dog", "luna", "max", "pet"],
            "hobby_guitar":        ["guitar"],
            "hobby_photography":   ["photography"],
            "hobby_running":       ["running"],
            "hobby_swimming":      ["swimming"],
            "salary":              ["salary", "comp", "000", "earn", "pay"],
            "diet":                ["vegetarian", "vegan", "omnivore", "diet", "meat"],
            "home":                ["apartment", "house", "studio", "rent", "home", "lincoln", "capitol", "austin"],
            "commute":             ["commute", "train", "minutes"],
            "gym":                 ["gym", "planet fitness", "crossfit", "membership"],
            "car":                 ["car", "civic", "tesla", "bmw", "drive"],
            "education":           ["ucla", "degree", "cs", "computer science", "studied"],
            "sibling":             ["sister", "emma"],
            "parents_city":        ["parents", "san francisco"],
            "language":            ["spanish", "japanese", "learning", "duolingo", "tutor"],
            "side_project":        ["app", "project", "side"],
            "work_project":        ["ml platform", "platform", "shipped"],
            "employer":            ["techcorp", "financestartu", "bigbank", "employer", "company"],
            "podcast":             ["podcast"],
        }
        kws = keywords.get(fact_key, [fact_key.split("_")])
        c = content.lower()
        return any(kw in c for kw in kws)

    # Count entries per fact_key (approximate)
    key_entry_count = defaultdict(int)
    for _, entry in store_entries:
        matched = False
        for fk in current_facts:
            if _matches_key(entry["content"], fk):
                key_entry_count[fk] += 1
                matched = True
                break

    stale_count = sum(
        key_entry_count.get(dk, 0) for dk in dead_keys
    )
    duplicate_count = sum(
        max(0, key_entry_count.get(lk, 0) - 1) for lk in live_keys
    )
    contradiction_pairs = sum(
        max(0, key_entry_count.get(lk, 0) - 1) for lk in live_keys
    )  # same as duplicates for active keys

    return {
        "turn_id":            stream_up_to_turn,
        "n_entries":          n_entries,
        "stale_count":        stale_count,
        "duplicate_count":    duplicate_count,
        "contradiction_pairs": contradiction_pairs,
        "rot_score":          stale_count + duplicate_count,
    }


# ── 3. Rot-curve summary ───────────────────────────────────────────────────

def rot_curve(snapshots):
    """
    Given a list of store_health_snapshot dicts (one per turn),
    compute the rot slope (rot_score per turn) via linear regression.
    Returns {slope, intercept, r2}.
    """
    if len(snapshots) < 2:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0}

    xs = [s["turn_id"] for s in snapshots]
    ys = [s["rot_score"] for s in snapshots]
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = y_mean - slope * x_mean
    y_pred = [slope * x + intercept for x in xs]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(ys, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return {"slope": round(slope, 4), "intercept": round(intercept, 4), "r2": round(r2, 4)}
