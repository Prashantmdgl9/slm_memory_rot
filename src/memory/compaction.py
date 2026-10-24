"""
Periodic compaction — deterministic dedup over the memory store.

Algorithm (no LLM calls needed — embeddings already stored):
  1. Compute pairwise cosine similarity over all entries (already normalised).
  2. For every pair with similarity > SIM_THRESHOLD, keep the entry whose
     created_at date is more recent and delete the other.
  3. Repeat until no pairs exceed the threshold (handles chains).

Cheap: O(n²) in #entries, but n is typically < 100 and no Ollama calls.
"""

import numpy as np

SIM_THRESHOLD = 0.88   # conservative — avoids false merges


def compact(store) -> int:
    """
    Run one compaction pass on `store`.
    Returns the number of entries deleted.
    """
    deleted = 0
    changed = True

    while changed:
        changed = False
        entries = store.all_entries()           # [(eid, {content, embedding, created_at})]
        if len(entries) < 2:
            break

        eids = [e[0] for e in entries]
        vecs = np.stack([e[1]["embedding"] for e in entries])   # already L2-normalised
        sim  = vecs @ vecs.T                                     # cosine similarity matrix

        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                if sim[i, j] < SIM_THRESHOLD:
                    continue
                # Near-duplicate — keep the newer entry
                ei, ej   = entries[i], entries[j]
                keep_idx = i if ei[1]["created_at"] >= ej[1]["created_at"] else j
                drop_idx = j if keep_idx == i else i
                drop_eid = eids[drop_idx]
                store.delete(drop_eid)
                deleted += 1
                changed  = True
                break       # entries list is stale; restart inner loop
            if changed:
                break

    return deleted
