import uuid
import numpy as np
from src.models.embedder import embed

class MemoryStore:
    def __init__(self):
        self._entries = {}  # id -> {content, embedding, created_at}
        self.op_log = []   # [{op, eid, content}] — cleared per turn by the runner

    def reset(self):
        self._entries = {}
        self.op_log = []

    def add(self, content, created_at=""):
        eid = str(uuid.uuid4())[:8]
        emb = embed([content])[0]
        self._entries[eid] = {"content": content, "embedding": emb, "created_at": created_at}
        self.op_log.append({"op": "ADD", "eid": eid, "content": content})
        return eid

    def update(self, eid, content):
        if eid not in self._entries:
            return
        self._entries[eid]["content"] = content
        self._entries[eid]["embedding"] = embed([content])[0]
        self.op_log.append({"op": "UPDATE", "eid": eid, "content": content})

    def delete(self, eid):
        content = self._entries.get(eid, {}).get("content", "")
        self._entries.pop(eid, None)
        self.op_log.append({"op": "DELETE", "eid": eid, "content": content})

    def search(self, query_text, k=10):
        if not self._entries:
            return []
        qvec = embed([query_text], is_query=True)[0]
        ids = list(self._entries.keys())
        vecs = np.stack([self._entries[i]["embedding"] for i in ids])
        scores = vecs @ qvec
        top_k = min(k, len(ids))
        top_idx = np.argsort(-scores)[:top_k]
        return [{"id": ids[i], **self._entries[ids[i]]} for i in top_idx]

    def all_entries(self):
        return list(self._entries.items())

    def __len__(self):
        return len(self._entries)


class BitemporalStore(MemoryStore):
    """Soft-invalidation store: DELETE marks an entry invalid instead of
    removing it, so wrongly-deleted facts stay recoverable. The *effective*
    store (len, all_entries, default search) exposes only valid entries."""

    def add(self, content, created_at=""):
        eid = super().add(content, created_at)
        self._entries[eid]["valid"] = True
        self._entries[eid]["invalidated_at"] = None
        return eid

    def delete(self, eid):
        if eid not in self._entries:
            return
        self._entries[eid]["valid"] = False
        self.op_log.append({"op": "DELETE", "eid": eid,
                            "content": self._entries[eid]["content"]})

    def _valid_ids(self):
        return [i for i, e in self._entries.items() if e.get("valid", True)]

    def _invalid_ids(self):
        return [i for i, e in self._entries.items() if not e.get("valid", True)]

    def all_entries(self):
        return [(i, self._entries[i]) for i in self._valid_ids()]

    def recoverable(self):
        """Soft-deleted entries, retained and recoverable."""
        return [(i, self._entries[i]) for i in self._invalid_ids()]

    def __len__(self):
        return len(self._valid_ids())

    def _rank(self, ids, qvec, k):
        if not ids:
            return []
        vecs = np.stack([self._entries[i]["embedding"] for i in ids])
        scores = vecs @ qvec
        order = np.argsort(-scores)[:min(k, len(ids))]
        return [{"id": ids[i], "score": float(scores[i]), **self._entries[ids[i]]}
                for i in order]

    def search(self, query_text, k=10, include_invalid=False):
        qvec = embed([query_text], is_query=True)[0]
        hits = self._rank(self._valid_ids(), qvec, k)
        if include_invalid:
            # invalid entries form a strictly lower tier, ranked after all valid
            hits += self._rank(self._invalid_ids(), qvec, k)
        return hits
