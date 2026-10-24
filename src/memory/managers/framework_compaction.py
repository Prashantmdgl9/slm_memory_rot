"""
Framework ablation: typed slots + deletion signals + periodic compaction.

Compaction runs every COMPACT_EVERY sessions using embedding-based dedup —
no LLM calls. Catches slot-key fragmentation where the same real-world fact
gets stored under slightly different key names (e.g. 'employer' vs 'company'),
producing two near-duplicate entries with cosine similarity > SIM_THRESHOLD.

After each compaction pass, _slots is pruned to remove any eids that were
deleted so the slot mapping stays consistent.
"""
from src.memory.managers.framework import FrameworkManager
from src.memory.compaction import compact

COMPACT_EVERY = 10   # sessions between compaction passes


class FrameworkCompactionManager(FrameworkManager):

    def __init__(self, model):
        super().__init__(model)
        self._session_count = 0

    def ingest_session_end(self):
        super().ingest_session_end()
        self._session_count += 1
        if self._session_count % COMPACT_EVERY == 0:
            n_deleted = compact(self.store)
            if n_deleted:
                # Remove from _slots any eid the compaction pass deleted
                live_eids = {eid for eid, _ in self.store.all_entries()}
                stale_keys = [k for k, v in self._slots.items()
                              if v["eid"] not in live_eids]
                for k in stale_keys:
                    del self._slots[k]

    def reset(self):
        super().reset()
        self._session_count = 0
