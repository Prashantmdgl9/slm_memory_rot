"""
Framework + bitemporal soft-invalidation (Candidate B).

DELETE becomes non-destructive: a soft-deleted fact is retained and recoverable
instead of erased, so false-positive deletions no longer permanently lose data
(cf. paper §5.3, where erasing correct facts dropped QA 64%->50%).

Write-decision logic is inherited unchanged from FrameworkManager — the parent's
deletion branch already calls store.delete(), which BitemporalStore makes soft.
Only retrieval changes: when the valid store cannot answer a query, fall back to
soft-deleted entries as a lower-priority tier.
"""
from src.memory.managers.framework import FrameworkManager, QUERY_PROMPT
from src.memory.store import BitemporalStore
from src.models import ollama_client as llm


class FrameworkBitemporalManager(FrameworkManager):

    FALLBACK_TAU = 0.5   # below this top valid score, consult recoverable entries

    def __init__(self, model):
        super().__init__(model)
        self.store = BitemporalStore()

    def _retrieve(self, question, k=15):
        """Return (valid_hits, recovered_hits). Recovered (soft-deleted) entries
        are consulted only when the valid store answers weakly."""
        valid = self.store.search(question, k=k)
        best = valid[0]["score"] if valid else 0.0
        if best < self.FALLBACK_TAU:
            recovered = [h for h in self.store.search(question, k=5,
                                                      include_invalid=True)
                         if not h.get("valid", True)]
            return valid, recovered
        return valid, []

    def query(self, question, question_date):
        self._flush()
        valid, recovered = self._retrieve(question)
        mem = "\n".join(f"- {h['content']}" for h in valid) or "(no memory)"
        if recovered:
            mem += "\n(possibly outdated:)\n" + "\n".join(
                f"- {h['content']}" for h in recovered)
        prompt = QUERY_PROMPT.format(
            date=question_date, memory=mem, question=question)
        return llm.generate(self.model, prompt)
