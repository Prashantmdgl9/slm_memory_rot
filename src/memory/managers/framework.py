"""
Framework manager — the paper's proposed fix.

Principle: move compounding write-decisions out of the small model into
deterministic structure. The model does only extraction (easy); the
framework handles ADD/UPDATE/DELETE via typed slot matching (trivial code).

Two mechanisms:
  1. Typed slots  — each fact_key maps to exactly one store entry.
                    When a new value arrives for an existing key → UPDATE.
                    No model judgment needed.
  2. Rule-based supersession — deletion signals ("quit", "sold", etc.)
                    trigger DELETE without asking the model.
"""

from src.memory.managers.base import MemoryManager
from src.memory.store import MemoryStore
from src.models import ollama_client as llm

# Deletion signal phrases — if the extracted value contains any of these,
# treat it as a DELETE rather than UPDATE/ADD.
DELETION_SIGNALS = [
    "no longer", "stopped", "gave up", "sold", "quit",
    "cancelled", "passed away", "divorced", "ended", "deleted",
    "removed", "fired", "left", "moved away", "__deleted__",
]

EXTRACT_PROMPT = """\
Extract factual key-value pairs about the user from this conversational turn.

Turn: {content}

Output JSON only:
{{"facts": [{{"key": "city", "value": "Austin"}}, {{"key": "job_title", "value": "CTO"}}]}}

Rules:
- key: snake_case category (e.g. city, job_title, salary, relationship_status,
  pet, diet, car, home, hobby_guitar, education, language, gym, commute,
  sibling, parents_city, podcast, work_project, side_project, employer)
- value: the current fact stated in the turn (under 10 words)
- If the turn says something is no longer true (sold, quit, stopped,
  passed away, cancelled, gave up), include it with a value like
  "no longer has guitar" so the key can be detected
- If nothing memorable, output {{"facts": []}}
"""

QUERY_PROMPT = """\
Answer the question using the user's memory. Be concise.

Today's date: {date}
User's memory:
{memory}

Question: {question}
Answer with just the fact:"""


class FrameworkManager(MemoryManager):
    """
    Typed-slot framework.

    Internal state: self._slots = {fact_key: {"eid": str, "value": str}}
    The store mirrors _slots — one entry per fact_key at most.
    """

    def __init__(self, model):
        self.model  = model
        self.store  = MemoryStore()
        self._slots: dict = {}      # fact_key → {eid, value}
        self._buf_turns = []
        self._buf_date  = None

    # ── write-decision logic (deterministic) ─────────────────────────────

    def _apply_fact(self, key: str, value: str, date: str):
        key   = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not key or not value:
            return

        is_deletion = any(sig in value.lower() for sig in DELETION_SIGNALS)

        if is_deletion:
            if key in self._slots:
                self.store.delete(self._slots[key]["eid"])
                del self._slots[key]
        elif key in self._slots:
            # Key already exists → deterministic UPDATE
            self.store.update(self._slots[key]["eid"], f"{key}: {value}")
            self._slots[key]["value"] = value
        else:
            # New key → ADD
            eid = self.store.add(f"{key}: {value}", created_at=date)
            self._slots[key] = {"eid": eid, "value": value}

    # ── ingestion ─────────────────────────────────────────────────────────

    def _flush(self):
        if not self._buf_turns:
            return
        for content in self._buf_turns:
            prompt = EXTRACT_PROMPT.format(content=content[:400])
            result = llm.generate_json(self.model, prompt, think=False)
            for fact in (result or {}).get("facts", []):
                if isinstance(fact, dict):
                    self._apply_fact(
                        fact.get("key", ""),
                        fact.get("value", ""),
                        self._buf_date or "",
                    )
        self._buf_turns = []

    def ingest(self, role, content, date):
        if role != "user" or not content.strip():
            return
        self._buf_date = date
        self._buf_turns.append(content)

    def ingest_session_end(self):
        self._flush()

    def query(self, question, question_date):
        self._flush()
        hits = self.store.search(question, k=15)
        mem_text = "\n".join(f"- {h['content']}" for h in hits) or "(no memory)"
        prompt = QUERY_PROMPT.format(
            date=question_date, memory=mem_text, question=question
        )
        return llm.generate(self.model, prompt)

    def reset(self):
        self._buf_turns = []
        self._buf_date  = None
        self._slots     = {}
        self.store.reset()
