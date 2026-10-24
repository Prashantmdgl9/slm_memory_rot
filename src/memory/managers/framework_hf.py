"""
Framework manager wired to the remote HF reference model (Qwen3-32B).

Same typed-slot logic as framework.py — only the LLM call changes.
"""
from src.memory.managers.framework import EXTRACT_PROMPT, QUERY_PROMPT, DELETION_SIGNALS
from src.memory.managers.base import MemoryManager
from src.memory.store import MemoryStore
from src.models import hf_client as hf


class HFFrameworkManager(MemoryManager):
    def __init__(self):
        self.store  = MemoryStore()
        self._slots = {}
        self._buf_turns = []
        self._buf_date  = None

    def _apply_fact(self, key, value, date):
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
            self.store.update(self._slots[key]["eid"], f"{key}: {value}")
            self._slots[key]["value"] = value
        else:
            eid = self.store.add(f"{key}: {value}", created_at=date)
            self._slots[key] = {"eid": eid, "value": value}

    def _flush(self):
        if not self._buf_turns:
            return
        for content in self._buf_turns:
            prompt = EXTRACT_PROMPT.format(content=content[:400])
            result = hf.generate_json(prompt)
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
        return hf.generate(prompt)

    def reset(self):
        self._buf_turns = []
        self._buf_date  = None
        self._slots     = {}
        self.store.reset()
