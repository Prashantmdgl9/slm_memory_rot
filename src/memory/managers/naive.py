from src.memory.managers.base import MemoryManager
from src.memory.store import MemoryStore
from src.models import ollama_client as llm

TOP_K = 10
MAX_MEM_SHOWN = 30
BATCH_SIZE = 50  # user turns per LLM call

# JSON-schema-constrained output: forces every model into the operations
# structure, isolating routing ability from schema-following ability.
OPERATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE", "NOOP"]},
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["op"],
            },
        },
    },
    "required": ["operations"],
}

EXTRACT_PROMPT = """\
You are a memory manager. Maintain a concise, accurate memory store for a user.

Current memory:
{memory}

New turn [{date}] {role}: {content}

Extract facts about the user and decide memory operations. Output JSON only:
{{"operations": [{{"op": "ADD", "content": "<new fact>"}}, {{"op": "UPDATE", "id": "<id>", "content": "<corrected fact>"}}, {{"op": "DELETE", "id": "<id>"}}, {{"op": "NOOP"}}]}}

Rules:
- ADD only facts not already in memory
- UPDATE when new info supersedes an existing entry (use its exact id from current memory)
- DELETE when something is no longer true with no replacement (use its exact id)
- NOOP if nothing worth remembering (small talk, greetings, filler)
- Keep facts short and specific"""

BATCH_EXTRACT_PROMPT = """\
You are a memory manager. Maintain a concise, accurate memory store for a user.

Current memory:
{memory}

New user turns from [{date}]:
{turns}

Extract facts about the user from ALL turns above and decide memory operations. Output JSON only:
{{"operations": [{{"op": "ADD", "content": "<new fact>"}}, {{"op": "UPDATE", "id": "<id>", "content": "<corrected fact>"}}, {{"op": "DELETE", "id": "<id>"}}, {{"op": "NOOP"}}]}}

Rules:
- ADD only facts not already in memory
- UPDATE when new info supersedes an existing entry (use its exact id from current memory)
- DELETE when something is no longer true with no replacement (use its exact id)
- NOOP if nothing worth remembering across all turns
- Keep facts short and specific"""

QUERY_PROMPT = """\
Answer the question using the user's memory. Be concise.

Today's date: {date}
User's memory:
{memory}

Question: {question}
Answer with just the fact:"""


class NaiveManager(MemoryManager):
    def __init__(self, model):
        self.model = model
        self.store = MemoryStore()
        self._buf_turns = []   # buffered (content, date) user turns
        self._buf_date = None

    def _mem_text(self):
        entries = self.store.all_entries()[-MAX_MEM_SHOWN:]
        return "\n".join(
            f"({eid}) [{e['created_at']}] {e['content']}"
            for eid, e in entries
        ) or "(empty)"

    def _apply_ops(self, result):
        for op in (result or {}).get("operations", []):
            if not isinstance(op, dict):
                continue
            op_type = op.get("op", "NOOP")
            if op_type == "ADD":
                c = str(op.get("content", "") or "").strip()
                if c:
                    self.store.add(c, created_at=self._buf_date)
            elif op_type == "UPDATE":
                self.store.update(op.get("id", ""), str(op.get("content", "") or ""))
            elif op_type == "DELETE":
                self.store.delete(op.get("id", ""))

    def _flush(self):
        if not self._buf_turns:
            return
        turns_text = "\n".join(f"- {c[:400]}" for c in self._buf_turns)
        prompt = BATCH_EXTRACT_PROMPT.format(
            memory=self._mem_text(), date=self._buf_date, turns=turns_text
        )
        self._apply_ops(llm.generate_json(self.model, prompt, think=False, schema=OPERATIONS_SCHEMA))
        self._buf_turns = []

    def ingest(self, role, content, date):
        if role != "user" or not content.strip():
            return
        self._buf_date = date
        self._buf_turns.append(content)
        if len(self._buf_turns) >= BATCH_SIZE:
            self._flush()

    def ingest_session_end(self):
        pass  # accumulate across sessions; flush fires when buffer hits BATCH_SIZE or at query time

    def query(self, question, question_date):
        self._flush()
        hits = self.store.search(question, k=TOP_K)
        mem_text = "\n".join(f"- {h['content']}" for h in hits) or "(no relevant memory)"
        prompt = QUERY_PROMPT.format(date=question_date, memory=mem_text, question=question)
        return llm.generate(self.model, prompt)

    def reset(self):
        self._buf_turns = []
        self._buf_date = None
        self.store.reset()
