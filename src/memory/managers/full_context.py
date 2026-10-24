from src.memory.managers.base import MemoryManager
from src.models import ollama_client as llm

MAX_CHARS = 12000  # ~3k tokens; truncates oldest turns first

class FullContextManager(MemoryManager):
    def __init__(self, model):
        self.model = model
        self._turns = []

    def ingest(self, role, content, date):
        self._turns.append(f"[{date}] {role}: {content}")

    def query(self, question, question_date):
        context = "\n".join(self._turns)
        if len(context) > MAX_CHARS:
            context = context[-MAX_CHARS:]
        prompt = (
            "Answer the question using the user's conversation history.\n"
            f"Today's date: {question_date}\n\n"
            f"History:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer concisely with just the fact:"
        )
        return llm.generate(self.model, prompt)

    def reset(self):
        self._turns = []
