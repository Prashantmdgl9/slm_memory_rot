from src.memory.managers.base import MemoryManager
from src.memory.store import MemoryStore
from src.models import ollama_client as llm

TOP_K = 10

class RagOnlyManager(MemoryManager):
    def __init__(self, model):
        self.model = model
        self.store = MemoryStore()

    def ingest(self, role, content, date):
        self.store.add(f"[{date}] {role}: {content}", created_at=date)

    def query(self, question, question_date):
        hits = self.store.search(question, k=TOP_K)
        context = "\n".join(h["content"] for h in hits)
        prompt = (
            "Answer the question using retrieved conversation history.\n"
            f"Today's date: {question_date}\n\n"
            f"Relevant history:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer concisely with just the fact:"
        )
        return llm.generate(self.model, prompt)

    def reset(self):
        self.store.reset()
