from src.memory.managers.base import MemoryManager
from src.models import ollama_client as llm

class NoMemoryManager(MemoryManager):
    def __init__(self, model):
        self.model = model

    def ingest(self, role, content, date):
        pass

    def query(self, question, question_date):
        prompt = (
            f"Today's date: {question_date}\n\n"
            f"Question: {question}\n"
            "Answer concisely with just the fact:"
        )
        return llm.generate(self.model, prompt)

    def reset(self):
        pass
