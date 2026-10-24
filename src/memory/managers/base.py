from abc import ABC, abstractmethod

class MemoryManager(ABC):
    @abstractmethod
    def ingest(self, role, content, date):
        pass

    @abstractmethod
    def query(self, question, question_date):
        pass

    @abstractmethod
    def reset(self):
        pass
