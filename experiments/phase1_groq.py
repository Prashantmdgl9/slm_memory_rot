"""
Phase 1 LongMemEval QA using Groq (Llama-3.3-70B) as the large-model reference.

Runs the naive manager condition matching the local-model Phase 1 setup.

Usage:
  export GROQ_API_KEY=gsk_...   (free key from console.groq.com)
  python experiments/phase1_groq.py [n_examples]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

import src.models.groq_client as groq_llm
from src.memory.managers.naive import NaiveManager, BATCH_EXTRACT_PROMPT, QUERY_PROMPT
from src.memory.store import MemoryStore

MODEL_SLUG = "llama3.3-70b-groq"
DATA       = "data/longmemeval_s.json"
OUT        = f"results/phase1_naive_{MODEL_SLUG}_n{N}.jsonl"
os.makedirs("results", exist_ok=True)


class NaiveGroqManager(NaiveManager):

    def __init__(self):
        self.model = groq_llm.active_model()
        self.store = MemoryStore()
        self._buf_turns = []
        self._buf_date  = None

    def _flush(self):
        if not self._buf_turns:
            return
        turns_text = "\n".join(f"- {c[:400]}" for c in self._buf_turns)
        prompt = BATCH_EXTRACT_PROMPT.format(
            memory=self._mem_text(), date=self._buf_date, turns=turns_text
        )
        result = groq_llm.generate_json(prompt, max_tokens=600)
        self._apply_ops(result)
        self._buf_turns = []

    def query(self, question, question_date):
        self._flush()
        hits = self.store.search(question, k=10)
        mem_text = "\n".join(f"- {h['content']}" for h in hits) or "(no relevant memory)"
        prompt = QUERY_PROMPT.format(
            date=question_date, memory=mem_text, question=question
        )
        return groq_llm.generate(prompt, max_tokens=256)


if __name__ == "__main__":
    active = groq_llm.active_model()
    print(f"=== Phase 1 LongMemEval — Groq Reference ===")
    print(f"Model : {active}  (slug: {MODEL_SLUG})")
    print(f"N     : {N}")
    print(f"Out   : {OUT}\n")

    manager = NaiveGroqManager()
    from src.eval.runners import run_longmemeval
    run_longmemeval(manager, DATA, n=N, output_path=OUT)
