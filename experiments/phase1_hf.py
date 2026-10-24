"""
Phase 1 LongMemEval QA using the remote 32B reference model (HF Inference API).

Runs the naive manager condition (LLM-driven fact extraction + RAG query) using
Qwen3-32B via HuggingFace Inference API, matching the local-model Phase 1 setup.

Usage:
  export HF_TOKEN=<your_token>
  python experiments/phase1_hf.py [n_examples]

  n_examples: number of LongMemEval_S examples (default: 20)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

import src.models.hf_client as hf_llm
from src.memory.managers.naive import NaiveManager, BATCH_EXTRACT_PROMPT, QUERY_PROMPT
from src.memory.store import MemoryStore

TOP_K        = 10
MAX_MEM_SHOWN = 30
BATCH_SIZE   = 50

MODEL_SLUG = "qwen3-32b-hf"
DATA       = "data/longmemeval_s.json"
OUT        = f"results/phase1_naive_{MODEL_SLUG}_n{N}.jsonl"
os.makedirs("results", exist_ok=True)


class NaiveHFManager(NaiveManager):
    """NaiveManager backed by hf_client instead of ollama_client."""

    def __init__(self):
        self.model = hf_llm.active_model()
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
        result = hf_llm.generate_json(prompt, max_tokens=600)
        self._apply_ops(result)
        self._buf_turns = []

    def query(self, question, question_date):
        self._flush()
        hits = self.store.search(question, k=TOP_K)
        mem_text = "\n".join(f"- {h['content']}" for h in hits) or "(no relevant memory)"
        prompt = QUERY_PROMPT.format(
            date=question_date, memory=mem_text, question=question
        )
        return hf_llm.generate(prompt, max_tokens=256)


if __name__ == "__main__":
    active = hf_llm.active_model()
    print(f"=== Phase 1 LongMemEval — Remote Reference ===")
    print(f"Model : {active}  (slug: {MODEL_SLUG})")
    print(f"N     : {N}")
    print(f"Out   : {OUT}\n")

    manager = NaiveHFManager()
    from src.eval.runners import run_longmemeval
    run_longmemeval(manager, DATA, n=N, output_path=OUT)
