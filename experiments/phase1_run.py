"""
Phase 1: Run a baseline manager on LongMemEval_S.

Usage:
  python experiments/phase1_run.py [manager] [model] [n_examples]

  manager: no_memory | full_context | rag_only | naive  (default: rag_only)
  model:   any Ollama model tag                          (default: qwen2.5:7b)
  n:       number of examples                            (default: 20)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MANAGER = sys.argv[1] if len(sys.argv) > 1 else "rag_only"
MODEL   = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:7b"
N       = int(sys.argv[3]) if len(sys.argv) > 3 else 20

DATA = "data/longmemeval_s.json"
OUT  = f"results/phase1_{MANAGER}_{MODEL.replace(':', '-')}_n{N}.jsonl"

def get_manager():
    if MANAGER == "no_memory":
        from src.memory.managers.no_memory import NoMemoryManager
        return NoMemoryManager(MODEL)
    if MANAGER == "full_context":
        from src.memory.managers.full_context import FullContextManager
        return FullContextManager(MODEL)
    if MANAGER == "rag_only":
        from src.memory.managers.rag_only import RagOnlyManager
        return RagOnlyManager(MODEL)
    if MANAGER == "naive":
        from src.memory.managers.naive import NaiveManager
        return NaiveManager(MODEL)
    raise ValueError(f"Unknown manager: {MANAGER}")

if __name__ == "__main__":
    print(f"Manager={MANAGER} | Model={MODEL} | N={N} | Out={OUT}")
    from src.eval.runners import run_longmemeval
    run_longmemeval(get_manager(), DATA, n=N, output_path=OUT)
