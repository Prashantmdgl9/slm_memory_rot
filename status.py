#!/usr/bin/env python3
"""Run: python status.py"""
import json, os, glob

MODELS = ["qwen3:4b", "qwen3:8b", "llama3.2:3b", "llama3.1:8b", "qwen2.5:7b"]
MANAGERS = ["no_memory", "full_context", "rag_only", "naive"]
N = 20

def load(path):
    rows = [json.loads(l) for l in open(path)]
    hit = sum(r["hit"] for r in rows) / len(rows)
    f1 = sum(r["f1"] for r in rows) / len(rows)
    return len(rows), hit, f1

done, pending = [], []
for model in MODELS:
    tag = model.replace(":", "-")
    for mgr in MANAGERS:
        path = f"results/phase1_{mgr}_{tag}_n{N}.jsonl"
        if os.path.exists(path):
            n, hit, f1 = load(path)
            done.append((mgr, model, n, hit, f1))
        else:
            pending.append((mgr, model))

print(f"\n{'='*62}")
print(f"  Phase 1 Status — {len(done)}/{len(done)+len(pending)} runs complete")
print(f"{'='*62}")

if done:
    print(f"\n  {'Manager':<14} {'Model':<16} {'N':>4}  {'Hit':>6}  {'F1':>6}")
    print(f"  {'-'*54}")
    for mgr, model, n, hit, f1 in done:
        print(f"  {mgr:<14} {model:<16} {n:>4}  {hit:>5.1%}  {f1:>6.3f}")

if pending:
    print(f"\n  Pending ({len(pending)}):")
    for mgr, model in pending:
        print(f"    - {mgr} / {model}")

print()
