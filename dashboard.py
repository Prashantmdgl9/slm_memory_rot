"""
Live progress dashboard for Phase 1 runs.
Usage: python3 dashboard.py [--no-launch]
  --no-launch   show dashboard only, don't start the runs
"""
import sys, json, os, time, subprocess
from pathlib import Path
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich import box

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
N = 20

RUNS = [
    # (manager, model_tag, result_file_model_slug)
    ("full_context", "qwen3:8b",    "qwen3-8b"),
    ("rag_only",     "qwen3:8b",    "qwen3-8b"),
    ("rag_only",     "llama3.1:8b", "llama3.1-8b"),
    ("naive",        "qwen3:4b",    "qwen3-4b"),
    ("naive",        "qwen3:8b",    "qwen3-8b"),
    ("naive",        "llama3.2:3b", "llama3.2-3b"),
    ("naive",        "llama3.1:8b", "llama3.1-8b"),
    ("naive",        "qwen2.5:7b",  "qwen2.5-7b"),
]

def result_path(manager, slug):
    return RESULTS / f"phase1_{manager}_{slug}_n{N}.jsonl"

def load_records(path):
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records

def bar(done, total, width=12):
    filled = int(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)

def make_table(start_times):
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
              title="[bold]Phase 1 — Live Progress[/bold]", title_style="bold white")
    t.add_column("Manager",    style="cyan",    width=14)
    t.add_column("Model",      style="magenta", width=14)
    t.add_column("Progress",   width=20, justify="left")
    t.add_column("Done",       width=7,  justify="right")
    t.add_column("EM",         width=7,  justify="right")
    t.add_column("F1",         width=7,  justify="right")
    t.add_column("Elapsed",    width=10, justify="right")
    t.add_column("Status",     width=10)

    for manager, model, slug in RUNS:
        path = result_path(manager, slug)
        records = load_records(path)
        done = len(records)
        elapsed_s = time.time() - start_times.get((manager, slug), time.time())

        if done == 0:
            progress = Text(bar(0, N) + f"  0/{N}", style="dim")
            em_str  = Text("—", style="dim")
            f1_str  = Text("—", style="dim")
            elapsed = Text("—", style="dim")
            if (manager, slug) in start_times:
                status = Text("starting…", style="yellow")
            else:
                status = Text("pending",   style="dim")
        elif done >= N:
            hits   = sum(r["hit"] for r in records)
            mean_f1 = sum(r["f1"] for r in records) / done
            mins   = elapsed_s / 60
            progress = Text(bar(N, N) + f" {N}/{N}", style="green")
            em_str  = Text(f"{hits/N:.0%}", style="green")
            f1_str  = Text(f"{mean_f1:.3f}", style="green")
            elapsed = Text(f"{mins:.0f}m", style="green")
            status  = Text("done ✓",  style="bold green")
        else:
            hits    = sum(r["hit"] for r in records)
            mean_f1 = sum(r["f1"] for r in records) / done
            mins    = elapsed_s / 60
            pct     = done / N
            # ETA
            if elapsed_s > 10 and done > 0:
                rate    = elapsed_s / done           # s per example
                eta_s   = rate * (N - done)
                eta_str = f"eta {eta_s/60:.0f}m"
            else:
                eta_str = "…"
            progress = Text(bar(done, N) + f" {done}/{N}", style="yellow")
            em_str   = Text(f"{hits/done:.0%}", style="yellow")
            f1_str   = Text(f"{mean_f1:.3f}", style="yellow")
            elapsed  = Text(f"{mins:.0f}m", style="yellow")
            status   = Text(f"running {eta_str}", style="bold yellow")

        t.add_row(manager, model, progress, Text(f"{done}/{N}"), em_str, f1_str, elapsed, status)

    return t

def launch_runs(procs, start_times):
    for manager, model, slug in RUNS:
        path = result_path(manager, slug)
        records = load_records(path)
        if len(records) >= N:
            continue  # already complete
        key = (manager, slug)
        if key in procs:
            continue  # already launched
        log = RESULTS / f"phase1_{manager}_{slug}_n{N}.log"
        with open(log, "w") as lf:
            p = subprocess.Popen(
                ["python3", "experiments/phase1_run.py", manager, model, str(N)],
                cwd=ROOT, stdout=lf, stderr=lf
            )
        procs[key] = p
        start_times[key] = time.time()
        time.sleep(0.3)  # slight stagger to avoid Ollama overload

def main():
    launch = "--no-launch" not in sys.argv
    procs = {}
    start_times = {}

    if launch:
        print("Launching all pending runs in background…")
        launch_runs(procs, start_times)

    with Live(make_table(start_times), refresh_per_second=0.2, screen=False) as live:
        try:
            while True:
                time.sleep(5)
                live.update(make_table(start_times))

                # check if all done
                all_done = all(
                    len(load_records(result_path(m, s))) >= N
                    for m, _, s in RUNS
                )
                if all_done:
                    live.update(make_table(start_times))
                    break
        except KeyboardInterrupt:
            pass

    print("\nAll Phase 1 runs complete." if all_done else "\nDashboard stopped.")

if __name__ == "__main__":
    main()
