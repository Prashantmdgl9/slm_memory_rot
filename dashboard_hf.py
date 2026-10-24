"""
Live dashboard for the Phase 2 HF (32B reference) run.
Usage:
  python3 dashboard_hf.py [--launch]
  --launch  also (re)start the phase2_hf.py process
"""
import sys, os, json, time, subprocess
from pathlib import Path
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.console import Console

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"
SLUG    = "qwen3-32b-hf"
N_TURNS = 80
OPS     = RESULTS / f"phase2_{SLUG}_ops.jsonl"
HEALTH  = RESULTS / f"phase2_{SLUG}_health.jsonl"
QA      = RESULTS / f"phase2_{SLUG}_qa.jsonl"
LOG     = RESULTS / f"phase2_{SLUG}.log"


def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def bar(done, total, width=20):
    filled = int(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def make_display(start_time):
    ops     = load_jsonl(OPS)
    health  = load_jsonl(HEALTH)
    qa      = load_jsonl(QA)
    done    = len(ops)
    elapsed = time.time() - start_time

    # ── Progress panel ────────────────────────────────────────────────────
    pct = 100 * done / N_TURNS
    eta_str = "—"
    if done > 0 and done < N_TURNS:
        rate    = elapsed / done
        eta_s   = rate * (N_TURNS - done)
        m, s    = divmod(int(eta_s), 60)
        eta_str = f"{m}m {s:02d}s"
    elif done >= N_TURNS:
        eta_str = "done"

    avg_call = ""
    if ops:
        times = [r["call_time_s"] for r in ops if "call_time_s" in r]
        avg_call = f"{sum(times)/len(times):.1f}s/call" if times else ""

    prog_table = Table(box=None, show_header=False, padding=(0, 1))
    prog_table.add_column(width=12)
    prog_table.add_column(width=30)
    prog_table.add_row("Model",    "[cyan]Qwen3-32B[/cyan] (HF Inference API)")
    prog_table.add_row("Progress", f"[{'green' if done>=N_TURNS else 'yellow'}]{bar(done, N_TURNS)}[/] {done}/{N_TURNS} ({pct:.0f}%)")
    prog_table.add_row("ETA",      f"[yellow]{eta_str}[/yellow]   elapsed {int(elapsed//60)}m {int(elapsed%60):02d}s")
    prog_table.add_row("Avg call", avg_call)

    # ── Per-op accuracy table ─────────────────────────────────────────────
    op_table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", title="Op Accuracy")
    op_table.add_column("Op",        width=8)
    op_table.add_column("Correct",   width=8,  justify="right")
    op_table.add_column("Total",     width=6,  justify="right")
    op_table.add_column("Acc",       width=6,  justify="right")

    op_counts = {}
    for rec in ops:
        gt   = rec["ground_truth_op"]
        pred = rec["predicted_op"]
        if gt not in op_counts:
            op_counts[gt] = {"correct": 0, "total": 0}
        op_counts[gt]["total"]   += 1
        if rec["correct"]:
            op_counts[gt]["correct"] += 1

    for op in ("ADD", "UPDATE", "DELETE", "NOOP"):
        c = op_counts.get(op, {"correct": 0, "total": 0})
        acc = c["correct"] / c["total"] if c["total"] else 0
        color = "green" if acc >= 0.8 else "yellow" if acc >= 0.5 else "red"
        op_table.add_row(
            op,
            str(c["correct"]),
            str(c["total"]),
            f"[{color}]{acc:.0%}[/{color}]" if c["total"] else "—",
        )

    if ops:
        total_correct = sum(1 for r in ops if r["correct"])
        overall = total_correct / len(ops)
        op_table.add_row("─" * 6, "─" * 6, "─" * 4, "─" * 4)
        op_table.add_row("[bold]Overall[/bold]", str(total_correct), str(len(ops)),
                         f"[bold]{overall:.0%}[/bold]")

    # ── Recent turns ──────────────────────────────────────────────────────
    recent = Table(box=box.SIMPLE, show_header=True, header_style="bold",
                   title=f"Recent turns (last 5 of {done})")
    recent.add_column("Turn", width=5,  justify="right")
    recent.add_column("GT",   width=7)
    recent.add_column("Pred", width=7)
    recent.add_column("",     width=5)
    recent.add_column("Fact", width=28)
    recent.add_column("Time", width=7,  justify="right")

    for rec in ops[-5:]:
        ok    = rec["correct"]
        color = "green" if ok else "red"
        recent.add_row(
            str(rec["turn_id"]),
            rec["ground_truth_op"],
            f"[{color}]{rec['predicted_op']}[/{color}]",
            "✓" if ok else "✗",
            (rec.get("fact_key") or "noop")[:28],
            f"{rec.get('call_time_s', 0):.1f}s",
        )

    # ── QA results (shown once available) ────────────────────────────────
    qa_text = ""
    if qa:
        hits = sum(r["hit"] for r in qa)
        f1   = sum(r["f1"] for r in qa) / len(qa)
        qa_text = f"\nQA ({len(qa)} questions):  EM=[green]{hits/len(qa):.0%}[/green]  F1=[green]{f1:.3f}[/green]"

    # ── Final store health ────────────────────────────────────────────────
    health_text = ""
    if health:
        h = health[-1]
        rot = h.get("rot_score", "?")
        health_text = (f"\nStore health:  entries={h.get('n_entries')}  "
                       f"stale={h.get('stale_count')}  dupes={h.get('duplicate_count')}  "
                       f"rot=[{'red' if isinstance(rot, float) and rot > 0.3 else 'green'}]{rot}[/]")

    status = "[bold green]COMPLETE ✓[/bold green]" if done >= N_TURNS else "[bold yellow]RUNNING…[/bold yellow]"

    panel = Panel(
        prog_table,
        title=f"[bold]Phase 2 — 32B Reference  {status}[/bold]",
        border_style="cyan" if done < N_TURNS else "green",
    )

    return panel, op_table, recent, qa_text + health_text


def launch():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("HF_TOKEN not set — set it before launching.")
        sys.exit(1)
    env = {**os.environ, "HF_TOKEN": token}
    with open(LOG, "a") as lf:
        p = subprocess.Popen(
            ["python3", "experiments/phase2_hf.py"],
            cwd=ROOT, stdout=lf, stderr=lf, env=env,
        )
    print(f"Launched phase2_hf.py  (PID {p.pid})")
    return p


def main():
    do_launch = "--launch" in sys.argv
    start_time = time.time()
    proc = None

    if do_launch:
        proc = launch()

    console = Console()

    with Live(console=console, refresh_per_second=0.2, screen=False) as live:
        try:
            while True:
                panel, op_table, recent, footer = make_display(start_time)
                from rich.console import Group
                from rich.text import Text as RText
                live.update(Group(panel, op_table, recent, RText.from_markup(footer) if footer else RText("")))

                ops_done = len(load_jsonl(OPS))
                qa_done  = len(load_jsonl(QA))
                if ops_done >= N_TURNS and qa_done > 0:
                    break

                # Check if process died
                if proc and proc.poll() is not None and ops_done < N_TURNS:
                    live.stop()
                    print(f"\n[!] Process exited early at turn {ops_done}. Check {LOG}")
                    break

                time.sleep(5)
        except KeyboardInterrupt:
            pass

    # Final render
    panel, op_table, recent, footer = make_display(start_time)
    console.print(panel)
    console.print(op_table)
    console.print(recent)
    if footer:
        console.print(footer)


if __name__ == "__main__":
    main()
