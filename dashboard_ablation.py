"""
Live dashboard for the 3-condition ablation runs.
Usage: python3 dashboard_ablation.py
"""
import json, time, sys
from pathlib import Path
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich import box
from rich.console import Console, Group

ROOT    = Path(__file__).parent
RESULTS = ROOT / "results"
N_OPS   = 80
N_QA    = 14

MODELS = [
    ("qwen3:4b",    "qwen3-4b"),
    ("qwen3:8b",    "qwen3-8b"),
    ("llama3.2:3b", "llama3.2-3b"),
    ("llama3.1:8b", "llama3.1-8b"),
    ("qwen2.5:7b",  "qwen2.5-7b"),
]

CONDITIONS = [
    ("naive",        "phase2",              "Naive"),
    ("typed_only",   "ablation_typed_only", "+Typed Slots"),
    ("full_fw",      "phase3",             "+Del Signals"),
    ("compaction",   "ablation_compaction", "+Compaction"),
]


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


def op_acc(ops, op_name):
    gt_pos = [r for r in ops if r["ground_truth_op"] == op_name]
    if not gt_pos:
        return None
    recall = sum(1 for r in gt_pos if r["predicted_op"] == op_name) / len(gt_pos)
    return recall


def bar(done, total, width=10):
    filled = int(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def make_table(start_time):
    elapsed = time.time() - start_time

    # ── Progress table ────────────────────────────────────────────────────
    prog = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                 title="[bold]Ablation — compaction runs[/bold]")
    prog.add_column("Model",    style="magenta", width=14)
    prog.add_column("Ops",      width=14)
    prog.add_column("Done",     width=7,  justify="right")
    prog.add_column("QA",       width=5,  justify="right")
    prog.add_column("Elapsed",  width=9,  justify="right")
    prog.add_column("Status",   width=16)

    all_done = True
    for model, slug in MODELS:
        ops_path = RESULTS / f"ablation_compaction_{slug}_ops.jsonl"
        qa_path  = RESULTS / f"ablation_compaction_{slug}_qa.jsonl"
        ops = load_jsonl(ops_path)
        qa  = load_jsonl(qa_path)
        done = len(ops)
        qa_done = len(qa)

        if done < N_OPS:
            all_done = False

        if done == 0:
            progress = Text(bar(0, N_OPS) + f"  0/{N_OPS}", style="dim")
            status   = Text("waiting…", style="dim")
            elapsed_str = Text("—", style="dim")
            qa_str = Text("—", style="dim")
        elif done >= N_OPS and qa_done >= N_QA:
            progress = Text(bar(N_OPS, N_OPS) + f" {N_OPS}/{N_OPS}", style="green")
            status   = Text("done ✓", style="bold green")
            elapsed_str = Text(f"{elapsed/60:.0f}m", style="green")
            qa_str = Text(f"{qa_done}/{N_QA}", style="green")
        else:
            eta_str = "…"
            if done > 0:
                rate  = elapsed / done
                eta_s = rate * (N_OPS - done)
                eta_str = f"eta {int(eta_s//60)}m{int(eta_s%60):02d}s"
            progress = Text(bar(done, N_OPS) + f" {done}/{N_OPS}", style="yellow")
            if done >= N_OPS:
                status = Text(f"QA {qa_done}/{N_QA}…", style="yellow")
            else:
                status = Text(f"running {eta_str}", style="bold yellow")
            elapsed_str = Text(f"{elapsed/60:.0f}m", style="yellow")
            qa_str = Text(f"{qa_done}/{N_QA}" if qa_done else "—", style="yellow")

        prog.add_row(model, progress, Text(f"{done}/{N_OPS}"), qa_str, elapsed_str, status)

    # ── Comparison table (UPDATE recall is the key metric) ─────────────────
    cmp = Table(box=box.SIMPLE, show_header=True, header_style="bold",
                title="[bold]UPDATE recall — 4-condition comparison[/bold]")
    cmp.add_column("Model",         style="magenta", width=14)
    cmp.add_column("Naive",         width=9,  justify="center")
    cmp.add_column("+Slots",        width=9,  justify="center")
    cmp.add_column("+Del",          width=9,  justify="center")
    cmp.add_column("+Compact",      width=10, justify="center")
    cmp.add_column("QA:Naive",      width=10, justify="center")
    cmp.add_column("QA:Slots",      width=10, justify="center")
    cmp.add_column("QA:Del",        width=9,  justify="center")
    cmp.add_column("QA:Compact",    width=11, justify="center")

    for model, slug in MODELS:
        row = [model]
        qa_row = []

        for cond_key, prefix, _ in CONDITIONS:
            ops_path = RESULTS / f"{prefix}_{slug}_ops.jsonl"
            qa_path  = RESULTS / f"{prefix}_{slug}_qa.jsonl"
            ops = load_jsonl(ops_path)
            qa  = load_jsonl(qa_path)

            upd = op_acc(ops, "UPDATE")
            if upd is None:
                row.append(Text("—", style="dim"))
            else:
                color = "green" if upd >= 0.7 else "yellow" if upd >= 0.3 else "red"
                row.append(Text(f"{upd:.0%}", style=color))

            if qa:
                hits = sum(r["hit"] for r in qa)
                color = "green" if hits/len(qa) >= 0.6 else "yellow" if hits/len(qa) >= 0.3 else "red"
                qa_row.append(Text(f"{hits}/{len(qa)}", style=color))
            else:
                qa_row.append(Text("—", style="dim"))

        cmp.add_row(*row, *qa_row)

    return prog, cmp, all_done


def main():
    start_time = time.time()
    console    = Console()

    with Live(console=console, refresh_per_second=0.2) as live:
        try:
            while True:
                prog, cmp, all_done = make_table(start_time)
                live.update(Group(prog, Text(""), cmp))
                if all_done:
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            pass

    prog, cmp, _ = make_table(start_time)
    console.print(prog)
    console.print(cmp)


if __name__ == "__main__":
    main()
