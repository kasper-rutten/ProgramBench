# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from programbench.lab import build_index, write_html_report, write_index

app = typer.Typer(
    name="lab",
    no_args_is_help=True,
    help="Index and browse local ProgramBench run artifacts.",
)


def _fmt_duration(seconds: object) -> str:
    if not isinstance(seconds, (int, float)):
        return ""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _fmt_tokens(value: object) -> str:
    if not isinstance(value, int):
        return ""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


@app.command("index")
def index(
    runs_root: Path = typer.Argument(Path("runs"), help="Directory containing run folders"),
    output: Path = typer.Option(Path("runs/index.json"), "-o", "--output", help="JSON index path"),
) -> None:
    """Write a machine-readable index for local run folders."""
    data = write_index(runs_root, output, repo_root=Path.cwd())
    Console().print(f"Wrote {output} with {data['total_runs']} run(s).")


@app.command()
def report(
    runs_root: Path = typer.Argument(Path("runs"), help="Directory containing run folders"),
    output: Path = typer.Option(Path("reports/programbench-lab.html"), "-o", "--output", help="HTML report path"),
) -> None:
    """Write a static HTML report for local run folders."""
    data = write_html_report(runs_root, output, repo_root=Path.cwd())
    Console().print(f"Wrote {output} with {data['total_runs']} run(s).")


@app.command()
def summary(
    runs_root: Path = typer.Argument(Path("runs"), help="Directory containing run folders"),
) -> None:
    """Print a compact run table."""
    data = build_index(runs_root, repo_root=Path.cwd())
    table = Table(title="ProgramBench Local Runs", show_lines=False)
    table.add_column("Experiment")
    table.add_column("Run", style="bold")
    table.add_column("Model")
    table.add_column("Reasoning")
    table.add_column("Instances", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Created")
    for run in data["runs"]:
        accounting = run.get("accounting") or {}
        experiment = run.get("experiment") or {}
        table.add_row(
            experiment.get("name") or run.get("label") or run["run_id"],
            run["run_id"],
            run.get("model") or "",
            run.get("reasoning_effort") or "",
            f"{run['evaluated_instances']}/{run['submitted_instances']}",
            f"{run['average_score_percent']:.0f}",
            f"{run['total_resolved']}/{run['total_tests']}",
            _fmt_duration(accounting.get("wall_time_seconds")),
            str(accounting.get("turns") or 0),
            str(accounting.get("tool_calls") or 0),
            _fmt_tokens(accounting.get("total_tokens")),
            run.get("created_at") or "",
        )
    Console().print(table)
