import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        """
        # ProgramBench Run Browser

        A notebook-like local browser for `runs/` artifacts. It reads
        `run.json`, Codex `events.jsonl`, `submission.tar.gz`, and
        `<instance_id>.eval.json` files, then applies the same score model as
        `programbench info`.
        """
    )
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <style>
        .pb-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
            margin: 1rem 0 1.25rem;
            font-size: 0.95rem;
        }
        .pb-table th,
        .pb-table td {
            padding: 0.35rem 0.35rem;
            border-bottom: 1px solid rgba(120, 120, 120, 0.22);
            text-align: left;
            vertical-align: top;
            overflow-wrap: anywhere;
        }
        .pb-table th {
            font-weight: 700;
            white-space: nowrap;
        }
        .pb-num {
            text-align: right !important;
            white-space: nowrap;
            overflow-wrap: normal !important;
        }
        .pb-accounting {
            font-variant-numeric: tabular-nums;
        }
        .pb-subtle {
            color: #667085;
            font-size: 0.88em;
        }
        .pb-detail {
            margin-top: 1.25rem;
            max-width: 100%;
        }
        .pb-detail p {
            margin: 0.35rem 0;
        }
        .pb-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem 1rem;
        }
        .pb-failures th:nth-child(1) { width: 34%; }
        .pb-failures th:nth-child(2) { width: 12%; }
        .pb-failures th:nth-child(3) { width: 54%; }
        </style>
        """
    )
    return


@app.cell
def _():
    from pathlib import Path
    import html

    import marimo as mo

    from programbench.lab import build_index

    return Path, build_index, html, mo


@app.cell
def _(mo):
    runs_root_input = mo.ui.text(value="runs", label="Runs root")
    runs_root_input
    return (runs_root_input,)


@app.cell
def _(Path, build_index, runs_root_input):
    runs_root = Path(runs_root_input.value)
    index = build_index(runs_root, repo_root=Path.cwd())
    runs = index["runs"]
    return index, runs, runs_root


@app.cell
def _(html, mo, runs):
    def score_cell(score):
        return "solved" if score == 1.0 else f"{score * 100:.0f}"

    def created_cell(value):
        text = str(value or "")
        if "T" in text:
            return html.escape(text.split("T", 1)[0])
        return html.escape(text)

    def duration_cell(seconds):
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

    def tokens_cell(value):
        if not isinstance(value, int):
            return ""
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return str(value)

    rows = []
    for run in runs:
        _accounting = run.get("accounting") or {}
        _experiment = run.get("experiment") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(_experiment.get('name') or run.get('label') or run['run_id']))}</td>"
            f"<td>{html.escape(run['run_id'])}</td>"
            f"<td>{html.escape(str(run.get('model') or ''))}</td>"
            f"<td>{html.escape(str(run.get('reasoning_effort') or ''))}</td>"
            f"<td>{run['evaluated_instances']}/{run['submitted_instances']}</td>"
            f"<td class='pb-num'>{score_cell(run['average_score'])}</td>"
            f"<td class='pb-num'>{run['total_resolved']}/{run['total_tests']}</td>"
            f"<td class='pb-accounting'>{duration_cell(_accounting.get('wall_time_seconds'))}"
            f"<br><span class='pb-subtle'>{_accounting.get('turns', 0)} turns, "
            f"{_accounting.get('tool_calls', 0)} tools, "
            f"{tokens_cell(_accounting.get('total_tokens'))}</span></td>"
            f"<td>{created_cell(run.get('created_at'))}</td>"
            "</tr>"
        )
    mo.Html(
        "<table class='pb-table'>"
        "<colgroup>"
        "<col style='width: 18%'><col style='width: 20%'><col style='width: 12%'><col style='width: 8%'>"
        "<col style='width: 6%'><col style='width: 7%'><col style='width: 10%'>"
        "<col style='width: 13%'><col style='width: 6%'>"
        "</colgroup>"
        "<thead><tr><th>Experiment</th><th>Run</th><th>Model</th><th>Effort</th><th>Eval</th>"
        "<th>Score</th><th>Passed</th><th>Accounting</th><th>Created</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else "<p>No runs found.</p>"
    )
    return duration_cell, score_cell, tokens_cell


@app.cell
def _(mo, runs):
    run_ids = [run["run_id"] for run in runs]
    selected_run_id = run_ids[0] if run_ids else ""
    run_selector = mo.ui.dropdown(options=run_ids, value=selected_run_id, label="Run")
    run_selector
    return (run_selector,)


@app.cell
def _(run_selector, runs):
    selected_run = next((run for run in runs if run["run_id"] == run_selector.value), None)
    selected_instances = selected_run["instances"] if selected_run else []
    return selected_instances, selected_run


@app.cell
def _(duration_cell, html, mo, score_cell, selected_run, tokens_cell):
    if selected_run is None:
        run_detail = mo.md("No run selected.")
    else:
        _accounting = selected_run.get("accounting") or {}
        _experiment = selected_run.get("experiment") or {}
        _factor_text = ", ".join(
            f"{key}: {value}" for key, value in (_experiment.get("factors") or {}).items()
        )
        _tag_text = ", ".join(_experiment.get("tags") or [])
        _usage = _accounting.get("tokens") or selected_run.get("event_summary", {}).get("usage", {})
        _usage_text = ", ".join(f"{k}: {v:,}" for k, v in _usage.items()) or "no usage event"
        links = []
        for label, key in [
            ("manifest", "manifest_path"),
            ("prompt", "prompt_path"),
            ("final", "final_path"),
            ("events", "events_path"),
        ]:
            if selected_run.get(key):
                links.append(f"<code>{html.escape(label)}</code>: {html.escape(selected_run[key])}")
        run_detail = mo.Html(
            f"""
            <section class="pb-detail">
            <h2>{html.escape(selected_run['run_id'])}</h2>
            <p><b>Experiment:</b> {html.escape(str(_experiment.get('name') or selected_run.get('label') or selected_run['run_id']))}</p>
            <p>{html.escape(str(_experiment.get('description') or ''))}</p>
            <p class="pb-subtle">{html.escape(_tag_text)}{'<br>' if _tag_text and _factor_text else ''}{html.escape(_factor_text)}</p>
            <p class="pb-meta">
            <span><b>Model:</b> {html.escape(str(selected_run.get('model') or ''))}</span>
            <span><b>Reasoning:</b> {html.escape(str(selected_run.get('reasoning_effort') or ''))}</span>
            <span><b>Score:</b> {score_cell(selected_run['average_score'])}</span>
            <span><b>Instances:</b> {selected_run['evaluated_instances']} evaluated /
            {selected_run['submitted_instances']} submitted</span>
            <span><b>Passed:</b> {selected_run['total_resolved']}/{selected_run['total_tests']}</span>
            <span><b>Harness:</b> {html.escape(str(selected_run.get('harness_mode') or ''))}</span>
            <span><b>Wall time:</b> {duration_cell(_accounting.get('wall_time_seconds'))}</span>
            <span><b>Turns:</b> {_accounting.get('turns', 0)}</span>
            <span><b>Tool calls:</b> {_accounting.get('tool_calls', 0)}</span>
            <span><b>Tokens:</b> {tokens_cell(_accounting.get('total_tokens'))}</span>
            <span><b>Validator calls:</b> {_accounting.get('validator_call_count', 0)}
            / {_accounting.get('validator_call_limit') or ''}</span>
            </p>
            <p><b>Codex usage:</b> {html.escape(_usage_text)}</p>
            <p><b>Wall time source:</b> {html.escape(str(_accounting.get('wall_time_source') or 'unknown'))}</p>
            <p>{'<br>'.join(links)}</p>
            <p>{html.escape(str(selected_run.get('notes') or ''))}</p>
            </section>
            """
        )
    run_detail
    return


@app.cell
def _(mo, selected_instances):
    instance_ids = [instance["instance_id"] for instance in selected_instances]
    selected_instance_id = instance_ids[0] if instance_ids else ""
    instance_selector = mo.ui.dropdown(options=instance_ids, value=selected_instance_id, label="Instance")
    instance_selector
    return (instance_selector,)


@app.cell
def _(duration_cell, html, instance_selector, mo, score_cell, selected_instances, tokens_cell):
    selected_instance = next(
        (instance for instance in selected_instances if instance["instance_id"] == instance_selector.value),
        None,
    )
    if selected_instance is None:
        instance_detail = mo.md("No instance selected.")
    else:
        _accounting = selected_instance.get("accounting") or {}
        _usage = _accounting.get("tokens") or {}
        _usage_text = ", ".join(f"{k}: {v:,}" for k, v in _usage.items()) or "no usage event"
        counts = ", ".join(f"{k}: {v}" for k, v in selected_instance["status_counts"].items())
        failures = selected_instance.get("top_failures") or []
        failure_rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(f['name'])}</code></td>"
            f"<td>{html.escape(f.get('status', ''))}</td>"
            f"<td>{html.escape(f.get('message', ''))}</td>"
            "</tr>"
            for f in failures
        )
        instance_detail = mo.Html(
            f"""
            <section class="pb-detail">
            <h3>{html.escape(selected_instance['instance_id'])}</h3>
            <p class="pb-meta">
            <span><b>Score:</b> {score_cell(selected_instance['score'])}</span>
            <span><b>Passed:</b> {selected_instance['n_resolved']}/{selected_instance['n_tests']}</span>
            <span><b>Status counts:</b> {html.escape(counts)}</span>
            <span><b>Wall time:</b> {duration_cell(_accounting.get('wall_time_seconds'))}</span>
            <span><b>Turns:</b> {_accounting.get('turns', 0)}</span>
            <span><b>Tool calls:</b> {_accounting.get('tool_calls', 0)}</span>
            <span><b>Tokens:</b> {tokens_cell(_accounting.get('total_tokens'))}</span>
            </p>
            <p><b>Codex usage:</b> {html.escape(_usage_text)}</p>
            <p><b>Events:</b> <code>{html.escape(str(selected_instance.get('events_path') or ''))}</code></p>
            <p><b>Eval JSON:</b> <code>{html.escape(selected_instance['eval_json'])}</code></p>
            <p><b>Submission:</b> <code>{html.escape(str(selected_instance.get('submission_tar') or ''))}</code></p>
            <table class='pb-table pb-failures'>
            <thead><tr><th>Failure</th><th>Status</th><th>Message</th></tr></thead>
            <tbody>{failure_rows}</tbody>
            </table>
            </section>
            """
        )
    instance_detail
    return (selected_instance,)


if __name__ == "__main__":
    app.run()
