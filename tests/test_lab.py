import json

from programbench.eval.eval import EvaluationResult, TestResult
from programbench.lab import build_index, read_event_summary, render_html_report, summarize_run


def test_read_event_summary_counts_completed_items(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "t1"}),
                json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}),
                json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
                json.dumps({"type": "item.completed", "item": {"type": "file_change"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 4}}),
            ]
        )
    )

    summary = read_event_summary(events)

    assert summary.thread_id == "t1"
    assert summary.agent_message_count == 1
    assert summary.command_count == 1
    assert summary.file_change_count == 1
    assert summary.completed_turns == 1
    assert summary.usage["input_tokens"] == 3


def test_run_accounting_sums_per_instance_events(tmp_path):
    run = tmp_path / "runs" / "demo"
    inst_a = run / "fake__a.abc1234"
    inst_b = run / "fake__b.abc1234"
    inst_a.mkdir(parents=True)
    inst_b.mkdir(parents=True)
    result = EvaluationResult(test_results=[TestResult(name="test_ok", branch="b1", status="passed", extra={})])
    (inst_a / "fake__a.abc1234.eval.json").write_text(result.model_dump_json())
    (inst_b / "fake__b.abc1234.eval.json").write_text(result.model_dump_json())
    events = run / "events"
    events.mkdir()
    (events / "fake__a.abc1234.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}),
            ]
        )
    )
    (events / "fake__b.abc1234.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "file_change"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 3}}),
            ]
        )
    )
    accounting = run / "accounting"
    accounting.mkdir()
    (accounting / "fake__a.abc1234.json").write_text(
        json.dumps(
            {
                "wall_time_seconds": 10,
                "wall_clock_limit_seconds": 1200,
                "validator_call_limit": 5,
                "validator_call_count": 2,
                "codex_exit_code": 0,
            }
        )
    )

    summary = summarize_run(run, repo_root=tmp_path, instances_by_id={})

    assert summary.accounting.turns == 2
    assert summary.accounting.tool_calls == 2
    assert summary.accounting.tokens["input_tokens"] == 15
    assert summary.accounting.total_tokens == 20
    assert summary.accounting.wall_time_seconds is not None
    assert summary.accounting.validator_call_count == 2
    assert summary.accounting.validator_call_limit == 5
    assert summary.instances[0].events_path is not None


def test_summarize_run_with_manifest_and_eval_json(tmp_path):
    run = tmp_path / "runs" / "demo"
    inst = run / "fake__task.abc1234"
    inst.mkdir(parents=True)
    (inst / "submission.tar.gz").write_bytes(b"fake")
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "demo",
                "agent": "codex-cli",
                "model": "gpt-test",
                "reasoning_effort": "high",
            }
        )
    )
    result = EvaluationResult(
        test_results=[
            TestResult(name="test_ok", branch="b1", status="passed", extra={}),
            TestResult(name="test_bad", branch="b1", status="failure", extra={"message": "nope\nmore"}),
        ],
        test_branches=["b1"],
    )
    (inst / "fake__task.abc1234.eval.json").write_text(result.model_dump_json())

    summary = summarize_run(run, repo_root=tmp_path, instances_by_id={})

    assert summary.run_id == "demo"
    assert summary.model == "gpt-test"
    assert summary.reasoning_effort == "high"
    assert summary.submitted_instances == 1
    assert summary.evaluated_instances == 1
    assert summary.average_score == 0.5
    assert summary.total_resolved == 1
    assert summary.total_tests == 2
    assert summary.instances[0].top_failures[0]["message"] == "nope"


def test_build_index_and_render_report(tmp_path):
    run = tmp_path / "runs" / "demo"
    inst = run / "fake__task.abc1234"
    inst.mkdir(parents=True)
    (inst / "submission.tar.gz").write_bytes(b"fake")
    result = EvaluationResult(
        test_results=[TestResult(name="test_ok", branch="b1", status="passed", extra={})],
        test_branches=["b1"],
    )
    (inst / "fake__task.abc1234.eval.json").write_text(result.model_dump_json())

    index = build_index(tmp_path / "runs", repo_root=tmp_path)
    html = render_html_report(index)

    assert index["total_runs"] == 1
    assert index["runs"][0]["run_id"] == "demo"
    assert "ProgramBench Lab Report" in html
    assert "fake__task.abc1234" in html
