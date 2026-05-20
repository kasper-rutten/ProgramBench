# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Local run browsing helpers for ProgramBench experiments.

The functions in this module intentionally sit on top of the official
``EvaluationResult`` and ``InstanceEvalSummary`` models instead of inventing a
parallel scoring path. They are for local observability: indexing run folders,
capturing run metadata, and producing lightweight reports.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from programbench.eval.eval import EvaluationResult
from programbench.eval.eval_batch import InstanceEvalSummary
from programbench.utils.load_data import get_active_branches, get_ignored_tests, load_all_instances


class EventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = None
    command_count: int = 0
    file_change_count: int = 0
    agent_message_count: int = 0
    completed_turns: int = 0
    usage: dict[str, int] = Field(default_factory=dict)


class AccountingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_time_seconds: float | None = None
    wall_time_source: str | None = None
    wall_clock_limit_seconds: int | None = None
    codex_exit_code: int | None = None
    validator_call_limit: int | None = None
    validator_call_count: int = 0
    turns: int = 0
    tool_calls: int = 0
    command_executions: int = 0
    file_changes: int = 0
    agent_messages: int = 0
    tokens: dict[str, int] = Field(default_factory=dict)
    total_tokens: int = 0
    same_cost_mode: bool = False
    simulated_cost_unit: str | None = None
    simulated_cost_limit: float | None = None
    simulated_cost: float | None = None
    simulated_cost_selected: float | None = None
    simulated_cost_limit_usd: float | None = None
    simulated_cost_usd: float | None = None
    simulated_cost_selected_usd: float | None = None
    simulated_cost_exceeded: bool = False
    checkpoint_count: int = 0
    selected_checkpoint: str | None = None
    selected_turn: int | None = None
    selected_validator_call_count: int | None = None


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    factors: dict[str, str] = Field(default_factory=dict)


class InstanceLabSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str
    score: float
    score_percent: float
    n_resolved: int
    n_tests: int
    raw_n_tests: int
    status_counts: dict[str, int]
    error_code: str | None = None
    n_warnings: int = 0
    n_system_errors: int = 0
    test_branches: list[str] = Field(default_factory=list)
    executable_hash: str | None = None
    eval_json: str
    submission_tar: str | None = None
    prompt_path: str | None = None
    final_path: str | None = None
    events_path: str | None = None
    accounting_path: str | None = None
    accounting: AccountingSummary = Field(default_factory=AccountingSummary)
    top_failures: list[dict[str, str]] = Field(default_factory=list)


class RunLabSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_dir: str
    created_at: str | None = None
    experiment: ExperimentSummary = Field(default_factory=ExperimentSummary)
    label: str | None = None
    agent: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    codex_version: str | None = None
    harness_git_sha: str | None = None
    docker_backend: str | None = None
    docker_platform: str | None = None
    harness_mode: str | None = None
    validator_access: str | None = None
    notes: str | None = None
    prompt_path: str | None = None
    final_path: str | None = None
    events_path: str | None = None
    manifest_path: str | None = None
    event_summary: EventSummary = Field(default_factory=EventSummary)
    accounting: AccountingSummary = Field(default_factory=AccountingSummary)
    submitted_instances: int = 0
    evaluated_instances: int = 0
    average_score: float = 0.0
    average_score_percent: float = 0.0
    total_resolved: int = 0
    total_tests: int = 0
    raw_total_tests: int = 0
    total_failures: int = 0
    total_errors: int = 0
    instances: list[InstanceLabSummary] = Field(default_factory=list)


class ExperimentLabSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    factors: dict[str, str] = Field(default_factory=dict)
    run_count: int = 0
    evaluated_runs: int = 0
    submitted_instances: int = 0
    evaluated_instances: int = 0
    average_score: float = 0.0
    average_score_percent: float = 0.0
    total_resolved: int = 0
    total_tests: int = 0
    total_failures: int = 0
    total_errors: int = 0
    accounting: AccountingSummary = Field(default_factory=AccountingSummary)
    run_ids: list[str] = Field(default_factory=list)


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if v is not None and str(v) != ""}


def _manifest_experiment(manifest: dict[str, Any]) -> ExperimentSummary:
    raw = manifest.get("experiment")
    raw = raw if isinstance(raw, dict) else {}
    name = raw.get("name") or manifest.get("experiment_name") or manifest.get("label")
    description = raw.get("description") or manifest.get("experiment_description")
    tags = _string_list(raw.get("tags")) or _string_list(manifest.get("experiment_tags"))
    factors = _string_dict(raw.get("factors")) | _string_dict(manifest.get("experiment_factors"))
    for key in [
        "model",
        "reasoning_effort",
        "harness_mode",
        "validator_access",
        "agent",
        "solver_network",
    ]:
        value = manifest.get(key)
        if isinstance(value, str) and value:
            factors.setdefault(key, value)
    return ExperimentSummary(
        name=str(name) if name else None,
        description=str(description) if description else None,
        tags=tags,
        factors=dict(sorted(factors.items())),
    )


def _event_accounting(
    summary: EventSummary,
    wall_time_seconds: float | None = None,
    wall_time_source: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AccountingSummary:
    tokens = dict(summary.usage)
    metadata = metadata or {}
    return AccountingSummary(
        wall_time_seconds=wall_time_seconds,
        wall_time_source=wall_time_source,
        wall_clock_limit_seconds=metadata.get("wall_clock_limit_seconds")
        if isinstance(metadata.get("wall_clock_limit_seconds"), int)
        else None,
        codex_exit_code=metadata.get("codex_exit_code") if isinstance(metadata.get("codex_exit_code"), int) else None,
        validator_call_limit=metadata.get("validator_call_limit")
        if isinstance(metadata.get("validator_call_limit"), int)
        else None,
        validator_call_count=metadata.get("validator_call_count")
        if isinstance(metadata.get("validator_call_count"), int)
        else 0,
        turns=summary.completed_turns,
        tool_calls=summary.command_count + summary.file_change_count,
        command_executions=summary.command_count,
        file_changes=summary.file_change_count,
        agent_messages=summary.agent_message_count,
        tokens=tokens,
        total_tokens=int(tokens.get("input_tokens", 0)) + int(tokens.get("output_tokens", 0)),
        same_cost_mode=bool(metadata.get("same_cost_mode")),
        simulated_cost_unit=metadata.get("simulated_cost_unit")
        if isinstance(metadata.get("simulated_cost_unit"), str)
        else ("api_usd" if isinstance(metadata.get("simulated_cost_limit_usd"), (int, float)) else None),
        simulated_cost_limit=metadata.get("simulated_cost_limit")
        if isinstance(metadata.get("simulated_cost_limit"), (int, float))
        else metadata.get("simulated_cost_limit_usd")
        if isinstance(metadata.get("simulated_cost_limit_usd"), (int, float))
        else None,
        simulated_cost=metadata.get("simulated_cost")
        if isinstance(metadata.get("simulated_cost"), (int, float))
        else metadata.get("simulated_cost_usd")
        if isinstance(metadata.get("simulated_cost_usd"), (int, float))
        else None,
        simulated_cost_selected=metadata.get("simulated_cost_selected")
        if isinstance(metadata.get("simulated_cost_selected"), (int, float))
        else metadata.get("simulated_cost_selected_usd")
        if isinstance(metadata.get("simulated_cost_selected_usd"), (int, float))
        else None,
        simulated_cost_limit_usd=metadata.get("simulated_cost_limit_usd")
        if isinstance(metadata.get("simulated_cost_limit_usd"), (int, float))
        else None,
        simulated_cost_usd=metadata.get("simulated_cost_usd")
        if isinstance(metadata.get("simulated_cost_usd"), (int, float))
        else None,
        simulated_cost_selected_usd=metadata.get("simulated_cost_selected_usd")
        if isinstance(metadata.get("simulated_cost_selected_usd"), (int, float))
        else None,
        simulated_cost_exceeded=bool(metadata.get("simulated_cost_exceeded")),
        checkpoint_count=metadata.get("checkpoint_count") if isinstance(metadata.get("checkpoint_count"), int) else 0,
        selected_checkpoint=metadata.get("selected_checkpoint")
        if isinstance(metadata.get("selected_checkpoint"), str)
        else None,
        selected_turn=metadata.get("selected_turn") if isinstance(metadata.get("selected_turn"), int) else None,
        selected_validator_call_count=metadata.get("selected_validator_call_count")
        if isinstance(metadata.get("selected_validator_call_count"), int)
        else None,
    )


def _sum_accounting(accounting: Iterable[AccountingSummary]) -> AccountingSummary:
    items = list(accounting)
    tokens: Counter[str] = Counter()
    wall_time = 0.0
    have_wall_time = False
    sources: set[str] = set()
    wall_clock_limits = [item.wall_clock_limit_seconds for item in items if item.wall_clock_limit_seconds is not None]
    codex_exit_codes = [item.codex_exit_code for item in items if item.codex_exit_code is not None]
    validator_call_limits = [item.validator_call_limit for item in items if item.validator_call_limit is not None]
    simulated_cost_limits = [
        item.simulated_cost_limit for item in items if item.simulated_cost_limit is not None
    ]
    simulated_costs = [item.simulated_cost for item in items if item.simulated_cost is not None]
    simulated_selected_costs = [
        item.simulated_cost_selected for item in items if item.simulated_cost_selected is not None
    ]
    simulated_cost_units = {item.simulated_cost_unit for item in items if item.simulated_cost_unit is not None}
    for item in items:
        tokens.update(item.tokens)
        if item.wall_time_seconds is not None:
            wall_time += item.wall_time_seconds
            have_wall_time = True
        if item.wall_time_source:
            sources.add(item.wall_time_source)
    return AccountingSummary(
        wall_time_seconds=wall_time if have_wall_time else None,
        wall_time_source="+".join(sorted(sources)) if sources else None,
        wall_clock_limit_seconds=sum(wall_clock_limits) if wall_clock_limits else None,
        codex_exit_code=codex_exit_codes[-1] if codex_exit_codes else None,
        validator_call_limit=sum(validator_call_limits) if validator_call_limits else None,
        validator_call_count=sum(item.validator_call_count for item in items),
        turns=sum(item.turns for item in items),
        tool_calls=sum(item.tool_calls for item in items),
        command_executions=sum(item.command_executions for item in items),
        file_changes=sum(item.file_changes for item in items),
        agent_messages=sum(item.agent_messages for item in items),
        tokens=dict(sorted(tokens.items())),
        total_tokens=sum(item.total_tokens for item in items),
        same_cost_mode=any(item.same_cost_mode for item in items),
        simulated_cost_unit=next(iter(simulated_cost_units)) if len(simulated_cost_units) == 1 else None,
        simulated_cost_limit=sum(simulated_cost_limits) if simulated_cost_limits else None,
        simulated_cost=sum(simulated_costs) if simulated_costs else None,
        simulated_cost_selected=sum(simulated_selected_costs) if simulated_selected_costs else None,
        simulated_cost_limit_usd=sum(
            item.simulated_cost_limit_usd for item in items if item.simulated_cost_limit_usd is not None
        )
        or None,
        simulated_cost_usd=sum(item.simulated_cost_usd for item in items if item.simulated_cost_usd is not None)
        or None,
        simulated_cost_selected_usd=sum(
            item.simulated_cost_selected_usd for item in items if item.simulated_cost_selected_usd is not None
        )
        or None,
        simulated_cost_exceeded=any(item.simulated_cost_exceeded for item in items),
        checkpoint_count=sum(item.checkpoint_count for item in items),
        selected_checkpoint=None,
        selected_turn=None,
        selected_validator_call_count=None,
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _attempt_wall_time(accounting_json: Path) -> tuple[float | None, str | None, dict[str, Any]]:
    data = _read_json(accounting_json)
    seconds = data.get("wall_time_seconds")
    if isinstance(seconds, (int, float)):
        return float(seconds), "attempt-metadata", data
    started = _parse_timestamp(data.get("started_at"))
    ended = _parse_timestamp(data.get("ended_at"))
    if started is not None and ended is not None:
        return max(0.0, (ended - started).total_seconds()), "attempt-metadata", data
    return None, None, data


def _file_observed_wall_time(files: Iterable[Path]) -> tuple[float | None, str | None]:
    starts: list[float] = []
    ends: list[float] = []
    for path in files:
        if not path.exists():
            continue
        stat = path.stat()
        birthtime = getattr(stat, "st_birthtime", None)
        if birthtime is not None:
            starts.append(float(birthtime))
        ends.append(float(stat.st_mtime))
    if not starts or not ends:
        return None, None
    seconds = max(ends) - min(starts)
    return max(0.0, seconds), "file-timestamps"


def _task_accounting(
    events_jsonl: Path | None,
    *,
    accounting_json: Path | None = None,
    observed_files: Iterable[Path] = (),
) -> AccountingSummary:
    summary = read_event_summary(events_jsonl) if events_jsonl is not None else EventSummary()
    wall_time_seconds = None
    wall_time_source = None
    metadata: dict[str, Any] = {}
    if accounting_json is not None and accounting_json.exists():
        wall_time_seconds, wall_time_source, metadata = _attempt_wall_time(accounting_json)
    if wall_time_seconds is None:
        wall_time_seconds, wall_time_source = _file_observed_wall_time(
            [p for p in [events_jsonl, *observed_files] if p is not None]
        )
    return _event_accounting(summary, wall_time_seconds, wall_time_source, metadata)


def read_event_summary(events_jsonl: Path) -> EventSummary:
    summary = EventSummary()
    if not events_jsonl.exists():
        return summary
    for line in events_jsonl.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        if event_type == "thread.started":
            summary.thread_id = event.get("thread_id")
        elif event_type == "turn.completed":
            summary.completed_turns += 1
            usage = event.get("usage")
            if isinstance(usage, dict):
                summary.usage = {k: int(v) for k, v in usage.items() if isinstance(v, int)}
        item = event.get("item")
        if not isinstance(item, dict):
            continue
        if event_type != "item.completed":
            continue
        item_type = item.get("type")
        if item_type == "command_execution":
            summary.command_count += 1
        elif item_type == "file_change":
            summary.file_change_count += 1
        elif item_type == "agent_message":
            summary.agent_message_count += 1
    return summary


def _load_instances_by_id() -> dict[str, dict[str, Any]]:
    return {i["instance_id"]: i for i in load_all_instances(include_tests=True)}


def summarize_eval_json(
    eval_json: Path,
    *,
    repo_root: Path | None = None,
    instances_by_id: dict[str, dict[str, Any]] | None = None,
    prompt_path: Path | None = None,
    final_path: Path | None = None,
    events_path: Path | None = None,
    accounting_path: Path | None = None,
) -> InstanceLabSummary:
    repo_root = repo_root or Path.cwd()
    instances_by_id = instances_by_id or _load_instances_by_id()
    instance_id = eval_json.parent.name
    result = EvaluationResult.model_validate_json(eval_json.read_text())
    raw_n_tests = len(result)
    inst = instances_by_id.get(instance_id)
    if inst is not None:
        active_branches = get_active_branches(inst)
        result = result.for_branches(active_branches).without_ignored(get_ignored_tests(inst))
    official = InstanceEvalSummary.from_eval_result(instance_id, result)
    status_counts = Counter(t.status for t in result.test_results)
    top_failures: list[dict[str, str]] = []
    for t in result.test_results:
        if t.status == "passed":
            continue
        message = str(t.extra.get("message") or t.extra.get("error_code") or "") if isinstance(t.extra, dict) else ""
        top_failures.append(
            {
                "name": t.name,
                "branch": t.branch,
                "status": t.status,
                "message": (message.splitlines() or [""])[0][:240],
            }
        )
        if len(top_failures) >= 8:
            break
    submission = eval_json.parent / "submission.tar.gz"
    accounting = _task_accounting(
        events_path,
        accounting_json=accounting_path,
        observed_files=[final_path, submission],
    )
    return InstanceLabSummary(
        instance_id=instance_id,
        score=official.score,
        score_percent=official.score * 100,
        n_resolved=official.n_resolved,
        n_tests=official.n_tests,
        raw_n_tests=raw_n_tests,
        status_counts=dict(sorted(status_counts.items())),
        error_code=official.error_code,
        n_warnings=official.n_warnings,
        n_system_errors=official.n_system_errors,
        test_branches=official.test_branches,
        executable_hash=result.executable_hash,
        eval_json=_rel(eval_json, repo_root),
        submission_tar=_rel(submission, repo_root) if submission.exists() else None,
        prompt_path=_rel(prompt_path, repo_root) if prompt_path is not None and prompt_path.exists() else None,
        final_path=_rel(final_path, repo_root) if final_path is not None and final_path.exists() else None,
        events_path=_rel(events_path, repo_root) if events_path is not None and events_path.exists() else None,
        accounting_path=_rel(accounting_path, repo_root) if accounting_path is not None and accounting_path.exists() else None,
        accounting=accounting,
        top_failures=top_failures,
    )


def _run_created_at(run_dir: Path, files: Iterable[Path]) -> str | None:
    mtimes = [p.stat().st_mtime for p in files if p.exists()]
    if not mtimes:
        return None
    return datetime.fromtimestamp(min(mtimes), tz=timezone.utc).isoformat(timespec="seconds")


def discover_run_dirs(runs_root: Path) -> list[Path]:
    if _looks_like_run_dir(runs_root):
        return [runs_root]
    if not runs_root.exists():
        return []
    return sorted(d for d in runs_root.iterdir() if d.is_dir() and _looks_like_run_dir(d))


def _looks_like_run_dir(path: Path) -> bool:
    return any(
        [
            (path / "run.json").exists(),
            (path / "events.jsonl").exists(),
            bool(list(path.glob("*/*.eval.json"))),
            bool(list(path.glob("*/submission.tar.gz"))),
        ]
    )


def _instance_artifact_path(run_dir: Path, dirname: str, instance_id: str, suffix: str = "") -> Path:
    return run_dir / dirname / f"{instance_id}{suffix}"


def summarize_run(
    run_dir: Path,
    *,
    repo_root: Path | None = None,
    instances_by_id: dict[str, dict[str, Any]] | None = None,
) -> RunLabSummary:
    repo_root = repo_root or Path.cwd()
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "run.json")
    instances_by_id = instances_by_id or _load_instances_by_id()
    eval_paths = sorted(run_dir.glob("*/*.eval.json"))
    top_level_events = run_dir / "events.jsonl"
    instances = []
    for p in eval_paths:
        instance_id = p.parent.name
        instance_events = _instance_artifact_path(run_dir, "events", instance_id, ".jsonl")
        if not instance_events.exists() and len(eval_paths) == 1 and top_level_events.exists():
            instance_events = top_level_events
        instances.append(
            summarize_eval_json(
                p,
                repo_root=repo_root,
                instances_by_id=instances_by_id,
                prompt_path=_instance_artifact_path(run_dir, "prompts", instance_id, ".txt"),
                final_path=_instance_artifact_path(run_dir, "finals", instance_id, ".txt"),
                events_path=instance_events,
                accounting_path=_instance_artifact_path(run_dir, "accounting", instance_id, ".json"),
            )
        )
    submitted_instances = len(list(run_dir.glob("*/submission.tar.gz")))
    prompt = run_dir / "prompt.txt"
    final = run_dir / "final.txt"
    events = top_level_events
    scores = [i.score for i in instances]
    total_tests = sum(i.n_tests for i in instances)
    total_resolved = sum(i.n_resolved for i in instances)
    total_failures = sum(i.status_counts.get("failure", 0) for i in instances)
    total_errors = sum(i.status_counts.get("error", 0) + i.status_counts.get("system_error", 0) for i in instances)
    created_at = manifest.get("created_at") or _run_created_at(
        run_dir,
        [run_dir / "run.json", prompt, final, events, *eval_paths],
    )
    return RunLabSummary(
        run_id=str(manifest.get("run_id") or run_dir.name),
        run_dir=_rel(run_dir, repo_root),
        created_at=str(created_at) if created_at else None,
        experiment=_manifest_experiment(manifest),
        label=manifest.get("label"),
        agent=manifest.get("agent"),
        model=manifest.get("model"),
        reasoning_effort=manifest.get("reasoning_effort"),
        codex_version=manifest.get("codex_version"),
        harness_git_sha=manifest.get("harness_git_sha"),
        docker_backend=manifest.get("docker_backend"),
        docker_platform=manifest.get("docker_platform"),
        harness_mode=manifest.get("harness_mode"),
        validator_access=manifest.get("validator_access"),
        notes=manifest.get("notes"),
        prompt_path=_rel(prompt, repo_root) if prompt.exists() else None,
        final_path=_rel(final, repo_root) if final.exists() else None,
        events_path=_rel(events, repo_root) if events.exists() else None,
        manifest_path=_rel(run_dir / "run.json", repo_root) if (run_dir / "run.json").exists() else None,
        event_summary=read_event_summary(events),
        accounting=_sum_accounting([i.accounting for i in instances])
        if instances
        else _task_accounting(events, observed_files=[final]),
        submitted_instances=submitted_instances,
        evaluated_instances=len(instances),
        average_score=statistics.fmean(scores) if scores else 0.0,
        average_score_percent=(statistics.fmean(scores) * 100) if scores else 0.0,
        total_resolved=total_resolved,
        total_tests=total_tests,
        raw_total_tests=sum(i.raw_n_tests for i in instances),
        total_failures=total_failures,
        total_errors=total_errors,
        instances=instances,
    )


def _experiment_key(run: RunLabSummary) -> str:
    return run.experiment.name or run.label or run.run_id


def summarize_experiments(runs: Iterable[RunLabSummary]) -> list[ExperimentLabSummary]:
    groups: dict[str, list[RunLabSummary]] = {}
    for run in runs:
        groups.setdefault(_experiment_key(run), []).append(run)
    experiments: list[ExperimentLabSummary] = []
    for name, grouped_runs in groups.items():
        scores = [run.average_score for run in grouped_runs if run.evaluated_instances]
        descriptions = [
            run.experiment.description for run in grouped_runs if run.experiment.description
        ]
        tags = sorted({tag for run in grouped_runs for tag in run.experiment.tags})
        factors: dict[str, set[str]] = {}
        for run in grouped_runs:
            for key, value in run.experiment.factors.items():
                factors.setdefault(key, set()).add(value)
        experiments.append(
            ExperimentLabSummary(
                name=name,
                description=descriptions[0] if descriptions else None,
                tags=tags,
                factors={key: ", ".join(sorted(values)) for key, values in sorted(factors.items())},
                run_count=len(grouped_runs),
                evaluated_runs=sum(1 for run in grouped_runs if run.evaluated_instances),
                submitted_instances=sum(run.submitted_instances for run in grouped_runs),
                evaluated_instances=sum(run.evaluated_instances for run in grouped_runs),
                average_score=statistics.fmean(scores) if scores else 0.0,
                average_score_percent=(statistics.fmean(scores) * 100) if scores else 0.0,
                total_resolved=sum(run.total_resolved for run in grouped_runs),
                total_tests=sum(run.total_tests for run in grouped_runs),
                total_failures=sum(run.total_failures for run in grouped_runs),
                total_errors=sum(run.total_errors for run in grouped_runs),
                accounting=_sum_accounting(run.accounting for run in grouped_runs),
                run_ids=[run.run_id for run in sorted(grouped_runs, key=lambda item: item.run_id)],
            )
        )
    experiments.sort(key=lambda exp: (exp.average_score, exp.name), reverse=True)
    return experiments


def build_index(runs_root: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or Path.cwd()
    instances_by_id = _load_instances_by_id()
    runs = [
        summarize_run(run_dir, repo_root=repo_root, instances_by_id=instances_by_id)
        for run_dir in discover_run_dirs(runs_root)
    ]
    runs.sort(key=lambda r: (r.created_at or "", r.run_id), reverse=True)
    scores = [r.average_score for r in runs if r.evaluated_instances]
    experiments = summarize_experiments(runs)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs_root": _rel(runs_root, repo_root),
        "total_runs": len(runs),
        "evaluated_runs": sum(1 for r in runs if r.evaluated_instances),
        "total_experiments": len(experiments),
        "average_score": statistics.fmean(scores) if scores else 0.0,
        "average_score_percent": (statistics.fmean(scores) * 100) if scores else 0.0,
        "experiments": [experiment.model_dump() for experiment in experiments],
        "runs": [r.model_dump() for r in runs],
    }


def write_index(runs_root: Path, output: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    index = build_index(runs_root, repo_root=repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


def _fmt_score(score: float) -> str:
    if math.isclose(score, 1.0):
        return "solved"
    return f"{score * 100:.0f}"


def _fmt_duration(seconds: Any) -> str:
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


def _fmt_tokens(value: Any) -> str:
    if not isinstance(value, int):
        return ""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def render_html_report(index: dict[str, Any]) -> str:
    experiment_rows = []
    for experiment in index.get("experiments", []):
        accounting = experiment.get("accounting") or {}
        tags = ", ".join(experiment.get("tags") or [])
        factor_text = ", ".join(
            f"{key}: {value}" for key, value in (experiment.get("factors") or {}).items()
        )
        experiment_rows.append(
            "<tr>"
            f"<td>{escape(str(experiment.get('name') or ''))}"
            f"<br><span class='muted'>{escape(str(experiment.get('description') or ''))}</span></td>"
            f"<td class='num'>{experiment.get('evaluated_runs', 0)}/{experiment.get('run_count', 0)}</td>"
            f"<td class='num'>{_fmt_score(float(experiment.get('average_score') or 0))}</td>"
            f"<td class='num'>{experiment.get('total_resolved', 0)}/{experiment.get('total_tests', 0)}</td>"
            f"<td>{escape(tags)}"
            f"<br><span class='muted'>{escape(factor_text)}</span></td>"
            f"<td>{escape(_fmt_duration(accounting.get('wall_time_seconds')))}"
            f"<br><span class='muted'>{accounting.get('turns', 0)} turns, "
            f"{accounting.get('tool_calls', 0)} tools, "
            f"{escape(_fmt_tokens(accounting.get('total_tokens')))}</span></td>"
            "</tr>"
        )
    rows = []
    details = []
    for run in index.get("runs", []):
        accounting = run.get("accounting") or {}
        experiment = run.get("experiment") or {}
        rows.append(
            "<tr>"
            f"<td>{escape(str(experiment.get('name') or run.get('label') or run['run_id']))}</td>"
            f"<td><a href='#{escape(run['run_id'])}'>{escape(run['run_id'])}</a></td>"
            f"<td>{escape(str(run.get('model') or ''))}</td>"
            f"<td>{escape(str(run.get('reasoning_effort') or ''))}</td>"
            f"<td>{run.get('evaluated_instances', 0)}/{run.get('submitted_instances', 0)}</td>"
            f"<td class='num'>{_fmt_score(float(run.get('average_score') or 0))}</td>"
            f"<td class='num'>{run.get('total_resolved', 0)}/{run.get('total_tests', 0)}</td>"
            f"<td>{escape(_fmt_duration(accounting.get('wall_time_seconds')))}"
            f"<br><span class='muted'>{accounting.get('turns', 0)} turns, "
            f"{accounting.get('tool_calls', 0)} tools, "
            f"{escape(_fmt_tokens(accounting.get('total_tokens')))}</span></td>"
            f"<td>{escape(str(run.get('created_at') or ''))}</td>"
            "</tr>"
        )
        inst_rows = []
        failure_blocks = []
        for inst in run.get("instances", []):
            inst_accounting = inst.get("accounting") or {}
            inst_rows.append(
                "<tr>"
                f"<td>{escape(inst['instance_id'])}</td>"
                f"<td class='num'>{_fmt_score(float(inst.get('score') or 0))}</td>"
                f"<td class='num'>{inst.get('n_resolved', 0)}/{inst.get('n_tests', 0)}</td>"
                f"<td class='num'>{escape(_fmt_duration(inst_accounting.get('wall_time_seconds')))}</td>"
                f"<td class='num'>{inst_accounting.get('turns', 0)}</td>"
                f"<td class='num'>{inst_accounting.get('tool_calls', 0)}</td>"
                f"<td class='num'>{escape(_fmt_tokens(inst_accounting.get('total_tokens')))}</td>"
                f"<td>{escape(', '.join(f'{k}:{v}' for k, v in inst.get('status_counts', {}).items()))}</td>"
                f"<td>{escape(str(inst.get('error_code') or ''))}</td>"
                "</tr>"
            )
            failures = inst.get("top_failures") or []
            if failures:
                items = "".join(
                    f"<li><code>{escape(f['name'])}</code> "
                    f"<span class='muted'>{escape(f.get('status', ''))}</span>"
                    f"<br><span>{escape(f.get('message', ''))}</span></li>"
                    for f in failures
                )
                failure_blocks.append(f"<h4>{escape(inst['instance_id'])}</h4><ol>{items}</ol>")
        links = []
        for label, key in [
            ("manifest", "manifest_path"),
            ("prompt", "prompt_path"),
            ("final", "final_path"),
            ("events", "events_path"),
        ]:
            if run.get(key):
                links.append(f"<a href='../{escape(run[key])}'>{label}</a>")
        tokens = accounting.get("tokens") or {}
        token_text = ", ".join(f"{k}: {v:,}" for k, v in tokens.items()) or "none"
        cost_text = ""
        if accounting.get("same_cost_mode"):
            selected_cost = accounting.get("simulated_cost_selected")
            total_cost = accounting.get("simulated_cost")
            limit_cost = accounting.get("simulated_cost_limit")
            unit = str(accounting.get("simulated_cost_unit") or "cost units")
            if unit == "api_usd":
                fmt_cost = lambda value: f"${value:.4f}"
            elif unit == "codex_credits":
                fmt_cost = lambda value: f"{value:.2f} credits"
            else:
                fmt_cost = lambda value: f"{value:.4f} {unit}"
            cost_bits = []
            if isinstance(selected_cost, (int, float)):
                cost_bits.append(f"selected {fmt_cost(selected_cost)}")
            if isinstance(total_cost, (int, float)):
                cost_bits.append(f"metered {fmt_cost(total_cost)}")
            if isinstance(limit_cost, (int, float)):
                cost_bits.append(f"limit {fmt_cost(limit_cost)}")
            if accounting.get("checkpoint_count"):
                cost_bits.append(f"{accounting.get('checkpoint_count')} checkpoint(s)")
            if cost_bits:
                cost_text = "<br><span class='muted'>same-cost: " + escape(", ".join(cost_bits)) + "</span>"
        validator_text = ""
        if accounting.get("validator_call_limit"):
            validator_text = (
                f", {accounting.get('validator_call_count', 0)}/"
                f"{accounting.get('validator_call_limit')} validator calls"
            )
        tags = ", ".join(experiment.get("tags") or [])
        factor_text = ", ".join(
            f"{key}: {value}" for key, value in (experiment.get("factors") or {}).items()
        )
        details.append(
            f"<section id='{escape(run['run_id'])}'>"
            f"<h2>{escape(run['run_id'])}</h2>"
            f"<p><b>Experiment:</b> {escape(str(experiment.get('name') or run.get('label') or run['run_id']))}</p>"
            f"<p>{escape(str(experiment.get('description') or run.get('notes') or ''))}</p>"
            f"<p class='muted'>{escape(tags)}"
            f"{'<br>' if tags and factor_text else ''}{escape(factor_text)}</p>"
            f"<p>{escape(str(run.get('notes') or ''))}</p>"
            f"<p><b>Harness:</b> {escape(str(run.get('harness_mode') or ''))}"
            f"{' / ' + escape(str(run.get('validator_access'))) if run.get('validator_access') else ''}</p>"
            f"<p><b>Accounting:</b> {escape(_fmt_duration(accounting.get('wall_time_seconds')))}"
            f" wall time ({escape(str(accounting.get('wall_time_source') or 'unknown'))}), "
            f"{accounting.get('turns', 0)} turns, {accounting.get('tool_calls', 0)} tool calls, "
            f"{escape(_fmt_tokens(accounting.get('total_tokens')))} total tokens."
            f"{validator_text}"
            f"<br><span class='muted'>{escape(token_text)}</span>{cost_text}</p>"
            f"<p class='muted'>{' | '.join(links)}</p>"
            "<table><thead><tr><th>Instance</th><th>Score</th><th>Passed</th><th>Time</th><th>Turns</th>"
            "<th>Tools</th><th>Tokens</th><th>Status Counts</th><th>Error</th></tr></thead>"
            f"<tbody>{''.join(inst_rows)}</tbody></table>"
            f"<div class='failures'>{''.join(failure_blocks)}</div>"
            "</section>"
        )
    css = """
    body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #17202a; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0 32px; }
    th, td { border-bottom: 1px solid #d6dde5; padding: 7px 9px; text-align: left; vertical-align: top; }
    th { background: #f3f6f9; font-weight: 650; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    .muted { color: #667085; }
    code { background: #f4f4f5; padding: 1px 4px; border-radius: 4px; }
    section { margin-top: 36px; }
    ol { padding-left: 24px; }
    li { margin: 8px 0; }
    """
    return (
        "<!doctype html><meta charset='utf-8'>"
        "<title>ProgramBench Lab Report</title>"
        f"<style>{css}</style>"
        "<h1>ProgramBench Lab Report</h1>"
        f"<p class='muted'>Generated at {escape(str(index.get('generated_at', '')))} from "
        f"{escape(str(index.get('runs_root', '')))}.</p>"
        "<h2>Experiments</h2>"
        "<table><thead><tr><th>Experiment</th><th>Runs</th><th>Score</th><th>Passed</th>"
        "<th>Factors</th><th>Accounting</th></tr></thead>"
        f"<tbody>{''.join(experiment_rows)}</tbody></table>"
        "<h2>Runs</h2>"
        "<table><thead><tr><th>Experiment</th><th>Run</th><th>Model</th><th>Reasoning</th><th>Eval/Sub</th>"
        "<th>Score</th><th>Passed</th><th>Accounting</th><th>Created</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"{''.join(details)}"
    )


def write_html_report(runs_root: Path, output: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    index = build_index(runs_root, repo_root=repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html_report(index))
    return index
