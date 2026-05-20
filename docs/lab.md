# ProgramBench Local Lab

This repo includes a small local lab for recurring ProgramBench experiments.
It is intentionally layered:

- `programbench lab summary runs/` prints a compact terminal table.
- `programbench lab index runs/` writes a JSON index for other tools.
- `programbench lab report runs/` writes a static HTML app for browsing and
  comparing experiments, runs, instances, failures, and artifact links.

## Run Metadata

Add a `run.json` file to each run directory when you know the harness details:

```json
{
  "run_id": "codex-5.5-high-smoke",
  "created_at": "2026-05-13T15:00:00+00:00",
  "experiment": {
    "name": "Validator feedback ablation",
    "description": "Compare the same model and task with and without official validator output at test time.",
    "tags": ["gron", "validator-visible"],
    "factors": {
      "model": "gpt-5.5",
      "harness_mode": "validator-visible",
      "system_prompt": "iterate-until-validator-pass"
    }
  },
  "agent": "codex-cli",
  "model": "gpt-5.5",
  "reasoning_effort": "high",
  "codex_version": "codex-cli 0.130.0-alpha.5",
  "docker_backend": "colima",
  "docker_platform": "linux/amd64",
  "notes": "Three-task black-box benchmark smoke run"
}
```

The lab can still infer runs without this file, but model and harness fields
are much better when the manifest is present.

Experiments are first-class in the index and HTML report. A single experiment
can contain many runs that vary model, reasoning effort, harness access,
validator budget, or solver instructions. Agents can stamp new runs without a
UI by writing the `experiment` object above into `run.json`, or by setting
these environment variables when using the Codex helper scripts:

```bash
PROGRAMBENCH_EXPERIMENT_NAME="Validator feedback ablation" \
PROGRAMBENCH_EXPERIMENT_DESCRIPTION="Same task/model matrix with and without validator access." \
PROGRAMBENCH_EXPERIMENT_TAGS="gron,validator-visible" \
PROGRAMBENCH_EXPERIMENT_FACTORS='{"system_prompt":"iterate-until-validator-pass"}' \
PROGRAMBENCH_SYSTEM_PROMPT_LABEL="iterate-until-validator-pass" \
scripts/run-codex-validator-task tomnomnom__gron.88a6234 \
  runs/gron-validator-5.5-high gpt-5.5 high 3600 5
```

Use `PROGRAMBENCH_SYSTEM_PROMPT_APPEND` for experimental instruction deltas that
should be appended to the generated solver prompt. This is intentionally a low
ceremony hook for agent-driven experiments rather than a first-class prompt
editor.

## Accounting

Codex JSONL event logs are indexed into an `accounting` object at both the run
and instance level. It includes wall-clock time, completed turns, command/file
tool calls, and token usage. New runs created with
`scripts/run-codex-blackbox-task` and `scripts/run-codex-validator-task` also write
`accounting/<instance_id>.json` with explicit start/end timestamps. Older runs
fall back to observed local file timestamps when the platform exposes file
birth times.

## Commands

```bash
uv run programbench lab summary runs/
uv run programbench lab index runs/ --output runs/index.json
uv run programbench lab report runs/ --output reports/programbench-lab.html
```

The score calculation follows the same `EvaluationResult` / ignored-test logic
used by `programbench info`.

## Solver Egress Guard

The Codex runners are paper-faithful by default:

```bash
PROGRAMBENCH_SOLVER_EGRESS_MODE=openai-only
PROGRAMBENCH_SOLVER_EGRESS_GUARD=codex-workspace-write-filesystem-oracle
```

In that mode, Codex shell commands run under the CLI `workspace-write` sandbox,
which blocks command network access while leaving Codex itself able to talk to
OpenAI. The reference oracle and validator are exposed through a filesystem
broker in `.programbench-oracle`; the unsandboxed harness side performs the
offline Docker calls and returns stdout/stderr to the solver. The task Docker
containers still run with `--network none`.

If you prefer a host/container firewall instead, set
`PROGRAMBENCH_SOLVER_EGRESS_GUARD_ACTIVE=1` only after that external guard is
actually active.

For deliberately non-faithful local debugging, opt out explicitly:

```bash
PROGRAMBENCH_SOLVER_EGRESS_MODE=off \
PROGRAMBENCH_ALLOW_UNGUARDED_SOLVER=1 \
scripts/run-codex-blackbox-task <instance_id> <run_dir> gpt-5.5 high 3600
```

## Validator-Visible Runs

Use the validator-visible runner when you want the solving model to see official
validator feedback during the attempt:

```bash
scripts/run-codex-validator-task tomnomnom__gron.88a6234 \
  runs/validator-visible-5.5-high-gron gpt-5.5 high
```

Defaults are intentionally bounded but roomy: `3600` seconds wall-clock, `5`
validator calls, and a `300` second timeout for each validator call. Override
them positionally or with environment variables:

```bash
VALIDATOR_TIMEOUT_SECONDS=600 \
EVAL_WORKERS=2 EVAL_BRANCH_WORKERS=2 EVAL_DOCKER_CPUS=4 \
scripts/run-codex-validator-task <instance_id> <run_dir> gpt-5.5 high 3600 5
```

`programbench eval` can also evaluate existing run artifacts in parallel:

```bash
PROGRAMBENCH_DOCKER_EXECUTABLE="$PWD/scripts/docker-colima" \
uv run programbench eval runs/run-a runs/run-b --workers 2 --branch-workers 2 --docker-cpus 4 --force
```

On Colima, increase these conservatively: total active containers can be roughly
`workers * branch_workers`, and each receives `--docker-cpus`.

Validator feedback defaults to the top `24` failures with each failure message
capped at `4000` characters. Override with `VALIDATOR_FAILURE_LIMIT` and
`VALIDATOR_FAILURE_CHARS` when you want a smaller or larger oracle transcript.

For fixed-budget comparisons on Codex plans, set `SIMULATED_COST_LIMIT`.
The runner will use the Codex token-based credit rate card by model, including
`gpt-5.3-codex`, `gpt-5.4`, and `gpt-5.5`, and checkpoint/resume until the
next turn would exceed the simulated credit budget:

```bash
SIMULATED_COST_LIMIT=500 \
scripts/run-codex-validator-task tomnomnom__gron.88a6234 \
  runs/gron-fixed-budget-5.4-high gpt-5.4 high 3600

SIMULATED_COST_LIMIT=500 \
scripts/run-codex-validator-task tomnomnom__gron.88a6234 \
  runs/gron-fixed-budget-5.3-codex-high gpt-5.3-codex high 3600
```

Use `SIMULATED_COST_LIMIT_USD` instead when you want API-dollar simulation, or
override `SIMULATED_COST_*_PER_MTOK` directly for custom rate experiments.

Black-box runs accept the same wall-clock argument, without validator access:

```bash
scripts/run-codex-blackbox-task <instance_id> <run_dir> gpt-5.5 high 3600
```

## Web App

The generated report is a Python-produced static web app. It embeds the same
run metadata produced by `programbench lab index`, then uses a small vanilla
browser script for filtering and drilldown. There is no notebook runtime and no
JavaScript framework dependency.

```bash
uv run programbench lab report runs/ --output reports/programbench-lab.html
python3 -m http.server 8765
```

Then open `http://127.0.0.1:8765/reports/programbench-lab.html`.

The app is experiment-first: select an experiment, compare its runs, inspect a
run, and drill into instance scores, artifact paths, and top failure messages.
