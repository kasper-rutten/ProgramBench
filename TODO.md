# TODO: Publishability And Portability Plan

This repo is starting to look useful as a general "run your own ProgramBench-style
benchmark" tool, but the current implementation has two different layers mixed
together:

- A mostly portable ProgramBench evaluator/reporting layer.
- A very specific Codex CLI + ChatGPT subscription + macOS/Colima runner layer.

Before presenting this as a general-purpose project, split those concepts
cleanly and make the Codex/Colima path a documented adapter rather than the
shape of the whole tool.

## Current Portability Assessment

The benchmark and evaluation core is not inherently MacBook-specific or
Codex-specific.

Portable pieces:

- `programbench eval` is ordinary Python plus Docker.
- The evaluator can already use a custom Docker executable through
  `PROGRAMBENCH_DOCKER_EXECUTABLE`.
- Task data, scoring, summaries, and eval JSON artifacts are host-neutral.
- `programbench lab` and the static HTML report/indexer are mostly generic local
  artifact tooling.
- Run accounting is conceptually agent-neutral: wall clock, turns, tool calls,
  tokens, validator calls, and score all make sense across runners.

Currently Codex-specific pieces:

- `scripts/run-codex-blackbox-task` and `scripts/run-codex-validator-task` are
  Codex CLI harnesses, not generic agent runners.
- The scripts explicitly refuse to run when `OPENAI_API_KEY` is set, because the
  intended path is Codex CLI logged in with a ChatGPT plan.
- Token, turn, and tool-call accounting is parsed from Codex JSONL event logs.
- Fixed-budget mode uses Codex credit assumptions and Codex model names.
- The egress-control default relies on Codex CLI `workspace-write` behavior plus
  a repo-local filesystem oracle broker.
- The runners assume Codex command syntax and reasoning-effort flags.

Currently macOS/Colima-specific pieces:

- `CODEX_BIN` defaults to `/Applications/Codex.app/Contents/Resources/codex`.
- `scripts/docker-colima` defaults Docker to
  `unix://$HOME/.colima/default/docker.sock`.
- Colima setup/gating is encoded in `scripts/gate-colima-docker`,
  `scripts/prepare-colima-docker-config`, and the Colima docs.
- Several docs are written as "ProgramBench with Codex CLI and Colima" rather
  than "ProgramBench with pluggable runners and Docker backends."

This is acceptable as a strong reference runner, but not as the public shape of
the whole project.

## Product Shape To Aim For

The clean public framing should be:

1. `programbench eval`
   - Official evaluator.
   - Docker-based.
   - Agent-neutral.
   - Works with submitted `submission.tar.gz` artifacts regardless of who made
     them.

2. `programbench lab`
   - Local run browser and report generator.
   - Agent-neutral metadata model.
   - Displays model, runner, harness mode, wall clock, turns, tool calls, token
     usage, validator calls, scores, and artifact links.

3. `programbench run codex`
   - Codex CLI adapter.
   - Supports ChatGPT-plan usage as a first-class "subscription escape hatch."
   - Supports black-box and validator-visible modes.
   - Supports fixed wall-clock and fixed simulated-cost modes.
   - Owns Codex-specific JSONL parsing and Codex credit math.

4. Future runner adapters
   - `programbench run openai-api`
   - `programbench run aider`
   - `programbench run swe-agent`
   - `programbench run shell`
   - Possibly a minimal adapter contract for external agents that just says:
     "Here is a workspace and task prompt; produce `submission.tar.gz`."

5. Docker backend presets
   - `docker`
   - `colima`
   - `podman`, if feasible
   - Optional explicit Docker executable/socket/config flags.

## Refactor Tasks

Separate runner abstraction from evaluator/reporting:

- Introduce a generic run manifest schema that does not assume Codex.
- Keep fields like `agent`, `runner`, `model`, `reasoning_effort`,
  `harness_mode`, `validator_access`, `wall_clock_limit_seconds`,
  `solver_network`, and `docker_backend`.
- Move Codex-specific event parsing behind a Codex adapter module.
- Ensure lab indexing gracefully handles non-Codex runs with missing token or
  tool-call fields.

Turn the shell runners into CLI commands or clearly documented examples:

- Prefer `programbench run codex blackbox ...` and
  `programbench run codex validator ...` over root-level script entry points.
- Keep scripts as thin wrappers if useful for local hacking.
- Add `--codex-bin`, `--docker-executable`, `--docker-backend`,
  `--egress-mode`, and `--allow-unguarded-solver` flags.
- Default to portable `codex` on `PATH`, then special-case the macOS app bundle
  only as a fallback.

Make Docker backend configuration explicit:

- Keep Colima support, but do not make it the assumed backend.
- Document `PROGRAMBENCH_DOCKER_EXECUTABLE` as the lowest-level escape hatch.
- Add a small backend probe command that reports Docker host, platform, CPU
  count, and whether `linux/amd64` containers can run.
- Avoid baking Docker Desktop, Colima, or private GHCR workflows into generic
  docs.

Clarify egress-control tiers:

- Tier 0: no guard, explicitly unsafe, useful only for debugging.
- Tier 1: Codex CLI `workspace-write` plus filesystem oracle broker. This is the
  current repo-local default and works well on this machine.
- Tier 2: host firewall or dedicated OS user that allows Codex/OpenAI traffic but
  blocks arbitrary solver egress.
- Tier 3: fully containerized/VM-isolated runner with explicit network policy.

For public docs, describe Tier 1 as convenient and reproducible enough for local
experiments, not as a formal security boundary.

Make validator-visible mode generic:

- Treat reference and validator access as "oracles" with well-defined IPC.
- Keep host-side validator implementation outside solver-writable paths.
- Exclude oracle helpers, validator call counters, and broker files from
  submissions.
- Record validator-call progression in structured artifacts so reports can show
  score over calls.

Clarify billing modes:

- Codex subscription mode should be one adapter behavior, not the whole project.
- API-billed OpenAI mode should be possible later with explicit API usage
  accounting.
- Same-cost experiments should use a generic cost model interface, with Codex
  credits and API USD as two rate-card implementations.

## Documentation Tasks

Write a public README structure like:

- What this tool does.
- Quick start: evaluate an existing submission.
- Quick start: browse local runs.
- Runner adapters overview.
- Codex CLI runner: ChatGPT-plan path.
- Docker backend setup.
- Egress and isolation model.
- Reproducing the gron smoke/matrix experiment.

Split current docs:

- Keep `docs/codex-colima-reproduction.md` as a specific recipe.
- Add `docs/runners.md` for the adapter model.
- Add `docs/docker-backends.md` for Docker/Colima/Podman setup.
- Add `docs/egress-control.md` for isolation tiers and caveats.
- Keep `docs/lab.md`, but make it less Codex-assumptive.

## Release Readiness Checklist

- `programbench eval` works on Linux with normal Docker.
- `programbench eval` works on macOS with Colima.
- Codex runner works when `codex` is on `PATH`.
- Codex runner works with the macOS app-bundle fallback.
- Lab report can index both Codex and non-Codex runs.
- Submission tarballs are verified to exclude oracle/helper files.
- Validator-visible mode stores per-call score progression.
- Egress guard defaults are documented and fail closed.
- Tests cover lab indexing, manifest parsing, and basic runner artifact layout.
- README clearly says what is benchmark-core versus adapter-specific.

## Current Best Framing

If published before the full refactor, describe the repo as:

"A local ProgramBench evaluator and run browser, plus an experimental Codex CLI
runner for using a ChatGPT Codex subscription to attempt black-box and
validator-visible reverse-engineering tasks on a Docker-backed benchmark."

Avoid claiming it is already a fully general multi-agent benchmark harness until
the runner abstraction exists.
