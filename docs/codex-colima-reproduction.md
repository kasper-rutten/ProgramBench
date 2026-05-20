# ProgramBench With Codex CLI And Colima

This repo-local setup avoids depending on `~/.docker/config.json`. That matters on
Colima hosts where the global Docker config may still point at Docker Desktop's
`osxkeychain` credential helper.

## Gates

Run the gates in this order.

```bash
./scripts/gate-colima-docker
uv run pytest tests/test_smoke.py tests/test_eval.py -q
./scripts/gate-calculator-eval
```

The calculator gate validates infrastructure: image build, submission archive
ownership, container lifecycle, compile, hash, test execution, and JUnit result
collection. The bundled toy fixture currently reports a warning because its
`tests.json` names omit the `eval.` package prefix emitted by pytest; that warning
is expected for this local gate.

The Docker wrapper used by the gates is:

```bash
./scripts/docker-colima
```

It sets:

- `DOCKER_CONFIG=$PWD/.docker-colima`
- `DOCKER_HOST=unix://$HOME/.colima/default/docker.sock`
- `DOCKER_DEFAULT_PLATFORM=linux/amd64`

ProgramBench can use the same wrapper through:

```bash
PROGRAMBENCH_DOCKER_EXECUTABLE="$PWD/scripts/docker-colima"
```

## Codex Plan Guard

The Codex smoke gate refuses to run if `OPENAI_API_KEY` is present or if
`codex login status` does not say `Logged in using ChatGPT`.

```bash
./scripts/gate-codex-calculator
```

That creates:

```text
runs/codex-calculator/testorg__calculator.abc1234/submission.tar.gz
```

Evaluate it with:

```bash
PROGRAMBENCH_DOCKER_EXECUTABLE="$PWD/scripts/docker-colima" \
PROGRAMBENCH_BLOB_DIR="$PWD/.programbench-blobs" \
uv run programbench eval runs/codex-calculator --filter 'testorg__calculator.abc1234' --force --docker-cpus 2
```

As with the gold calculator gate, expect the toy fixture to show the local
`tests.json`/JUnit naming warning even when the generated calculator behavior is
correct.

## Solver Network Guard

ProgramBench expects no general internet access during inference. The repo-local
Codex runners default to:

```bash
PROGRAMBENCH_SOLVER_EGRESS_MODE=openai-only
PROGRAMBENCH_SOLVER_EGRESS_GUARD=codex-workspace-write-filesystem-oracle
```

This uses Codex CLI `workspace-write` for solver commands, so arbitrary command
egress fails while Codex model traffic still works. Because the sandbox cannot
reach the Colima Docker socket, the visible helper scripts talk to a
filesystem broker; the harness side then performs the offline Docker oracle and
validator calls.

A dedicated macOS user plus packet-filter rules is still useful for a stronger
host-level variant: packet-filter rules can target that uid without disrupting
your normal browser, shells, or Docker preparation steps. A containerized solver
or another host-level policy is also fine if it lets Codex reach OpenAI while
preventing arbitrary `git`, `curl`, or package-manager egress. Use
`PROGRAMBENCH_SOLVER_EGRESS_GUARD_ACTIVE=1` only after such an external guard is
actually active.

For local harness debugging where contamination is acceptable, opt out
explicitly:

```bash
PROGRAMBENCH_SOLVER_EGRESS_MODE=off \
PROGRAMBENCH_ALLOW_UNGUARDED_SOLVER=1 \
scripts/run-codex-blackbox-task <instance_id> <run_dir> gpt-5.5 high 3600
```

## GHCR

For private GHCR access, keep credentials repo-local:

```bash
mkdir -p .docker-colima-ghcr
printf '%s' "$GHCR_TOKEN" | docker \
  --config "$PWD/.docker-colima-ghcr" \
  login ghcr.io \
  -u "$GITHUB_USER" \
  --password-stdin
```

Then run with:

```bash
PROGRAMBENCH_DOCKER_CONFIG="$PWD/.docker-colima-ghcr" \
PROGRAMBENCH_DOCKER_EXECUTABLE="$PWD/scripts/docker-colima" \
uv run programbench eval ...
```
