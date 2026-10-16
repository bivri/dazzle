# Testing and evidence

## What CI proves

CI (`.github/workflows/ci.yml`) runs on GitHub-hosted CPU runners: lock check, ruff, the policy
gate, and `pytest -m "not gpu"` with CPU PyTorch. Tests there cover contracts that hold without
models: generator output bound and shape, save/load round trip, native-size protection, the E/F
and proxy estimands. Owned GPU machines are not CI runners; GitHub advises against self-hosted
runners for public repositories.

## What needs a GPU run

Anything whose claim is about an effect: a training change, a teacher objective, a filter, the
evaluation harness itself. The evidence is a run on a GPU machine (`docs/agents/compute.md`)
reported as run id, code version from `manifest.json`, and the numbers from `report.json`.
Mark GPU-only tests `@pytest.mark.gpu`; they are run by hand on a GPU machine and their output is
quoted in the pull request.

## Writing tests

Test behaviour a user of the library or a reader of a report would notice: bounds, sizes,
round trips, estimand endpoints (no effect = 0, all style gain removed = 1), rejection by the
policy gate. A new check is tested on a case it must reject and one it must accept. Do not test
wiring, copies of constants, or that a function merely runs.

## Reporting

A local pass, a CPU test, a proxy screen and a LoRA evaluation are different evidence; name which
one supports each claim. Report what did not run and why.
