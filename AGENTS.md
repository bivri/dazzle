# Dazzle agent guide

Repository-wide rules for coding agents. `CLAUDE.md` imports this file. A nested `AGENTS.md` adds
rules for its subtree and does not repeat this one.

## Map

- `src/dazzle/`: the installable library. `generator.py` and `protect.py` are the shipped inference
  path; `training.py` trains the generator; `evaluation/` is the measurement harness
  (protected sets, transformation filters, mimicry LoRA, renders, E/F, proxy).
- `experiments/<yyyy-mm-dd>-<slug>/`: concluded experiment records. Read `experiments/README.md`
  before adding one.
- `docs/adr/`: settled decisions. `docs/decisions-log/`: observations that justified them.
  `docs/index.md` lists every document.
- `CONTEXT.md`: the domain language. Use its terms in code, prose and commit messages.
- `$DAZZLE_WORKSPACE` (default `~/dazzle-v2`): data, base models, run outputs. Layout in
  `src/dazzle/workspace.py`. Nothing from it enters git.

## Constraints

- The generator has one output, `delta = epsilon * tanh(G(x))`. Code computes no explicit spatial
  mask or routing head, uses LPIPS only as a scalar, and keeps no bank of stored patterns
  ([ADR 0002](docs/adr/0002-single-head-generator.md)). Perceptual shaping uses image-independent
  filters and scalar losses.
- Runtime code, weights and assets are Apache-2.0, MIT, BSD or CC0. Evaluation-only models
  (OpenRAIL SDXL checkpoints, CC-BY CSD) are downloaded at run time and never redistributed or
  used as training objectives. Non-commercial models are not used at all
  ([ADR 0001](docs/adr/0001-public-history-and-license.md)).
- The repository will be public. Internal planning, legal review and hostnames stay in the
  ignored `docs/internal/`, which may be absent in a fresh clone.
- CSD is the evaluator. It is never a loss term, and neither is anything tuned against the proxy.
- `scripts/check_policy.py` enforces the first and the vocabulary rule in pre-commit and CI.

## Commands

```bash
uv sync --group dev                        # GPU machine or full local environment
uv venv && uv pip install --torch-backend cpu -e . --group dev   # CPU-only, as CI does
uv run pytest -m "not gpu"
uvx ruff check && uvx ruff format --check
python scripts/check_policy.py
uvx prek install                           # once per clone: runs the above on commit
```

## Workflow

- Branch from `main`: `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `exp/<yyyymmdd>-<slug>`.
  Commits follow Conventional Commits: `<type>(<scope>): <description>`.
- Experiments run in their own worktree and branch: `docs/agents/experiments.md`.
- Names say what code does. Version and phase belong to git tags, run ids and experiment
  directories, never to file or module names (`train.py`, not `train_new.py`).
- A change to generator behaviour, training or the evaluation harness carries evidence: the run
  id and the `report.json` numbers it produced, or the reason no run was possible.
- Progress lives in agent todos. Plans and their grounds live in `docs/`; a settled decision gets
  an ADR.

## Agent guides

Read the one that governs the task; each owns its subject.

- `docs/agents/experiments.md`: setting up, running, recording and concluding an experiment.
- `docs/agents/compute.md`: the GPU machines, herdr panes, syncing code, workspace layout.
- `docs/agents/authority.md`: what needs the user's approval (push, merge, Hub, publication).
- `docs/agents/testing.md`: what CI covers, what needs a GPU run, what counts as evidence.
- `docs/agents/writing.md`: what instruction files, ADRs, decision logs and records carry.

## Asking a person

The harness owns the human-in-the-loop contract: resolve the channel with `herdr-hitl channel`
and ask through `herdr-hitl ask` when it answers `messenger`. A scheduled, unattended run records
the blocked point in its own output instead.
