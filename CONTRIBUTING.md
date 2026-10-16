# Contributing

## Setup

```bash
git clone https://github.com/bivri/dazzle && cd dazzle
uv sync --group dev            # or, without a GPU: uv venv && uv pip install --torch-backend cpu -e . --group dev
uvx prek install               # runs Ruff, file checks and the policy gate on each commit
```

Training and evaluation need the `train` / `eval` extras, a workspace with data and base models
(`src/dazzle/workspace.py`) and, for evaluation, a kohya-ss/sd-scripts checkout.

## Checks

```bash
uv run pytest -m "not gpu"
uvx ruff check && uvx ruff format --check
python scripts/check_policy.py
```

CI runs the same on every pull request. The policy gate enforces the single-head generator rule
([ADR 0002](docs/adr/0002-single-head-generator.md)) and the vocabulary in [CONTEXT.md](CONTEXT.md).

## Changes

- Branch from `main` (`feat/`, `fix/`, `docs/`, `exp/`) and open a pull request to `main`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- A change to generator behaviour, training or evaluation states the evaluation run that supports
  it: run id and numbers from its `report.json`.
- Experiments follow [docs/agents/experiments.md](docs/agents/experiments.md) and end with a record
  in `experiments/`.
- If an AI assistant wrote a substantial part of a change, say so in the pull request; the author
  remains responsible for every line.

## Releases

1. Bump `__version__` in `src/dazzle/__init__.py`, merge, and push tag `vX.Y.Z` on `main`.
   CI builds the package and creates the GitHub Release.
2. On the GPU machine holding the weights, evaluate them, then publish under the same tag:
   `python scripts/publish_model.py --model-dir <dir> --report <run>/report.json --tag vX.Y.Z`.
3. Add the printed Hub commit to the GitHub Release notes.

By contributing you agree that your contribution is licensed under Apache-2.0.
