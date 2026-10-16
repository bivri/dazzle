# Experiments

An experiment answers one pre-registered question by measurement: a new teacher objective, a
pattern parameterization, a training recipe, a transformation filter. It is developed in its own
worktree and branch so `main` only receives the record and any code that earned its place.

## Start

```bash
herdr worktree create --cwd <repository checkout> --branch exp/<yyyymmdd>-<slug> --base main \
    --label <slug> --no-focus
```

Herdr checks the branch out under `~/.herdr/worktrees/dazzle/exp-<yyyymmdd>-<slug>` and opens it
as a workspace linked to the repository's; the response gives the path
(`.result.root_pane.cwd`) and the workspace id. Without `--base` the same command opens an
existing `exp/` branch. Do not create experiment worktrees with `git worktree add`: Herdr does not
list them as workspaces.

In the worktree create `experiments/<yyyy-mm-dd>-<slug>/README.md` from the template in
`experiments/README.md` and fill **Question**, **Conditions** and **Decision rule** before the first
run. Commit it. The decision rule names the number that would make you adopt, reject or extend
the idea, e.g. "adopt if 4-style mean E/F ≥ 0.25 with CI lower bound > 0 at LPIPS ≤ 0.08".

## Build

- Experiment-only code lives in `experiments/<date>-<slug>/` and imports `dazzle`. File names say
  what the code does (`optimize_per_image.py`, `screen_objectives.py`).
- Code that should outlive the experiment goes into `src/dazzle/` on the same branch with tests,
  so the record and the library change land together.
- Architecture and vocabulary rules apply here as everywhere (`scripts/check_policy.py`).

## Run

- Ship a commit with `scripts/sync_checkout.sh` and run it as described in `docs/agents/compute.md`.
- Name runs `<slug>-<condition>`; every run writes `runs/<run-id>/manifest.json` with the code
  version.
- Screen many candidates with the proxy (`--proxy-only`), then train LoRAs only for the best few.
  The proxy ranks; only LoRA E/F is a result.
- Keep failed and negative runs in the record. A run stopped early is reported as stopped.

## Conclude

1. Copy each run's `report.json` and `manifest.json` into the record directory.
2. Fill **Results** (a table with LPIPS and E/F mean and CI per condition, with the run ids) and
   **Conclusion** (which decision rule fired, what it changes).
3. Open a pull request from `exp/<date>-<slug>` to `main`. Throwaway code can be dropped from the
   branch before merging; the record and any `src/` change stay.
4. Remove the worktree: `herdr worktree remove --workspace <workspace id>`.
