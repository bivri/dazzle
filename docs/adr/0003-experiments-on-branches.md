# 0003. Experiments on their own branches and worktrees

Status: accepted, 2026-10-08; amended 2026-10-13 (worktrees are Herdr-managed)

## Context

Earlier research produced loose top-level scripts named by phase and iteration, copied by hand to
GPU machines. The history could not tell which code produced which number, and dead variants
accumulated next to live ones.

## Decision

- Each experiment is developed on `exp/<yyyymmdd>-<slug>` in a Herdr-managed worktree
  (`herdr worktree create`, `docs/agents/experiments.md`), so it opens as its own workspace and
  its GPU tabs and panes sit next to the code they run.
- `main` receives the experiment record (`experiments/<date>-<slug>/`) and any library code that
  earned adoption; throwaway code may be dropped before merging.
- Runs execute code shipped by commit (`scripts/sync_checkout.sh`); each run records its code
  version in `manifest.json`.
- File and module names describe function. Versions live in git tags, run ids and record names.

## Consequences

Every reported number traces to a run id, and every run id to a commit. Experiment code that
nobody adopted does not appear on `main`.
