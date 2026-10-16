# GPU machines

Training, evaluation and per-image optimization run on GPU machines registered as Herdr machines.
`herdr machine list` is the inventory; this file does not repeat hostnames.

## Run through Herdr panes

1. `herdr --machine <label> workspace list`, then create your own tab:
   `herdr --machine <label> tab create --workspace <id> --label <experiment-slug> --cwd <workspace> --no-focus`.
   Work only in tabs and panes you created.
2. Run each long job in the **foreground** of its pane with `herdr --machine <label> pane run`.
   The pane is the job: read it with `pane read --source recent-unwrapped`, wait with
   `pane wait-output --regex`, stop it with `pane send-keys <pane> ctrl+c`. Background `&`
   subshells are invisible to the user and to the next agent.
3. Keep a second pane in the same tab for status checks so they do not interleave with job output.

## Code on the machine

`scripts/sync_checkout.sh <label> [<ref>]` copies a **committed** tree to
`$DAZZLE_WORKSPACE/checkouts/<sha12>` and prints the run command: source the workspace `env.sh`
(virtualenv on `PATH`, `HF_HOME` inside the workspace), then `PYTHONPATH=src` and `DAZZLE_GIT_SHA`.
Commit on the experiment branch first; a run from uncommitted code cannot be attributed.

The workspace layout (`data/styles`, `data/decoys`, `data/reference-renders`, `models/`,
`sd-scripts/`, `runs/`) is documented in `src/dazzle/workspace.py`. kohya-ss/sd-scripts is pinned
at commit `690ea7f96c23182352ec63def76d431c6120bd2f`.

## Sharing the GPU

- Check `nvidia-smi` before starting. Other services may hold memory on a machine; never stop a
  process you did not start.
- Run one GPU job per machine. Two concurrent per-image optimization jobs on one RTX 5090 each
  ran about 2.5x slower than one alone.
- A staging machine is shared with deployed services; long jobs on it are approved once per plan
  (`docs/agents/authority.md`).

## Bringing results back

Copy `runs/<run-id>/report.json` and `manifest.json` into the experiment record with `rsync`
(`experiments/README.md`). Images, LoRA weights and generator checkpoints stay on the machine.
