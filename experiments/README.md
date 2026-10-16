# Experiment records

One directory per concluded experiment, named `<yyyy-mm-dd>-<slug>`, developed on the branch
`exp/<yyyymmdd>-<slug>` (`docs/agents/experiments.md`). A record lets a reader check every number
without re-running anything.

Each directory holds:

- `README.md` with the sections below, written before the first run and completed at the end;
- the `report.json` and `manifest.json` of every run it cites, renamed `<run-id>.report.json` and
  `<run-id>.manifest.json`;
- experiment-only code that produced them, if it was kept.

Images, LoRA weights and generator checkpoints are never committed; they stay in the workspace of
the machine that produced them.

## README template

```markdown
# <Title>

Branch: exp/<yyyymmdd>-<slug>. Status: running | concluded | abandoned.

## Question
## Conditions
## Decision rule
## Results
| Condition | Run id | LPIPS | E/F mean [95% CI] |
## Conclusion
```
