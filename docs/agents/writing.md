# Writing documents

| Document | Carries | Does not carry |
| --- | --- | --- |
| `AGENTS.md` (root ≤ 150 lines, nested ≤ 100) | rules and paths | how a rule was reached |
| `CONTEXT.md` | each term, its meaning, the forms to avoid | dates, status, decisions |
| `docs/adr/NNNN-<title>.md` | one decision: context, decision, consequences | progress |
| `docs/decisions-log/<topic>.md` | dated observations that justified or overturned a rule, newest first | rules |
| `experiments/<date>-<slug>/README.md` | question, conditions, decision rule, results with run ids, conclusion | library documentation |
| `docs/*.md` for people | how the system works and how to use it, current state only | history of edits |

- Repository documents are in English; the repository will be public. Quote numbers with their
  run id or source file.
- Add every new document to `docs/index.md`. Pair a nested `AGENTS.md` with a one-line
  `CLAUDE.md` containing `@AGENTS.md`.
- A rule lives in one file; other files link to it.
- Inline comments state why the code exists and what breaks without it, in one to three lines.
- A document that records a past state opens with:
  `> Archived on YYYY-MM-DD. This document records a past state and is not current behaviour.`
