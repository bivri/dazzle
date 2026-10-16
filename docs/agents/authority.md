# Approval boundaries

| Operation | Judgment |
| --- | --- |
| Editing files, local commits, creating worktrees and `exp/` branches, the check/test gates | Autonomous |
| Proxy screening and short runs on a GPU machine in your own Herdr tab | Autonomous |
| Long GPU jobs (LoRA evaluations, generator training) on a shared staging machine | Approved once per plan |
| Push to a feature or `exp/` branch, opening or updating a pull request | Approved once per plan |
| Merge to `main` | Its own approval, after CI passes on the exact head commit |
| Pushing a `v*` tag (creates a GitHub Release) | Its own approval |
| Uploading weights or a model card to the Hub, creating or tagging a Hub repo | Its own approval |
| Changing visibility of the GitHub or Hub repository | Its own approval |
| Force-push, history rewrite, deleting a branch or a Hub revision | Its own approval |
| Stopping a process or closing a pane you did not start | Prohibited without the owner's request |
| `--no-verify` | Only when the user asks for it |

A review finding is evidence, not authority to write: confirm its premise from the code or a
reproduction before acting, and judge the proposed fix separately.

When recommending a course, name the observation that would overturn it and what undoing it costs
(a revert, a deleted Hub revision, an already-public release that cannot be recalled).
