# Reference projects for a GitHub + Hugging Face Hub release

Question: which repository structure, tools, CI and Hub integration should a small single-model
library with an offline evaluation harness adopt? Sources were read on 2026-10-08 at the pinned
commits below; workflows were read, not executed.

## Findings

**Artifact boundary.** timm publishes `model.safetensors` + `config.json` + a model card per Hub
repo and loads them with a revision-aware loader; its MobileNetV3 example card is 2.5M parameters,
close to Dazzle's 1.95M
([timm `_hub.py`](https://raw.githubusercontent.com/huggingface/pytorch-image-models/83e6eb59b1cf2e0df321a4df4131baa16d745247/timm/models/_hub.py)).
`PyTorchModelHubMixin` produces the same layout from constructor arguments
([docs](https://huggingface.co/docs/huggingface_hub/package_reference/mixins)). LeRobot writes
the same files with its own mixin and uploads training checkpoints separately
([pretrained.py](https://raw.githubusercontent.com/huggingface/lerobot/ca69a2068462a37f7cdcb74180927a2f863d2bf7/src/lerobot/policies/pretrained.py)).

**Library vs research code.** SAM 2 keeps the reusable package apart from `training/`, `tools/`
and dataset evaluation, and its training model subclasses the library model instead of copying it
([tree](https://api.github.com/repos/facebookresearch/sam2/git/trees/2b90b9f5ceec907a1c18123530e92e794ad901a4?recursive=1)).
Depth Anything V2 duplicates its model package inside `metric_depth/`, the pattern to avoid
([tree](https://api.github.com/repos/DepthAnything/Depth-Anything-V2/git/trees/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf?recursive=1)).
stable-audio-tools splits inference, `train` and `ui` extras and strips training wrappers before
release ([pyproject](https://raw.githubusercontent.com/Stability-AI/stable-audio-tools/3241adba4fc2a85cf5b29d9eb68d42f40a28e820/pyproject.toml)).

**Versioning.** None of the inspected Hub model repos uses tags; Depth Anything links mutable
`main` downloads. `hf_hub_download` accepts a branch, tag or commit as `revision`
([download guide](https://huggingface.co/docs/huggingface_hub/guides/download)).

**Tooling.** LeRobot and PEFT pin Ruff in `.pre-commit-config.yaml`; LeRobot uses `uv.lock` with
`uv sync --locked` ([lerobot pyproject](https://raw.githubusercontent.com/huggingface/lerobot/ca69a2068462a37f7cdcb74180927a2f863d2bf7/pyproject.toml)).
Astral documents `astral-sh/setup-uv` with caching and `uv sync --locked`
([uv GitHub guide](https://docs.astral.sh/uv/guides/integration/github/)).

**CI cost and GPUs.** timm runs a 22-job CPU matrix and excludes large models on hosted runners
([tests.yml](https://raw.githubusercontent.com/huggingface/pytorch-image-models/83e6eb59b1cf2e0df321a4df4131baa16d745247/.github/workflows/tests.yml));
SAM 2 runs only a formatter. GitHub states self-hosted runners should almost never serve public
repositories ([secure use](https://docs.github.com/en/actions/reference/security/secure-use)) and
that full commit SHAs are the only immutable action reference.

**Releases.** Diffusers and LeRobot publish to PyPI from tags with `id-token: write` (trusted
publishing). The Hub supports fine-grained tokens and repository-scoped trusted publishers for
GitHub OIDC ([tokens](https://huggingface.co/docs/hub/security-tokens),
[trusted publishers](https://huggingface.co/docs/hub/trusted-publishers)). Space sync from GitHub
is documented with `huggingface/hub-sync`
([Spaces + Actions](https://huggingface.co/docs/hub/spaces-github-actions)); no inspected project
uses it.

**Agent docs.** timm, LeRobot, Diffusers and PEFT ship `AGENTS.md` and `CLAUDE.md`; timm's
`CLAUDE.md` is only `@AGENTS.md`
([timm CLAUDE.md](https://raw.githubusercontent.com/huggingface/pytorch-image-models/83e6eb59b1cf2e0df321a4df4131baa16d745247/CLAUDE.md)).
Claude Code reads `AGENTS.md` imports through `CLAUDE.md`
([memory docs](https://code.claude.com/docs/en/memory)).

**License.** Apache-2.0 is the standard permissive license across Hugging Face Hub model
repositories ([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)).

## Adopted

| Topic | Choice | Source pattern |
| --- | --- | --- |
| Layout | `src/dazzle` library with `training` and `evaluation` subpackages; `experiments/` records | SAM 2, stable-audio-tools |
| Weights | `PyTorchModelHubMixin`, safetensors + config + card, Hub tag = git tag | timm, HF docs |
| Tooling | uv + `uv.lock`, hatchling, Ruff, pytest, pre-commit config run with `prek` | LeRobot, PEFT, Astral |
| CI | hosted CPU tests with CPU PyTorch, SHA-pinned actions, build artifact; no self-hosted GPU runners | timm (scaled down), GitHub guidance |
| Release | tag → GitHub Release; PyPI trusted publishing wired but off; weights uploaded from the GPU machine by `scripts/publish_model.py` | Diffusers, LeRobot |
| Agent docs | short `AGENTS.md` + `CLAUDE.md` import, topic guides under `docs/agents/` | timm, Bivri's internal convention |

Not adopted yet: a demo Space, documentation site, type checker in CI, REUSE per-file headers.
