#!/usr/bin/env python3
"""Publish a trained generator to the Hugging Face Hub under the release tag.

Run on the machine that holds the weights, after the evaluation run for exactly these weights:

    HF_TOKEN=... python scripts/publish_model.py --model-dir <save_pretrained dir> \\
        --report <workspace>/runs/<run-id>/report.json --tag v0.1.0 [--dry-run]

Uploads config.json, model.safetensors and a model card rendered from docs/hub-model-card.md,
then creates the Hub tag. The printed Hub commit goes into the GitHub release notes, so the code
tag, the weights revision and the evaluation report identify each other.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import dazzle
from dazzle.generator import DazzleGenerator

ROOT = Path(__file__).resolve().parents[1]


def results_table(report: dict) -> str:
    rows = ["| Training set given to the mimicry LoRA | LPIPS | E/F mean [95% CI] |", "| --- | --- | --- |"]
    for variant, ef in report.get("effect_ratio", {}).items():
        m = ef.get("mean")
        if m is None:
            continue
        lp = report["quality"][variant]["lpips"]
        rows.append(f"| {variant} | {lp:.3f} | {m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}] |")
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--repo", default="bivri/dazzle-generator")
    p.add_argument("--public", action="store_true", help="create the repo public (default private)")
    p.add_argument("--dry-run", action="store_true", help="render the card and print it; upload nothing")
    a = p.parse_args()
    if a.tag != f"v{dazzle.__version__}":
        p.error(f"tag {a.tag} does not match package version v{dazzle.__version__}")

    gen = DazzleGenerator.from_pretrained(a.model_dir)  # fails on a missing or mismatched config/weights
    report = json.loads(a.report.read_text())
    card = (
        (ROOT / "docs" / "hub-model-card.md")
        .read_text()
        .format(
            version=a.tag,
            code_version=report["manifest"]["code"],
            run_id=report["manifest"]["args"]["run_id"],
            params=sum(t.numel() for t in gen.parameters()),
            epsilon_levels=round(gen.default_epsilon * 255 / 2, 2),
            results=results_table(report),
        )
    )
    if a.dry_run:
        print(card)
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("config.json", "model.safetensors"):
            shutil.copy(a.model_dir / name, Path(tmp) / name)
        (Path(tmp) / "README.md").write_text(card)
        info = api.upload_folder(repo_id=a.repo, folder_path=tmp, commit_message=f"Release {a.tag}")
    api.create_tag(a.repo, tag=a.tag, revision=info.oid)
    print(f"{a.repo}@{a.tag} commit {info.oid}")


if __name__ == "__main__":
    main()
