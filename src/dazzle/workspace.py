"""Locations of data, base models and run outputs, outside the git checkout.

`DAZZLE_WORKSPACE` (default `~/dazzle-v2`) holds everything that must not enter git:

    data/styles/<style>/train/*.png (+ .txt captions)   images a mimicry LoRA is trained on
    data/styles/<style>/ref/*.png                        held-out style references for scoring
    data/decoys/<style>/*.png                            decoy-style versions of the train images
    data/reference-renders/<style>/{base,lora}/*.png     base-model and unprotected-LoRA renders
    models/                                              base checkpoints and generator weights
    sd-scripts/                                          kohya-ss/sd-scripts checkout (pinned)
    runs/<run-id>/                                       everything one evaluation run writes
"""

import os
from dataclasses import dataclass
from pathlib import Path

STYLES = ("s1", "s2", "s3", "s4")


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def from_env(cls) -> "Workspace":
        return cls(Path(os.environ.get("DAZZLE_WORKSPACE", "~/dazzle-v2")).expanduser())

    @property
    def styles(self) -> Path:
        return self.root / "data" / "styles"

    @property
    def decoys(self) -> Path:
        return self.root / "data" / "decoys"

    @property
    def reference_renders(self) -> Path:
        return self.root / "data" / "reference-renders"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def sd_scripts(self) -> Path:
        return self.root / "sd-scripts"

    def run(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def train_images(self, style: str) -> list[Path]:
        return sorted((self.styles / style / "train").glob("*.png"))

    def ref_images(self, style: str) -> list[Path]:
        return sorted((self.styles / style / "ref").glob("*.png"))
