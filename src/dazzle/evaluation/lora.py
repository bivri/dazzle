"""Mimicry LoRA training with kohya-ss/sd-scripts (the style-mimicry recipe of the evaluation).

Recipe: SDXL base, rank 16 / alpha 16, UNet only, 1500 steps, batch 2, 1024 buckets, cached latents
and text-encoder outputs, AdamW8bit lr 1e-4 constant, bf16. Same recipe for every condition.
"""

import subprocess
import sys
from pathlib import Path

from dazzle.workspace import Workspace

BASE_MODEL = "Illustrious-XL-v2.0.safetensors"


def train_lora(ws: Workspace, set_dir: Path, style: str, out_dir: Path, seed: int = 42, steps: int = 1500) -> Path:
    """Train one LoRA on `set_dir/<style>`; returns the weights path. Skips if it already exists."""
    weights = out_dir / f"{style}.safetensors"
    if weights.exists():
        return weights
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / f"{style}.toml"
    cfg.write_text(
        '[general]\nenable_bucket = true\ncaption_extension = ".txt"\n'
        "[[datasets]]\nresolution = 1024\nbatch_size = 2\n"
        f'[[datasets.subsets]]\nimage_dir = "{set_dir / style}"\nnum_repeats = 1\n'
    )
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1", "--num_machines", "1", "--mixed_precision", "bf16", "--dynamo_backend", "no",
        str(ws.sd_scripts / "sdxl_train_network.py"),
        "--pretrained_model_name_or_path", str(ws.models / BASE_MODEL),
        "--dataset_config", str(cfg), "--output_dir", str(out_dir), "--output_name", style,
        "--network_module", "networks.lora", "--network_dim", "16", "--network_alpha", "16",
        "--network_train_unet_only", "--cache_text_encoder_outputs", "--cache_latents",
        "--learning_rate", "1e-4", "--optimizer_type", "AdamW8bit", "--lr_scheduler", "constant",
        "--max_train_steps", str(steps), "--mixed_precision", "bf16", "--save_precision", "bf16",
        "--gradient_checkpointing", "--sdpa", "--no_half_vae", "--seed", str(seed),
        "--max_data_loader_n_workers", "2",
    ]  # fmt: skip
    log = out_dir / f"{style}.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ws.sd_scripts, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not weights.exists():
        raise RuntimeError(f"LoRA training failed for {set_dir / style} (exit {rc}); see {log}")
    return weights
