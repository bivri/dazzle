"""CSD style embeddings (tomg-group-umd/CSD-ViT-L, CC-BY-4.0; evaluation only, never a training target)."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = "tomg-group-umd/CSD-ViT-L"
REVISION = "5bc26a6fb0487f3f00a2a7313135103a005b1b67"


def _hf_state_dict(ck: dict) -> dict:
    src = {k.removeprefix("module.backbone."): v for k, v in ck.items() if k.startswith("module.backbone.")}
    m = {
        "vision_model.embeddings.class_embedding": src["class_embedding"],
        "vision_model.embeddings.position_embedding.weight": src["positional_embedding"],
        "vision_model.embeddings.patch_embedding.weight": src["conv1.weight"],
        "vision_model.pre_layrnorm.weight": src["ln_pre.weight"],
        "vision_model.pre_layrnorm.bias": src["ln_pre.bias"],
        "vision_model.post_layernorm.weight": src["ln_post.weight"],
        "vision_model.post_layernorm.bias": src["ln_post.bias"],
    }
    for i in sorted({int(k.split(".")[2]) for k in src if k.startswith("transformer.resblocks.")}):
        b, t = f"transformer.resblocks.{i}.", f"vision_model.encoder.layers.{i}."
        for s in ("weight", "bias"):
            q, k_, v = src[f"{b}attn.in_proj_{s}"].chunk(3, dim=0)
            m[f"{t}self_attn.q_proj.{s}"], m[f"{t}self_attn.k_proj.{s}"], m[f"{t}self_attn.v_proj.{s}"] = q, k_, v
            m[f"{t}self_attn.out_proj.{s}"] = src[f"{b}attn.out_proj.{s}"]
            m[f"{t}layer_norm1.{s}"] = src[f"{b}ln_1.{s}"]
            m[f"{t}layer_norm2.{s}"] = src[f"{b}ln_2.{s}"]
            m[f"{t}mlp.fc1.{s}"] = src[f"{b}mlp.c_fc.{s}"]
            m[f"{t}mlp.fc2.{s}"] = src[f"{b}mlp.c_proj.{s}"]
    return m


class CSD:
    def __init__(self, device: str = "cuda"):
        from huggingface_hub import hf_hub_download
        from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

        ck = torch.load(
            hf_hub_download(REPO, "pytorch_model.bin", revision=REVISION), map_location="cpu", weights_only=False
        )["model_state_dict"]
        cfg = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            patch_size=14,
            image_size=224,
            hidden_act="quick_gelu",
        )
        model = CLIPVisionModel(cfg)
        sd = _hf_state_dict(ck)
        if not any(k.startswith("vision_model.") for k in model.state_dict()):  # transformers 5 layout
            sd = {k.removeprefix("vision_model."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        self.model = model.to(device).eval()
        self.proj = ck["module.last_layer_style"].to(device)
        self.proc = CLIPImageProcessor()
        self.device = device

    @torch.inference_mode()
    def embed(self, paths: list[Path], batch: int = 16) -> np.ndarray:
        """L2-normalized style embeddings, float64 (N, 768)."""
        if not paths:
            raise ValueError("no images to embed")
        out = []
        for i in range(0, len(paths), batch):
            ims = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
            px = self.proc(images=ims, return_tensors="pt")["pixel_values"].to(self.device)
            feats = self.model(pixel_values=px).pooler_output @ self.proj
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu())
        return torch.cat(out).numpy().astype(np.float64)
