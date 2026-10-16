"""Transformation filters a mimicry pipeline may apply to collected images before training."""

import io
from collections.abc import Callable

import torch
from PIL import Image

from dazzle.protect import to_image, to_tensor


def jpeg(quality: int = 85) -> Callable[[Image.Image], Image.Image]:
    def apply(im: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return apply


def resize_roundtrip(scale: float = 0.5) -> Callable[[Image.Image], Image.Image]:
    def apply(im: Image.Image) -> Image.Image:
        w, h = im.size
        small = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BICUBIC)
        return small.resize((w, h), Image.Resampling.BICUBIC)

    return apply


def vae_roundtrip(device: str = "cuda") -> Callable[[Image.Image], Image.Image]:
    """SDXL VAE encode (latent mean) and decode at native resolution (sides must be multiples of 8)."""
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32).to(device).eval()

    @torch.inference_mode()
    def apply(im: Image.Image) -> Image.Image:
        x = to_tensor(im).to(device)
        return to_image(vae.decode(vae.encode(x).latent_dist.mean).sample)

    return apply


def build(name: str, device: str = "cuda") -> Callable[[Image.Image], Image.Image]:
    """Filter by name: `jpeg85`, `resize0.5`, `vae`."""
    if name.startswith("jpeg"):
        return jpeg(int(name.removeprefix("jpeg")))
    if name.startswith("resize"):
        return resize_roundtrip(float(name.removeprefix("resize")))
    if name == "vae":
        return vae_roundtrip(device)
    raise ValueError(f"unknown filter {name!r}")
