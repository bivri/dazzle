"""Image-level protection: PIL in, PIL out, at the image's native resolution."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dazzle.generator import DazzleGenerator

_MULTIPLE = 8  # three stride-2 stages


def to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_image(x: torch.Tensor) -> Image.Image:
    arr = ((x[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).round().astype(np.uint8)
    return Image.fromarray(arr)


@torch.inference_mode()
def protect_image(
    image: Image.Image,
    generator: DazzleGenerator,
    epsilon: float | None = None,
) -> Image.Image:
    """Add the generator's protective pattern to `image`.

    The image is reflect-padded to a multiple of 8 and cropped back, so the output has the input's
    size. `epsilon` is in [-1, 1] units (see `levels_to_epsilon`); None uses the model default.
    """
    device = next(generator.parameters()).device
    x = to_tensor(image).to(device)
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % _MULTIPLE, (-w) % _MULTIPLE
    xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x
    delta = generator(xp, epsilon)[..., :h, :w]
    return to_image((x + delta).clamp(-1.0, 1.0))
