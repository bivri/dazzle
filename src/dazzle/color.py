"""Oklab conversion and separable Gaussian blur for tensors in [-1, 1] sRGB."""

import math

import torch
import torch.nn.functional as F

_M1 = torch.tensor(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = torch.tensor(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M1_INV = torch.linalg.inv(_M1)
_M2_INV = torch.linalg.inv(_M2)


def _mix(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", m.to(x), x)


def rgb_to_oklab(x: torch.Tensor) -> torch.Tensor:
    rgb = ((x + 1) * 0.5).clamp(0, 1)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp(min=1e-8) ** 2.4)
    return _mix(_M2, _mix(_M1, lin).clamp(min=1e-8) ** (1 / 3))


def oklab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    lin = (_mix(_M1_INV, _mix(_M2_INV, lab) ** 3)).clamp(1e-8, 1)
    rgb = torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return (rgb * 2 - 1).clamp(-1, 1)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.1:
        return x
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-0.5 * ((torch.arange(2 * r + 1, device=x.device, dtype=x.dtype) - r) / sigma) ** 2)
    k = k / k.sum()
    c, h, w = x.shape[1], x.shape[2], x.shape[3]
    mode = "reflect" if (r < h and r < w) else "replicate"
    x = F.pad(x, (r, r, r, r), mode=mode)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), padding=0, groups=c)
    return F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), padding=0, groups=c)
