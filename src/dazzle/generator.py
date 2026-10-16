"""DazzleGenerator: a fully convolutional U-Net that emits a bounded protective pattern.

The single output head is `delta = epsilon * tanh(G(x))`. There is no separate map, mask or
routing output; `docs/adr/0002-single-head-generator.md` records why.
"""

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn


def levels_to_epsilon(levels: float) -> float:
    """8-bit intensity levels -> epsilon in the generator's [-1, 1] image units."""
    return levels / 255.0 * 2.0


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.conv(x), 0.2, inplace=True)


def _up(in_ch: int, out_ch: int) -> nn.Sequential:
    # Bilinear upsampling + conv instead of transposed conv avoids checkerboard patterns.
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
    )


class DazzleGenerator(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="dazzle",
    repo_url="https://github.com/bivri/dazzle",
    license="apache-2.0",
    tags=["image-protection", "style-mimicry", "protective-pattern"],
):
    """Input `x` is (B, 3, H, W) in [-1, 1] with H and W divisible by 8.
    Output is `delta` with |delta| <= epsilon, in the same units.

    `base_ch` and `default_epsilon` are serialized to the Hub `config.json`.
    """

    def __init__(self, base_ch: int = 32, default_epsilon: float = levels_to_epsilon(20)):
        super().__init__()
        self.default_epsilon = default_epsilon
        c = base_ch
        self.enc1 = ConvBlock(3, c)
        self.down1 = nn.Conv2d(c, c, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(c, c * 2)
        self.down2 = nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.down3 = nn.Conv2d(c * 4, c * 4, 4, stride=2, padding=1)
        self.bot1 = ConvBlock(c * 4, c * 8)
        self.bot_res = ResBlock(c * 8)
        self.up3 = _up(c * 8, c * 4)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = _up(c * 4, c * 2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = _up(c * 2, c)
        self.dec1 = ConvBlock(c * 2, c)
        self.out_conv = nn.Conv2d(c, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        eps = self.default_epsilon if epsilon is None else epsilon
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bot_res(self.bot1(self.down3(e3)))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return eps * torch.tanh(self.out_conv(d1))

    def protect(self, x: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        return (x + self.forward(x, epsilon)).clamp(-1.0, 1.0)
