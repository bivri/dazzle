"""Generator training against the SDXL-VAE target-texture objective (ADR 0005).

The generator is trained to move the SDXL VAE latent of the filtered candidate image toward a
composite target latent combining an Oklab chiaroscuro counter-bias image and a deterministic
procedural CC0 structural relief texture, regularized by image-independent multi-band frequency
and collinearity scalar hinge losses (ADR 0002, ADR 0005):

    4.0 * target-latent ratio  ||E(T(x+d)) - z_target||^2 / ||E(T(x)) - z_target||^2
  + 1.5 * multi-scale latent cosine repulsion (high/mid/low bands)
  + LPIPS hinge                15 * relu(LPIPS - limit) + 0.5 * LPIPS
  + 8.0 * multi-band non-invasive scalar hinge loss (HF/Mid/LF bounds + collinear exposure)
  + 0.8 * total variation of d

Run: python -m dazzle.training --out <dir>   (data from $DAZZLE_WORKSPACE, see dazzle.workspace)
"""

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dazzle.color import gaussian_blur, oklab_to_rgb, rgb_to_oklab
from dazzle.generator import DazzleGenerator, levels_to_epsilon
from dazzle.workspace import STYLES, Workspace


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    lr: float = 2e-4
    batch_size: int = 4
    img_size: int = 512
    lpips_limit: float = 0.065
    epsilon_levels: float = 20.0
    hf_levels: float = 26.0
    mid_levels: float = 23.0
    lf_levels: float = 20.0
    seed: int = 0


def generate_structural_texture(size: int = 512, seed: int = 42, device: str | torch.device = "cpu") -> torch.Tensor:
    """Deterministic CC0 multi-scale anisotropic cross-hatching and chiaroscuro relief texture."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(size)[:, None]
    kx = np.fft.fftfreq(size)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)
    k[0, 0] = 1.0

    b_fine = np.exp(-((np.log(k) - np.log(1.0 / 14.0)) ** 2) / (2 * 0.38**2)) * (
        1.0 + 0.65 * np.cos(2 * (theta - 0.55))
    )
    b_mid = np.exp(-((np.log(k) - np.log(1.0 / 30.0)) ** 2) / (2 * 0.42**2)) * (1.0 + 0.65 * np.cos(2 * (theta + 0.85)))
    b_coarse = np.exp(-((np.log(k) - np.log(1.0 / 68.0)) ** 2) / (2 * 0.48**2))

    n_lum = rng.standard_normal((size, size))
    f_lum = np.fft.fft2(n_lum) * (0.55 * b_fine + 1.0 * b_mid + 0.65 * b_coarse)
    f_lum[0, 0] = 0.0
    s_lum = np.real(np.fft.ifft2(f_lum))
    s_lum = np.tanh(2.6 * (s_lum / (np.std(s_lum) + 1e-8))).clip(-1.0, 1.0)

    ch = (0.78 * s_lum).clip(-1.0, 1.0).astype(np.float32)
    rgb = np.stack([ch, ch, ch], axis=0)
    return torch.from_numpy(rgb).unsqueeze(0).to(device)


def make_counter_bias_image(x: torch.Tensor) -> torch.Tensor:
    """Image-independent Oklab chiaroscuro contrast flattening and phase-inverted mid/high bands
    with uniform radial chroma attenuation (0-degree hue shift, ADR 0002).
    """
    with torch.no_grad():
        lab = rgb_to_oklab(x)
        l_ch, a_ch, b_ch = lab[:, :1], lab[:, 1:2], lab[:, 2:3]
        l_low = gaussian_blur(l_ch, 24.0)
        l_mid = gaussian_blur(l_ch, 3.5) - l_low
        l_high = l_ch - gaussian_blur(l_ch, 3.5)
        l_new = (0.52 + 0.22 * (l_low - 0.52) - 1.85 * l_mid - 1.95 * l_high).clamp(0.06, 0.94)
        return oklab_to_rgb(torch.cat([l_new, 0.22 * a_ch, 0.22 * b_ch], dim=1))


def compute_frequency_bounds(
    x: torch.Tensor,
    hf_levels: float = 26.0,
    mid_levels: float = 23.0,
    lf_levels: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute image-independent derivative bounds for HF, Mid, LF bands and collinear exposure.

    Uses a two-sided valley detector for thin lineart, 4-directional one-sided ink enclosure for
    smooth interior shading, and 0-clamped white-paper / flat-skin / dark-shadow gates across
    subculture (cel, pastel watercolor, retro anime) and painterly styles.
    """
    with torch.no_grad():
        _, _, h, w = x.shape
        dev = x.device
        ry = torch.minimum(torch.arange(h, device=dev), torch.arange(h - 1, -1, -1, device=dev)).float()
        rx = torch.minimum(torch.arange(w, device=dev), torch.arange(w - 1, -1, -1, device=dev)).float()
        edge_3px = ((ry.view(1, 1, h, 1) - 3.0) / 3.0).clamp(0.0, 1.0) * ((rx.view(1, 1, 1, w) - 3.0) / 3.0).clamp(
            0.0, 1.0
        )

        rgb01 = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        lum = rgb01.mean(dim=1, keepdim=True)
        chr_raw = rgb01.max(dim=1, keepdim=True).values - rgb01.min(dim=1, keepdim=True).values

        lum_sm = gaussian_blur(lum, 14.0)
        lum_mid = gaussian_blur(lum, 6.0)
        chr_sm = gaussian_blur(chr_raw, 14.0)
        chr_mid = gaussian_blur(chr_raw, 6.0)

        warm_scene = (rgb01[:, 0:1] - rgb01[:, 2:3]).mean(dim=(2, 3), keepdim=True)
        cel_scene_gate = ((0.14 - warm_scene) / 0.03).clamp(0.0, 1.0)
        painterly_gate = 1.0 - cel_scene_gate

        cel_art_gate = cel_scene_gate * ((0.20 - chr_sm) / 0.07).clamp(0.25, 1.0)
        lum_pad2 = F.pad(lum, (2, 2, 2, 2), mode="reflect")
        lum_pad4 = F.pad(lum, (4, 4, 4, 4), mode="reflect")
        val_x2 = torch.minimum(lum_pad2[:, :, 2:-2, :-4], lum_pad2[:, :, 2:-2, 4:]) - lum
        val_y2 = torch.minimum(lum_pad2[:, :, :-4, 2:-2], lum_pad2[:, :, 4:, 2:-2]) - lum
        val_x4 = torch.minimum(lum_pad4[:, :, 4:-4, :-8], lum_pad4[:, :, 4:-4, 8:]) - lum
        val_y4 = torch.minimum(lum_pad4[:, :, :-8, 4:-4], lum_pad4[:, :, 8:, 4:-4]) - lum
        valley_ink = torch.maximum(torch.maximum(val_x2, val_y2), torch.maximum(val_x4, val_y4))
        thin_ink = ((valley_ink - 0.045) / 0.06).clamp(0.0, 1.0) * cel_art_gate

        d2 = (x - gaussian_blur(x, 2.2)).abs().mean(dim=1, keepdim=True) * edge_3px
        d6 = (x - gaussian_blur(x, 6.0)).abs().mean(dim=1, keepdim=True) * edge_3px
        d12 = (x - gaussian_blur(x, 12.0)).abs().mean(dim=1, keepdim=True) * edge_3px

        step_edge_cel = ((d2 - 0.055) / 0.035).clamp(0.0, 1.0) * cel_scene_gate
        step_edge_paint = ((d2 - 0.120) / 0.045).clamp(0.0, 1.0) * painterly_gate
        ink_boundary_guard = 1.0 - (
            gaussian_blur(torch.maximum(thin_ink, step_edge_cel + step_edge_paint), 2.6) * 2.4
        ).clamp(0.0, 0.95)
        ink_pixel_guard = 1.0 - ((0.24 - lum) / 0.06).clamp(0.0, 1.0) * cel_scene_gate

        r_enc = min(64, max(h // 4, 4), max(w // 4, 4))
        t_grid = torch.arange(r_enc + 1, device=dev, dtype=x.dtype)
        w_1d = torch.exp(-0.5 * (t_grid / 26.0) ** 2)
        w_1d = w_1d / w_1d.sum()
        k_l = torch.cat([w_1d.flip(0), torch.zeros(r_enc, device=dev, dtype=x.dtype)]).view(1, 1, 1, -1)
        k_r = torch.cat([torch.zeros(r_enc, device=dev, dtype=x.dtype), w_1d]).view(1, 1, 1, -1)
        k_u = torch.cat([w_1d.flip(0), torch.zeros(r_enc, device=dev, dtype=x.dtype)]).view(1, 1, -1, 1)
        k_d = torch.cat([torch.zeros(r_enc, device=dev, dtype=x.dtype), w_1d]).view(1, 1, -1, 1)

        ink_l = F.conv2d(thin_ink, k_l, padding=(0, r_enc))
        ink_r = F.conv2d(thin_ink, k_r, padding=(0, r_enc))
        ink_u = F.conv2d(thin_ink, k_u, padding=(r_enc, 0))
        ink_d = F.conv2d(thin_ink, k_d, padding=(r_enc, 0))

        enc_h = ((torch.minimum(ink_l, ink_r) * 38.0 - 0.10) / 0.14).clamp(0.0, 1.0)
        enc_u = ((ink_u * 34.0 - 0.08) / 0.14).clamp(0.0, 1.0)
        enc_d = ((ink_d * 34.0 - 0.08) / 0.14).clamp(0.0, 1.0)
        subject_enc = gaussian_blur(enc_h * enc_u * enc_d, 4.0) * ink_boundary_guard

        tex_core = F.relu(d2 - 0.030) + 0.85 * F.relu(d6 - 0.040)
        tex_sp = F.relu(gaussian_blur(tex_core * ink_boundary_guard, 4.5) - 0.0035)
        tex_enc = (tex_sp / (tex_sp + 0.013)).clamp(0.0, 1.0)

        sh_gate_cel = ((lum - 0.14) / 0.08).clamp(0.0, 1.0) * ((lum_sm - 0.14) / 0.08).clamp(0.25, 1.0)
        wh_gate_cel = ((0.91 - lum + 0.70 * chr_mid) / 0.07).clamp(0.0, 1.0) * (
            (0.92 - lum_sm + 0.70 * chr_sm) / 0.06
        ).clamp(0.0, 1.0)

        sh_gate_paint = ((lum - 0.21) / 0.08).clamp(0.0, 1.0) * ((lum_sm - 0.21) / 0.07).clamp(0.0, 1.0)
        wh_gate_paint = ((0.82 - lum_mid) / 0.10).clamp(0.15, 1.0)
        active_paint = ((gaussian_blur(d12, 10.0) - 0.024) / 0.020).clamp(0.0, 1.0)

        hf_core = F.relu(d2 - 0.032)
        hf_sp = F.relu(gaussian_blur(hf_core, 2.2) - 0.003)
        hf_norm = hf_sp / (hf_sp + 0.014)
        hf_cel = hf_norm * sh_gate_cel * wh_gate_cel * ink_boundary_guard * ink_pixel_guard
        hf_paint = (22.0 / 26.0) * hf_norm * sh_gate_paint * wh_gate_paint * ink_boundary_guard
        hf_bound = (cel_scene_gate * hf_cel + painterly_gate * hf_paint) * edge_3px

        mid_core = F.relu(d6 - 0.032 * cel_scene_gate - 0.022 * painterly_gate)
        mid_sp = F.relu(gaussian_blur(mid_core, 3.8) - 0.002)
        mid_norm = (mid_sp / (mid_sp + 0.012)) * ink_boundary_guard
        mid_cel = mid_norm * tex_enc * sh_gate_cel * wh_gate_cel * ink_boundary_guard * ink_pixel_guard
        mid_paint = (
            (18.0 / 23.0)
            * torch.maximum(mid_norm, 0.55 * active_paint * ink_boundary_guard)
            * sh_gate_paint
            * wh_gate_paint
        )
        mid_bound = (cel_scene_gate * mid_cel + painterly_gate * mid_paint) * edge_3px

        lf_cel = (
            0.55 * torch.maximum(subject_enc, 0.70 * tex_enc * mid_norm) * sh_gate_cel * wh_gate_cel * ink_pixel_guard
        )
        lf_paint = 0.20 * active_paint * sh_gate_paint * wh_gate_paint * ink_boundary_guard
        lf_bound = (cel_scene_gate * lf_cel + painterly_gate * lf_paint) * edge_3px

        w_rgb = (rgb01 / (lum + 0.03)).clamp(0.0, 1.75)
        return (
            (hf_levels / 127.5) * hf_bound,
            (mid_levels / 127.5) * mid_bound,
            (lf_levels / 127.5) * lf_bound,
            w_rgb,
        )


def multi_band_hinge_loss(
    x: torch.Tensor,
    delta: torch.Tensor,
    hf_levels: float = 26.0,
    mid_levels: float = 23.0,
    lf_levels: float = 20.0,
) -> torch.Tensor:
    """Scalar hinge loss enforcing multi-band frequency bounds and collinear RGB exposure."""
    hf_allow, mid_allow, lf_allow, w_rgb = compute_frequency_bounds(x, hf_levels, mid_levels, lf_levels)
    d_lf = gaussian_blur(delta, 14.0)
    d_mid = gaussian_blur(delta, 3.5) - d_lf
    d_hf = delta - gaussian_blur(delta, 3.5)
    d_lum = delta.mean(dim=1, keepdim=True)
    off_axis = (delta - d_lum * w_rgb).abs().mean()
    return (
        F.relu(d_hf.abs() - hf_allow).mean()
        + F.relu(d_mid.abs() - mid_allow).mean()
        + F.relu(d_lf.abs() - lf_allow).mean()
        + 0.5 * off_axis
    )


class StyleImages(Dataset):
    """Training illustrations resized once to a square training size and cached in memory."""

    def __init__(self, ws: Workspace, styles: tuple[str, ...], size: int):
        self.size = size
        self.files = [f for s in styles for f in ws.train_images(s)]
        self.images = [
            torch.from_numpy(
                np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.Resampling.LANCZOS), np.float32)
                / 127.5
                - 1.0
            ).permute(2, 0, 1)
            for f in self.files
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.images[i]


def jpeg_like_blur(x: torch.Tensor, quality: float = 85.0) -> torch.Tensor:
    """Differentiable stand-in for JPEG detail loss: blend with a Gaussian blur."""
    sigma = (100.0 - quality) / 50.0
    return 0.7 * x + 0.3 * gaussian_blur(x, sigma)


def total_variation(d: torch.Tensor) -> torch.Tensor:
    return (d[:, :, 1:] - d[:, :, :-1]).abs().mean() + (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean()


def train(cfg: TrainConfig, ws: Workspace, out: Path, device: str) -> DazzleGenerator:
    import lpips
    from diffusers import AutoencoderKL

    torch.manual_seed(cfg.seed)
    use_cuda = device.startswith("cuda")
    if use_cuda:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    eps = levels_to_epsilon(cfg.epsilon_levels)
    gen = DazzleGenerator(base_ch=32, default_epsilon=eps).to(device)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32).to(device).eval()
    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    vae.requires_grad_(False)
    lp.requires_grad_(False)

    def encode(x: torch.Tensor) -> torch.Tensor:
        return vae.encode(x).latent_dist.mean

    tex_struct = generate_structural_texture(size=cfg.img_size, seed=42, device=device)
    with torch.no_grad():
        z_struct_single = encode(tex_struct)

    data = StyleImages(ws, STYLES, cfg.img_size)
    loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, drop_last=True, pin_memory=use_cuda)
    opt = torch.optim.AdamW(gen.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    print(f"config={asdict(cfg)} images={len(data)}", flush=True)
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        gen.train()
        sums = np.zeros(3)
        for x in loader:
            x = x.to(device, non_blocking=use_cuda)
            delta = gen(x, eps)
            xp = (x + delta).clamp(-1, 1)
            if torch.rand(()) > 0.5:
                x_t, xp_t = x.flip(-1), xp.flip(-1)
            else:
                x_t, xp_t = x, xp
            if torch.rand(()) > 0.5:
                xp_t = jpeg_like_blur(xp_t)
            with torch.no_grad():
                zx = encode(x_t)
                z_counter = encode(make_counter_bias_image(x_t))
                z_struct = z_struct_single.expand_as(z_counter)
                z_target = 0.55 * z_counter + 0.45 * z_struct
                denom = (zx - z_target).pow(2).mean().clamp(min=1e-5)
                zx_hf = zx - gaussian_blur(zx, 1.5)
                zx_mid = gaussian_blur(zx, 1.5) - gaussian_blur(zx, 5.0)
            zxp = encode(xp_t)
            zxp_hf = zxp - gaussian_blur(zxp, 1.5)
            zxp_mid = gaussian_blur(zxp, 1.5) - gaussian_blur(zxp, 5.0)
            latent_ratio = (zxp - z_target).pow(2).mean() / denom
            repulsion = 0.80 * _cos_sim(zxp_hf, zx_hf) + 0.70 * _cos_sim(zxp_mid, zx_mid)
            perc = lp(x, xp).mean()
            hinge_mb = multi_band_hinge_loss(x, delta, cfg.hf_levels, cfg.mid_levels, cfg.lf_levels)
            loss = (
                4.0 * latent_ratio
                + 1.5 * repulsion
                + 15.0 * F.relu(perc - cfg.lpips_limit)
                + 0.5 * perc
                + 8.0 * hinge_mb
                + 0.8 * total_variation(delta)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sums += [loss.item(), latent_ratio.item(), perc.item()]
        sched.step()
        n = len(loader)
        print(
            f"epoch {epoch}/{cfg.epochs} loss={sums[0] / n:.4f} latent_ratio={sums[1] / n:.3f} "
            f"lpips={sums[2] / n:.3f} elapsed={time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    gen.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    gen.save_pretrained(out)
    return gen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in asdict(TrainConfig()).items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--out", type=Path, required=True, help="output directory (save_pretrained layout)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = vars(p.parse_args())
    out, device = a.pop("out"), a.pop("device")
    train(TrainConfig(**a), Workspace.from_env(), out, device)


if __name__ == "__main__":
    main()
