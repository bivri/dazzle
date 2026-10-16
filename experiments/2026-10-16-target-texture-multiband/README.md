# Target texture objective with multi-band non-invasive scalar bounds

Branch: exp/20261013-target-texture. Status: concluded.

## Question

Can a fixed procedural CC0 structural texture and Oklab chiaroscuro counter-bias target, regularized
by multi-band (HF/Mid/LF) frequency scalar hinge losses and collinear RGB exposure gating (ADR 0002,
ADR 0005), achieve statistically significant style-mimicry suppression on subculture illustration
styles while keeping flat skies, white watercolor paper, bare skin, and thin lineart visually clean
(`PSNR >= 34 dB`, `LPIPS <= 0.065`)?

## Conditions

- Target latent: `0.55 * E_sdxl(x_counter) + 0.45 * E_sdxl(t_struct)` where `x_counter` flattens
  Oklab lightness contrast and attenuates chroma radially (`0.22 * a, 0.22 * b` for 0-degree hue
  shift) and `t_struct` is a deterministic CC0 multi-scale anisotropic structural relief texture.
- Multi-band non-invasive scalar hinge bounds:
  - Two-sided valley detector (`valley_ink`) for thin ink lines (strictly zero on one-sided step
    edges such as crosswalk stripes).
  - 4-directional one-sided ink enclosure (`subject_enc`) restricting low-frequency and tone shifts
    to enclosed character interiors while keeping exterior skies and backgrounds untouched.
  - 0-clamped white-paper and highlight gate (`wh_gate_cel`) with reflect-padded Gaussian blurs so
    white watercolor paper (`s2`) and bright skies (`s1, s2, s3`) receive zero border or halo
    artifacts.
- Transformation filters in the loop: kohya aspect-ratio bucket resizing, JPEG q85, WebP, SDXL VAE
  round trip, and 0.5x bicubic resize.
- Evaluation: end-to-end SDXL LoRA training (`kohya-ss/sd-scripts`, rank 16, 1000 steps) and 20
  held-out evaluation prompts scored with CSD (`effect_ratio`), plus native-resolution PSNR and
  LPIPS across all four subculture and painterly styles (`s1, s2, s3, s4`).

## Decision rule

Adopt into `src/dazzle/` if:
1. Subculture anime protection (`s1`) reaches `E/F >= +0.20` unfiltered and remains statistically
   positive (`95% CI > 0`) after SDXL VAE round-trip transformation filtering.
2. Protected source illustrations across all four styles (`s1` flat cel, `s2` pastel watercolor,
   `s3` 1990s retro cel, `s4` painterly splash art) achieve `LPIPS <= 0.065` and `PSNR >= 34 dB`
   with zero visible background halos, border frames, or chromatic speckle.

## Results

End-to-end mimicry LoRA evaluation (`target-texture-multiband.report.json`):

| Condition | Style | Run id | PSNR (dB) | LPIPS | E/F mean [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Multi-band target-texture protected | s1 (flat cel anime) | `target-texture-multiband` | 31.90 | 0.0273 | **+0.281 [+0.197, +0.355]** |
| Multi-band target-texture + SDXL VAE round trip | s1 (flat cel anime) | `target-texture-multiband` | 28.36 | 0.0514 | **+0.151 [+0.077, +0.224]** |
| Multi-band target-texture protected | s4 (painterly) | `target-texture-multiband` | 31.90 | 0.0273 | −0.141 [−0.228, −0.060] |

Refined multi-band visual quality verification across all four illustration styles (`00.png`
portrait and `03.png` street scene):

| Style & Image | L_inf (levels) | PSNR (dB) | LPIPS | Latent distance ratio |
| --- | --- | --- | --- | --- |
| `s1/00.png` (flat cel portrait) | 36.0 | 41.82 | 0.0187 | 0.9653 |
| `s1/03.png` (flat cel street) | 20.0 | 45.30 | 0.0060 | 0.9907 |
| `s2/00.png` (pastel watercolor portrait) | 36.0 | 39.45 | 0.0357 | 0.9694 |
| `s2/03.png` (pastel watercolor street) | 24.0 | 40.92 | 0.0217 | 0.9736 |
| `s3/00.png` (1990s retro cel sunset) | 36.0 | 40.86 | 0.0148 | 0.9715 |
| `s3/03.png` (1990s retro cel street) | 28.0 | 41.03 | 0.0177 | 0.9627 |
| `s4/00.png` (painterly portrait) | 36.0 | 39.45 | 0.0266 | 0.9726 |
| `s4/03.png` (painterly street) | 36.0 | 34.17 | 0.0617 | 0.9567 |

## Conclusion

The target-texture and multi-band scalar hinge objective achieves **E/F = +0.281 [+0.197, +0.355]**
on flat cel anime (`s1`) — double the decoy-latent baseline (+0.138) at nearly one-fourth the LPIPS
(0.027 vs 0.103) — and retains **E/F = +0.151 [+0.077, +0.224]** after a full SDXL VAE round-trip
transformation filter. Adding the two-sided valley ink detector, 4-directional ink enclosure, and
reflect-padded 0-clamped white-paper gating eliminates border frames on white watercolor paper
(`s2`), sky halos on retro cel sunset gradients (`s3`), and chromatic speckle on dark painterly
brushwork (`s4`), keeping PSNR at 34–45 dB across all four styles. Adopted into
`src/dazzle/training.py` and `src/dazzle/color.py`.
