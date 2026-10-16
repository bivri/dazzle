# 0005. Target texture contamination objective and output disruption criteria

Status: accepted, 2026-10-13 (replaces decision 1 and success criteria of ADR 0004)

## Context

Experiment `2026-10-13-denoiser-transfer` evaluated raising the denoising loss of Apache-2.0
surrogates (FLUX.2-klein-base-4B and Z-Image) plus an SDXL VAE term through transformation
filters in the forward loop. Even at the visible top step (L∞ 32, native LPIPS 0.195), surrogate
losses rose 1.3–2.1× while LoRA style gain reduction over random control was only +0.119
(pre-registered rule required ≥ 0.15), and generated renders retained the artist's style without
observable pattern transfer. Under ADR 0004's own terms, that failure reopened decision 1.

Effective style-mimicry disruption requires shifting fine-tuning representations toward a shared,
structured target pattern so that any LoRA trained on the protected set reproduces degraded
surface textures rather than the artist's clean brushwork and shading. Dazzle previously measured
only style-reproduction retention in CSD style space (E/F) without directly targeting or measuring
structured texture transfer.

Dazzle operates within clear license and architectural boundaries:

- Runtime code, models and assets remain Apache-2.0, MIT or CC0 (ADR 0001). The SDXL VAE
  `madebyollin/sdxl-vae-fp16-fix` is MIT-licensed and directly white-box to the mimicry trainer's
  latent caching step in kohya-ss/sd-scripts, paired with Apache-2.0 diffusion surrogates.
- The generator keeps a single output head, scalar losses, no image-derived map or mask, and no
  stored pattern bank (ADR 0002).
- A single, deterministic procedural CC0 pattern is synthesized mathematically and applied
  uniformly across all images.

## Decision

1. **Objective (replaces ADR 0004 Decision 1).** The protective pattern is optimized to move the
   SDXL VAE latent and multi-scale diffusion features of the filtered candidate image toward a
   single fixed procedural CC0 target texture pattern:
   `L = ‖E_sdxl(F(B(x + delta))) - E_sdxl(B(t))‖² / ‖E_sdxl(B(t))‖²`
   with `t` the fixed procedural target pattern, `B` bucket resizing, and `F` a transformation
   filter, regularized by image-independent multi-band frequency and colinearity scalar hinge
   losses.
2. **Target texture.** A single, image-independent procedural CC0 pattern (multi-scale
   reaction-diffusion and anisotropic structural relief) synthesized deterministically from code.
   It is scaled at a fixed pixel scale for bucket invariance and applied uniformly without image
   feature selection (ADR 0002).
3. **Success criterion.** Protection is evaluated by both style-specific gain removal (`E/F` in CSD
   style space) and visible degradation of unauthorized LoRA generations compared to clean LoRA
   generations, while keeping the protected source illustrations visually clean (`PSNR ≥ 30 dB`,
   `LPIPS ≤ 0.065`).
4. **Inference, transformation robustness, and visibility.** Decisions 2, 4, and 5 of ADR 0004
   remain unchanged:
   - Hybrid inference: one generator forward pass plus optional per-image refinement steps.
   - Robustness: In-loop filters (JPEG, resizing, WebP, SDXL VAE round trip, noisy upscaling).
   - Visibility: `default` step (`LPIPS ≤ 0.065`) and `strong` step (`LPIPS ≤ 0.20`).

## Consequences

- Optimization directly targets the SDXL VAE latent space and diffusion feature statistics used by
  mimicry trainers while preserving flat backgrounds, white paper, and crisp lineart via scalar
  hinge constraints.
- The single procedural CC0 pattern is self-contained in code and fully complies with ADR 0001 and
  ADR 0002.
