# 0004. Denoiser objective, generator with short refinement, filters in the loop

Status: accepted, 2026-10-13

## Context

Dazzle exists to disrupt unauthorized style mimicry. Earlier baselines show two key observations:

- Encoder-latent decoy objectives moved the SDXL VAE latent toward a different style's rendering of
  the same content, but did not prevent the LoRA from learning the artist's style: the decoy-latent
  baseline reached E/F −0.013 (`experiments/2026-10-08-decoy-latent-baseline`), and a stronger
  decoy-latent pattern at LPIPS 0.25 left both styles unchanged (`exp/20261009-strong-decoys`).
- A single-pass generator trained without per-image refinement left part of the attainable latent
  shift unused (squared distance ratio 0.77 against 0.52 for per-image optimization,
  `experiments/2026-10-08-decoy-latent-baseline`).
- Transformation filters such as SDXL VAE round trips, JPEG compression, and bucket resizing
  attenuate high-frequency protective patterns unless included directly in the optimization loop.

## Decision

1. **Objective.** The protective pattern is trained to raise the denoising loss of
   Apache-2.0 diffusion models, each through its own VAE, plus a latent term on the MIT SDXL VAE
   that the mimicry recipe encodes with. Candidates within available memory (≤ 32 GB):
   `black-forest-labs/FLUX.2-klein-4B` and `Tongyi-MAI/Z-Image-Turbo`. OpenRAIL denoisers (SD1.5,
   SDXL UNets) stay evaluation-only; ADR 0001 is unchanged.
2. **Inference.** A single-head generator produces the starting pattern in one pass; a few
   per-image optimization steps (5–10) on the same objective refine it on the serving GPU. Both
   stages obey ADR 0002: one output, scalar losses, no image-derived map, no stored pattern set,
   no selection by image features.
3. **Target.** The first target is LoRA fine-tuning of SDXL-family models, measured with the
   recipe in `docs/evaluation.md`. Reference editing and regeneration come later.
4. **Transformation robustness.** Training and refinement pass the image through the filters a
   published image meets (JPEG, resizing including kohya bucket resizing, WebP/AVIF re-encoding)
   and the cheap deliberate ones (SDXL VAE round trip, noisy upscaling with a permissively licensed
   upscaler) in the forward path, not through stand-ins. A remover model trained on protected
   images is measured and reported, never claimed.
5. **Visibility.** Strength keeps three steps. The default step stays within the invisible budget
   (saved PNG change ≤ 20 levels per channel, native LPIPS ≤ 0.08); the top step may be visible,
   chosen explicitly by the creator.

## Consequences

- A candidate replaces the production service only after blind human judgment shows a LoRA trained
  on its output reproduces the style less than one trained on the originals, with the filters
  above applied, at the default step.
- The decoy-latent objective in `dazzle.training` and its blur stand-in for JPEG are replaced when
  the first candidate of this design is adopted.
- Serving stays on a GPU; CPU-only protection is not a goal of this design.
- `Qwen/Qwen-Image` is Apache-2.0 but too large for the training loop on the available GPUs.
- Whether transfer from these denoisers to SDXL UNets is strong enough is unknown; an experiment
  measures it first, and its failure reopens decision 1, not ADR 0001 by default.
