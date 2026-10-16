# How Dazzle is evaluated

Dazzle's claim is narrow: a LoRA trained on protected images learns less of the artist's style
than one trained on the originals. This document describes how that is measured.

## Setup

- **Styles.** Four synthetic illustration styles (flat cel shading, pastel watercolor, 1990s cel
  animation, painterly splash art), each with 30 training images and 10 held-out reference
  images, generated with Apache-2.0 models.
- **Mimicry recipe.** One LoRA per style and condition on Illustrious-XL v2.0 with
  kohya-ss/sd-scripts: rank 16, alpha 16, UNet only, 1,500 steps, batch 2, 1024 buckets, cached
  latents, AdamW8bit at 1e-4, seed 42 (`dazzle.evaluation.lora`).
- **Renders.** 10 prompts not in the training captions × 4 seeds, 1024², Euler ancestral, 28 steps,
  CFG 6, for the base model, the unprotected LoRA and each condition (`dazzle.evaluation.render`).
- **Transformation filters.** JPEG q85, 0.5× bicubic down-and-up, SDXL VAE round trip, applied to
  the protected images before LoRA training (`dazzle.evaluation.filters`).

## Measures

- **E/F** (`dazzle.evaluation.score`): style-specific gain F = how much closer the renders moved to
  the target style's held-out references than to the other styles' references, in CSD style space,
  relative to the base model. E/F = (F_unprotected − F_condition) / F_unprotected. 95% intervals
  bootstrap over references and over renders within prompts. Each style is one LoRA run, so the
  intervals do not include run-to-run training variance; random-noise controls varied by up to
  ±0.19 on single styles.
- **Quality**: LPIPS (AlexNet) and PSNR between original and protected image at native
  resolution, averaged over all training images.
- **Proxy** (`dazzle.evaluation.proxy`): the drop in style specificity of the protected training
  images themselves. Used only to rank candidates before LoRA training
  ([calibration](decisions-log/evaluation.md)).
- **Output disruption** ([ADR 0005](adr/0005-target-texture-objective.md)): primary validation of
  protection via blind 2AFC human judgment of artwork unusability and visible contamination by the
  target pattern, accompanied by automated spectral texture similarity against the target pattern.

## Running

```bash
export DAZZLE_WORKSPACE=~/dazzle-v2
python -m dazzle.evaluation.run --run-id <id> --generator bivri/dazzle-generator --filters jpeg85 vae resize0.5
python -m dazzle.evaluation.run --run-id <id> --import-set candidate=/path/to/sets --proxy-only
```

A run of four styles with three filters trains 16 LoRAs; on an RTX 5090 one LoRA takes about
15 minutes and rendering 160 images about 10 minutes.

## Limits

Synthetic styles are easier to learn than most real artists' styles, so effects here may not
transfer. CSD agreement with human judgments on subtle pairs was 9 of 15 in an internal study;
human 2AFC judgments are required before any public effect claim.
