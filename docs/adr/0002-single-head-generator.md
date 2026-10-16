# 0002. Single-head generator, image-independent shaping

Status: accepted, 2026-10-08

## Context

To keep the inference path lightweight, deterministic, and exportable to a single-graph ONNX model
at arbitrary resolutions, Dazzle avoids multi-branch routing, auxiliary segmentation or mask heads,
and external pattern banks.

## Decision

- The generator's only output is `delta = epsilon * tanh(G(x))`. No map, mask or routing head.
- No code computes an explicit spatial map or mask tensor from the image, uses LPIPS with
  `spatial=True`, uses a per-region epsilon tensor, or keeps a bank of stored patterns.
- Perceptual shaping uses only image-independent operations (fixed frequency or color-space
  filters) and scalar losses. Spatial adaptation is learned end-to-end within the single U-Net
  backbone or via scalar hinge penalties.

## Consequences

`scripts/check_policy.py` enforces these structural rules in pre-commit and CI, keeping the
generator and training pipeline self-contained and uniform.
