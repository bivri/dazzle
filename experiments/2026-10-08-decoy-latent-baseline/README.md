# Decoy-latent baseline, re-measured

Branch: none (measurement predates the branch workflow). Status: concluded.

## Question

What does the decoy-latent baseline generator (1.95M-parameter U-Net, decoy-latent teacher objective,
LPIPS hinge 0.09, epsilon 20/255) actually achieve, measured with held-out style references and
native-resolution quality?

## Conditions

- Protected sets were produced before this repository existed: each training image was resized to
  1024×1024, passed through the generator, and resized back to its original size. The library's
  `protect_image` instead works at native resolution, so a re-run with the library is a new
  condition, not a reproduction.
- Filters: JPEG q85, 0.5× bicubic round trip, SDXL VAE round trip, each applied after protection.
- Mimicry recipe and renders as in `docs/evaluation.md`, seed 42, one LoRA per style and condition.
- The generator was trained for 50 epochs on the same 120 training images it was evaluated on.

## Decision rule

Not pre-registered; this is a measurement of an existing baseline model.

## Results

LPIPS and proxy from `proxy-decoy-latent.json` (native resolution, 120 images); E/F from
`effect-ratio.heldout-refs.json`.

| Training set given to the mimicry LoRA | LPIPS | Proxy | E/F mean [95% CI] | s1 | s2 | s3 | s4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| protected | 0.103 | 0.045 | −0.013 [−0.055, +0.028] | +0.138 | −0.245 | +0.026 | +0.027 |
| protected + JPEG q85 | 0.132 | 0.053 | +0.030 [−0.009, +0.065] | +0.113 | +0.033 | −0.053 | +0.028 |
| protected + VAE round trip | 0.128 | 0.057 | +0.028 [−0.010, +0.061] | +0.155 | −0.067 | −0.021 | +0.045 |
| protected + 0.5× resize | – | – | +0.003 [−0.035, +0.036] | +0.043 | −0.139 | +0.041 | +0.068 |

The originally reported figures (`effect-ratio.train-refs.json`) used training images as style
references because the held-out references were missing on the scoring machine; their means differ
by up to 0.014 from the table above and lead to the same conclusion.

For comparison, earlier per-image optimization of the same decoy objective reached E/F +0.359
[+0.307, +0.415] at LPIPS 0.218. `proxy-calibration.json` holds the proxy for that and 13
other earlier conditions.

## Conclusion

The decoy-latent baseline has no measurable average effect; every 4-style interval includes 0. Only
style s1 shows a consistent gain of about +0.14. Its LPIPS at native resolution, 0.103, is above the
0.08 target. A per-image optimum of the same objective at LPIPS 0.075 moves the SDXL latent toward
the decoy further (squared latent distance ratio about 0.52 on 28 style-s1 images, against 0.77 for
the generator at the end of training), so the single-pass decoy-latent generator leaves most of the
attainable latent shift unused. Subsequent experiments screen target-texture objectives and
multi-band frequency bounds at LPIPS ≤ 0.08.
