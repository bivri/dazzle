# Evaluation decisions log

## 2026-10-08 — Decoy-latent baseline figures corrected

Re-measurement of the decoy-latent baseline protected sets showed four problems in the earlier
report, recorded in `experiments/2026-10-08-decoy-latent-baseline/`:

- Its LPIPS of 0.079 came from 3 images per style resized to a square 1024×1024. At native
  resolution on all 120 images the mean is 0.103.
- The scoring machine had no held-out style references and silently used the training images as
  references. With held-out references the 4-style mean E/F is −0.013 [−0.055, +0.028].
- "Retention" divided by an E/F near zero and reported 234,158 %. Retention is now undefined
  unless the unprotected-set CI lower bound is above 0.
- The generator was trained on the same 120 images it was evaluated on.

Decided: E/F is computed only against held-out references (`dazzle.evaluation.run` reads
`data/styles/<style>/ref`); LPIPS is reported at native resolution on the full set. Overturned
if a different reference protocol is adopted for all conditions at once.

## 2026-10-08 — Proxy calibrated against LoRA E/F

The proxy (style-specificity drop of the protected training images) was computed for 14 earlier
conditions with known LoRA E/F (62 style points, LPIPS 0.02–0.47): Spearman 0.81 per style,
Pearson 0.91 per condition, E/F ≈ 0.8 × proxy.

Decided: the proxy ranks candidate objectives before LoRA training. It is never reported as a
result and never optimized. Overturned if a LoRA-evaluated condition lands outside the calibration
trend by more than the per-style scatter.
