# Dazzle domain language

Terms used in code, docs, commit messages and reports. `scripts/check_policy.py` rejects the
_Avoid_ forms; a quoted third-party title may keep its own wording on a line marked
`policy: allow-term`.

| Term | Meaning | _Avoid_ |
| --- | --- | --- |
| protective pattern / protective signal | The bounded image change `delta` Dazzle adds. | adversarial perturbation, perturbation, noise |
| protected image, protected set | An image with the protective pattern; the training images of one style after protection. | poisoned image |
| unauthorized style mimicry | Fine-tuning a model (here a LoRA) on an artist's images to reproduce their style without consent. | attack, adversarial attack |
| mimicry recipe | The fixed LoRA training configuration used to measure protection (`dazzle.evaluation.lora`). | attacker, attack model |
| transformation filter | A pre-processing step applied to collected images before training: JPEG re-encoding, resizing, SDXL VAE round trip, upscaling. | purification, purification attack |
| transformation robustness | How much of the effect survives a transformation filter. | purification resistance |
| insufficient signal | A protective pattern too weak to change what the LoRA learns. | under-perturbation |
| decoy | A rendering of the same picture in a different style; the teacher objective moves the protected image's VAE latent toward the decoy's. | target image |
| teacher objective | The offline loss the generator is trained to minimize (`dazzle.training`). Per-image optimization of the same loss is its upper bound. | |
| generator | `DazzleGenerator`: one forward pass producing the protective pattern, single output head. | student (outside training code) |
| epsilon | Per-channel bound of the pattern; in code in [-1, 1] units, in user-facing options in 8-bit levels. | strength |
| E/F | Fraction of the unprotected LoRA's style-specific gain removed by protection; the primary effect measure (`dazzle.evaluation.score`). | defense rate, 방어율 |
| proxy | Style-specificity drop of the protected training images themselves; ranks candidates without training a LoRA (`dazzle.evaluation.proxy`). Never a reported result. | |
| run | One evaluation of one or more protected sets, identified by a run id; outputs in `$DAZZLE_WORKSPACE/runs/<run-id>/`. | |
| experiment | A pre-registered question answered by runs, developed on an `exp/` branch and recorded in `experiments/`. | phase, p6, p7 |
