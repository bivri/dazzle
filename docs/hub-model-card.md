---
license: apache-2.0
library_name: dazzle
tags:
- image-protection
- style-mimicry
- protective-pattern
---

# Dazzle generator {version}

A {params:,}-parameter fully convolutional U-Net that adds a bounded protective pattern to an
illustration. The pattern is meant to reduce how much of the artist's style an SDXL LoRA learns
when the image is used for unauthorized style-mimicry fine-tuning. One forward pass, any resolution
whose sides are multiples of 8 (the library pads and crops other sizes).

Dazzle is developed by [Bivri (비브리)](https://bivri.app), an illustration community built on the
AT Protocol where artists share artwork and exchange commissions.

- Code: https://github.com/bivri/dazzle (tag `{version}`, built from `{code_version}`)
- Output: `delta = epsilon * tanh(G(x))`, default bound {epsilon_levels} of 255 levels per channel

## Use

```python
from PIL import Image
from dazzle import DazzleGenerator, protect_image

gen = DazzleGenerator.from_pretrained("bivri/dazzle-generator", revision="{version}")
protect_image(Image.open("art.png"), gen).save("art.protected.png")
```

## Evaluation

Evaluation run `{run_id}`: for each of four synthetic illustration styles, one rank-16 SDXL LoRA
(Illustrious-XL v2.0, 1,500 steps, kohya-ss/sd-scripts) was trained on the protected set, optionally
after a transformation filter, and 40 renders were compared with an unprotected LoRA's renders in
CSD style space. E/F is the fraction of the unprotected LoRA's style-specific gain that the
protected set removed: 1 removes all of it, 0 none. LPIPS is measured at native resolution against
the original images.

{results}

## Limitations

- Results are from four synthetic styles, one LoRA recipe and one training seed per condition. CSD
  similarity is not a human judgment.
- A transformation filter or a differently configured trainer can reduce the effect. The pattern
  does not make mimicry impossible.
- The pattern is visible on close inspection of flat regions at the default bound.
