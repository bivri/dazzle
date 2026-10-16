"""Screening proxy: style-specificity drop of the protected training images themselves.

drop(a) = 1 - mean M(protected_i, a) / mean M(clean_i, a),  M(x, a) = S(x, a) - mean_{b!=a} S(x, b)

No LoRA is trained, so a candidate costs seconds instead of hours. Calibrated on 14 earlier
conditions (62 style points): Spearman 0.81 per style, Pearson 0.91 per condition, E/F ~ 0.8 x drop
(docs/decisions-log/evaluation.md). It ranks candidates only; claims use LoRA E/F.
"""

import numpy as np


def specificity(emb: np.ndarray, refs: dict[str, np.ndarray], style: str) -> np.ndarray:
    s = {b: (emb @ r.T).mean(1) for b, r in refs.items()}
    return s[style] - np.mean([s[b] for b in refs if b != style], axis=0)


def specificity_drop(clean: np.ndarray, protected: np.ndarray, refs: dict[str, np.ndarray], style: str) -> float:
    """clean and protected are row-aligned embeddings of the same training images."""
    return float(1 - specificity(protected, refs, style).mean() / specificity(clean, refs, style).mean())
