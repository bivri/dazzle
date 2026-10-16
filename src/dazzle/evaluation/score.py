"""E/F: how much of the unprotected LoRA's style gain a protected condition removes.

S(x, b)  = mean cosine between render x and the held-out references of style b (CSD space)
F(a, L)  = mean over renders of [S(L, a) - S(base, a)] - mean_{b != a} [S(L, b) - S(base, b)]
E/F      = (F(a, clean) - F(a, condition)) / F(a, clean)

1.0 means the condition removed all style-specific gain; 0 means no effect; negative means the
condition LoRA learned the style more. Confidence intervals come from a bootstrap over reference
images and over renders within each prompt.
"""

import numpy as np

from dazzle.evaluation.render import SEEDS


def style_gain(base: np.ndarray, lora: np.ndarray, refs: dict[str, np.ndarray], style: str, ri, pi) -> float:
    def s(x: np.ndarray, b: str) -> np.ndarray:
        return (x[pi] @ refs[b][ri[b]].T).mean(1)

    d_tgt = s(lora, style) - s(base, style)
    d_ref = np.mean([s(lora, b) - s(base, b) for b in refs if b != style], axis=0)
    return float((d_tgt - d_ref).mean())


def effect_ratio(
    base: dict[str, np.ndarray],
    clean: dict[str, np.ndarray],
    cond: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    draws: int = 500,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Per-style and mean E/F as [estimate, 2.5 %, 97.5 %]. Inputs map style -> embeddings; render
    rows are ordered prompt-major with len(SEEDS) seeds per prompt (render.render_names order)."""
    styles = list(cond)
    n = next(iter(base.values())).shape[0]
    prompt_of = np.arange(n) // len(SEEDS)
    rng = np.random.default_rng(seed)
    scores = {s: [] for s in styles}
    for _ in range(draws):
        ri = {b: rng.choice(len(r), len(r), replace=True) for b, r in refs.items()}
        pi = np.concatenate(
            [
                rng.choice(np.flatnonzero(prompt_of == u), (prompt_of == u).sum(), replace=True)
                for u in np.unique(prompt_of)
            ]
        )
        for s in styles:
            fc = style_gain(base[s], clean[s], refs, s, ri, pi)
            scores[s].append((fc - style_gain(base[s], cond[s], refs, s, ri, pi)) / max(fc, 1e-6))
    res = {s: _summary(np.array(v)) for s, v in scores.items()}
    if len(styles) == len(refs):
        res["mean"] = _summary(np.mean([scores[s] for s in styles], axis=0))
    return res


def _summary(v: np.ndarray) -> list[float]:
    return [float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
