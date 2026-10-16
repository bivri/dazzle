import numpy as np
import torch
from PIL import Image

from dazzle import DazzleGenerator, levels_to_epsilon, protect_image


def _gen() -> DazzleGenerator:
    torch.manual_seed(0)
    g = DazzleGenerator(base_ch=8, default_epsilon=levels_to_epsilon(8)).eval()
    with torch.no_grad():
        g.out_conv.bias.fill_(5.0)  # saturate tanh so the bound is actually reached
    return g


def test_delta_is_bounded_by_epsilon_and_keeps_shape():
    g = _gen()
    x = torch.rand(2, 3, 64, 96) * 2 - 1
    with torch.no_grad():
        d = g(x)
        d4 = g(x, levels_to_epsilon(4))
    assert d.shape == x.shape
    assert d.abs().max() <= levels_to_epsilon(8) + 1e-6
    assert d.abs().max() > levels_to_epsilon(8) * 0.99
    assert d4.abs().max() <= levels_to_epsilon(4) + 1e-6


def test_save_and_load_pretrained_round_trip(tmp_path):
    g = _gen()
    g.save_pretrained(tmp_path)
    loaded = DazzleGenerator.from_pretrained(tmp_path).eval()
    assert loaded.default_epsilon == g.default_epsilon
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        assert torch.equal(g(x), loaded(x))


def test_protect_image_keeps_odd_size_and_level_bound():
    g = _gen()
    rng = np.random.default_rng(0)
    im = Image.fromarray(rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))
    out = protect_image(im, g)
    assert out.size == im.size
    diff = np.abs(np.asarray(out, np.int16) - np.asarray(im, np.int16))
    assert diff.max() <= 8 + 1  # 8 levels plus rounding
    assert diff.max() >= 7


def test_target_texture_and_frequency_bounds_preserve_white_paper():
    from dazzle.training import (
        compute_frequency_bounds,
        generate_structural_texture,
        make_counter_bias_image,
        multi_band_hinge_loss,
    )

    tex = generate_structural_texture(size=64, seed=42)
    assert tex.shape == (1, 3, 64, 64)
    assert tex.abs().max() <= 1.0

    # Pure white paper (sRGB = +1.0) must have strictly zero HF/Mid/LF allowance
    x_white = torch.ones(1, 3, 64, 64)
    hf_allow, mid_allow, lf_allow, _ = compute_frequency_bounds(x_white)
    assert float(hf_allow.max()) == 0.0
    assert float(mid_allow.max()) == 0.0
    assert float(lf_allow.max()) == 0.0

    x_rand = torch.rand(2, 3, 64, 64) * 2 - 1
    cb = make_counter_bias_image(x_rand)
    assert cb.shape == x_rand.shape
    loss = multi_band_hinge_loss(x_rand, 0.05 * torch.randn_like(x_rand))
    assert loss.ndim == 0 and float(loss) >= 0.0
