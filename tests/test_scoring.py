import numpy as np
import pytest

from dazzle.evaluation.proxy import specificity_drop
from dazzle.evaluation.score import effect_ratio

STYLES = ("s1", "s2", "s3", "s4")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    centers = {s: np.eye(16)[i] for i, s in enumerate(STYLES)}  # orthonormal styles
    refs = {s: _unit(centers[s] + 0.3 * rng.normal(size=(10, 16))) for s in STYLES}
    neutral = _unit(rng.normal(size=16))
    base = {s: _unit(neutral + 0.3 * rng.normal(size=(40, 16))) for s in STYLES}
    clean = {s: _unit(neutral + 2.0 * centers[s] + 0.3 * rng.normal(size=(40, 16))) for s in STYLES}
    return centers, refs, base, clean


def test_condition_equal_to_unprotected_lora_has_zero_effect(world):
    _, refs, base, clean = world
    res = effect_ratio(base, clean, clean, refs, draws=50)
    assert abs(res["mean"][0]) < 1e-9


def test_condition_equal_to_base_model_removes_all_style_gain(world):
    _, refs, base, clean = world
    res = effect_ratio(base, clean, base, refs, draws=50)
    assert res["mean"][0] == pytest.approx(1.0)
    assert res["s1"][1] <= res["s1"][0] <= res["s1"][2]


def test_partial_condition_lands_between_and_mean_needs_all_styles(world):
    centers, refs, base, clean = world
    half = {s: _unit(0.5 * (base[s] + clean[s])) for s in ("s1", "s2")}
    res = effect_ratio(base, clean, half, refs, draws=50)
    assert "mean" not in res
    assert 0.1 < res["s1"][0] < 0.9


def test_proxy_is_zero_for_unchanged_images_and_exceeds_one_when_images_carry_another_style(world):
    centers, refs, _, _ = world
    rng = np.random.default_rng(1)
    imgs = _unit(centers["s1"] + 0.3 * rng.normal(size=(30, 16)))
    assert specificity_drop(imgs, imgs, refs, "s1") == pytest.approx(0.0)
    other = _unit(centers["s2"] + 0.3 * rng.normal(size=(30, 16)))
    assert specificity_drop(imgs, other, refs, "s1") > 1.0
