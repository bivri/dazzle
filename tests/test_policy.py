import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("check_policy", Path(__file__).parents[1] / "scripts" / "check_policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize(
    "line",
    [
        "mask = compute_spatial_mask(x)",
        "tex = texture_mask_map(img)",
        "d = lp(x, y, spatial=True)",
        "pattern_bank = torch.load(p)",
        "out = routing_head(feat)",
    ],
)
def test_code_rules_reject_mask_and_bank_constructs(line):
    assert policy.violations("src/x.py", line)


def test_code_rules_apply_only_to_python_files():
    assert policy.violations("docs/x.md", "a spatial=True example") == []


@pytest.mark.parametrize("line", ["an adversarial example", "purification attack", "under-perturbation", "적대적 섭동"])
def test_wording_rules_reject_forbidden_vocabulary(line):
    assert policy.violations("docs/x.md", line)


def test_wording_marker_allows_a_quoted_title_but_not_code_constructs():
    assert policy.violations("docs/x.md", "Anti-DreamBooth (adversarial) policy: allow-term") == []
    assert policy.violations("src/x.py", "m = spatial_mask(x)  # policy: allow-term")


def test_compliant_text_passes():
    text = "delta = eps * tanh(G(x)) is the protective pattern; JPEG is a transformation filter."
    assert policy.violations("src/x.py", text) == []
