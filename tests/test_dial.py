import torch

from anatta.dial import Intervention, intervene, random_like
from anatta.probe import capture


def _ids(tok):
    return tok("the quick brown fox", return_tensors="pt")["input_ids"]


def _read(model, tok, layer, ivs=()):
    with intervene(model, list(ivs)), capture(model, [layer]) as store:
        model(_ids(tok))
    return store[layer][0]


def test_steer_adds_alpha_d(tiny):
    model, tok = tiny
    d = torch.randn(64)
    base = _read(model, tok, 1)
    steered = _read(model, tok, 1, [Intervention(1, d, "steer", alpha=3.0)])
    assert torch.allclose(steered - base, 3.0 * d / d.norm(), atol=1e-4)


def test_ablate_removes_component(tiny):
    model, tok = tiny
    d = torch.randn(64)
    h = _read(model, tok, 2, [Intervention(2, d, "ablate")])
    assert (h @ (d / d.norm())).abs().max() < 1e-4


def test_cap_bounds_projection(tiny):
    model, tok = tiny
    d = torch.randn(64)
    u = d / d.norm()
    base = _read(model, tok, 2) @ u
    hi = float(base.median())
    capped = _read(model, tok, 2, [Intervention(2, d, "cap", hi=hi)]) @ u
    assert capped.max() <= hi + 1e-4
    below = base <= hi
    assert torch.allclose(capped[below], base[below], atol=1e-4)


def test_hooks_removed_after_block(tiny):
    model, tok = tiny
    before = _read(model, tok, 1)
    _read(model, tok, 1, [Intervention(1, torch.randn(64), "steer", alpha=10.0)])
    assert torch.allclose(_read(model, tok, 1), before)


def test_random_control_is_unit():
    r = random_like(torch.randn(64), seed=1)
    assert abs(float(r.norm()) - 1) < 1e-5
