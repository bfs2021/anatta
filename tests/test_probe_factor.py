import torch

from anatta.factor import composite, unity
from anatta.probe import Direction, Trait, extract, list_traits


def test_shipped_traits_load():
    names = list_traits()
    assert set(list_traits("ego")) == {"self_preservation", "goal_fixation", "face_saving",
                                       "approval_seeking", "power_seeking", "desperation"}
    for n in names:
        t = Trait.load(n)
        assert len(t.pairs) >= 4 and len(t.questions) >= 8
        assert all("pos" in p and "neg" in p for p in t.pairs)


def test_extract_roundtrip(tiny, tmp_path):
    model, tok = tiny
    t = Trait.load("desperation")
    t.pairs, t.questions = t.pairs[:2], t.questions[:4]
    d = extract(model, tok, t, layers=[1, 2], max_new_tokens=8, batch_size=8)
    for v in d.vectors.values():
        assert abs(float(v.norm()) - 1) < 1e-4
    assert d.best_layer in (1, 2)
    d.save(tmp_path / "x.pt")
    d2 = Direction.load(tmp_path / "x.pt")
    assert torch.allclose(d2.vectors[1], d.vectors[1])


def _dir(name, v):
    v = v / v.norm()
    return Direction(name, {0: v}, {0: 1.0}, {0: 1.0}, {0: 0.0})


def test_unity_detects_shared_axis():
    g = torch.Generator().manual_seed(0)
    shared = torch.randn(256, generator=g)
    dirs = {f"e{i}": _dir(f"e{i}", shared + 0.5 * torch.randn(256, generator=g)) for i in range(5)}
    dirs |= {f"n{i}": _dir(f"n{i}", torch.randn(256, generator=g)) for i in range(2)}
    fam = {n: ("ego" if n.startswith("e") else "null") for n in dirs}
    r = unity(dirs, fam, 0)
    assert r.pc1_share > 0.6 and r.mean_ego_cos > 0.5 > r.mean_null_cos
    v = composite(dirs, [n for n in dirs if n.startswith("e")], [0])
    assert float(v.vectors[0] @ (shared / shared.norm())) > 0.8
