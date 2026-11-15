"""Properties of DSC that must hold whatever the data."""
import numpy as np
import torch
import torch.nn.functional as F

from dsc.method import DSC, MERGED, centre_by_group, dsc, gate_open, merge_small, name_depictions


def test_groups_have_zero_mean_after_centring():
    x = torch.randn(300, 8)
    g = np.repeat([0, 1, 2], 100)
    z = centre_by_group(x, g)
    for w in range(3):
        assert torch.allclose(z[torch.tensor(g == w)].mean(0), torch.zeros(8), atol=1e-6)


def test_gate_closed_returns_normalised_input():
    x = torch.randn(100, 8)
    g = np.array([0] * 80 + [1] * 20)                # largest group 0.8 >= 2/3
    assert not gate_open(g)
    assert torch.allclose(dsc(x, g), F.normalize(x, dim=1))


def test_gate_open_centres():
    x = torch.randn(90, 8)
    g = np.repeat([0, 1, 2], 30)
    assert gate_open(g) and torch.allclose(dsc(x, g), centre_by_group(x, g))


def test_small_groups_are_merged():
    g = merge_small(np.array([0] * 50 + [1] * 5 + [2] * 30 + [3] * 3), min_group=20)
    assert set(g) == {0, 2, MERGED} and (g == MERGED).sum() == 8


def test_naming_picks_closest_word():
    words = F.normalize(torch.eye(4)[:3], dim=1)
    img = torch.tensor([[1.0, .1, 0, 0], [0, 1.0, .1, 0], [0, 0, 1.0, .1]]).repeat_interleave(25, 0)
    assert list(np.unique(name_depictions(img, words))) == [0, 1, 2]


def test_fit_transform_matches_functional_form():
    torch.manual_seed(0)
    words = F.normalize(torch.randn(3, 6), dim=1)
    img = words.repeat_interleave(40, 0) + 0.01 * torch.randn(120, 6)
    x = torch.randn(120, 16)
    d = DSC().fit(x, img, words)
    assert d.active and torch.allclose(d.transform(x), dsc(x, d.groups), atol=1e-6)
    # a query identical to collection image 0 is corrected with the mean of its group
    assert torch.allclose(d.transform_queries(x[:1], img[:1]), d.transform(x)[:1], atol=1e-6)
