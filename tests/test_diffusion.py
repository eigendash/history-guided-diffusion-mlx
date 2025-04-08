import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from hgd.diffusion import (
    add_noise,
    alpha_sigma,
    binary_dropout_levels,
    eps_from_v,
    independent_levels,
    training_loss,
    v_target,
    x0_from_v,
)
from hgd.model import Config, VideoDiT


def tiny():
    return VideoDiT(Config(size=8, frames=4, patch=2, dim=32, heads=2, depth=2))


def test_schedule_is_variance_preserving_with_the_right_endpoints():
    k = mx.linspace(0, 1, 11)
    a, s = alpha_sigma(k)
    np.testing.assert_allclose(np.array(a**2 + s**2), 1.0, atol=1e-6)
    assert float(a[0]) == 1.0 and float(s[0]) == 0.0
    assert float(a[-1]) < 1e-6 and float(s[-1]) == pytest.approx(1.0)


def test_v_parameterisation_inverts_exactly():
    mx.random.seed(0)
    x0, eps = mx.random.normal((3, 4, 8, 8, 1)), mx.random.normal((3, 4, 8, 8, 1))
    k = mx.random.uniform(0.05, 0.95, (3, 4))
    xk, v = add_noise(x0, k, eps), v_target(x0, k, eps)
    np.testing.assert_allclose(np.array(x0_from_v(xk, k, v)), np.array(x0), atol=1e-5)
    np.testing.assert_allclose(np.array(eps_from_v(xk, k, v)), np.array(eps), atol=1e-5)


def test_each_frame_is_noised_at_its_own_level():
    x0 = mx.ones((1, 3, 4, 4, 1))
    eps = mx.full(x0.shape, 2.0)
    out = np.array(add_noise(x0, mx.array([[0.0, 1.0, 0.5]]), eps))
    np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-6)  # clean
    np.testing.assert_allclose(out[0, 1], 2.0, atol=1e-5)  # pure noise
    mid = np.cos(np.pi / 4) + 2 * np.sin(np.pi / 4)
    np.testing.assert_allclose(out[0, 2], mid, atol=1e-5)


def test_independent_levels_differ_across_frames_and_supervise_all():
    levels, weight = independent_levels(64, 8)
    assert levels.shape == (64, 8) and float(weight.min()) == 1.0
    assert float(levels.std(axis=1).mean()) > 0.15  # not one shared level per clip


def test_binary_dropout_levels_have_a_clean_prefix_and_a_shared_level():
    mx.random.seed(1)
    levels, weight = binary_dropout_levels(512, 8, drop=0.25)
    lv, w = np.array(levels), np.array(weight)
    for row, wrow in zip(lv, w):
        history = wrow == 0
        assert history.sum() < 8
        assert np.all(np.diff(history.astype(int)) <= 0)  # history is a prefix
        assert len(set(row[~history].round(6))) == 1  # generated frames share one level
        assert np.all((row[history] == 0) | (row[history] == 1))
    n_hist = (w == 0).any(axis=1)
    masked = np.array([(row[wrow == 0] == 1).all() for row, wrow in zip(lv, w) if (wrow == 0).any()])
    assert 0.15 < masked.mean() < 0.35 and n_hist.mean() > 0.8


def test_model_shape_and_zero_init_blocks():
    mx.random.seed(0)
    model = tiny()
    x, k = mx.random.normal((2, 4, 8, 8, 1)), mx.random.uniform(shape=(2, 4))
    assert model(x, k).shape == x.shape
    h = mx.random.normal((2, 4, 16, 32))
    for block in model.blocks:
        np.testing.assert_allclose(np.array(block(h, mx.zeros((2, 4, 32)))), np.array(h), atol=1e-6)


def test_patchify_round_trips():
    model = tiny()
    x = mx.random.normal((2, 4, 8, 8, 1))
    np.testing.assert_allclose(np.array(model.unpatchify(model.patchify(x))), np.array(x), atol=1e-6)


def test_model_is_non_causal_and_reads_the_noise_level_of_other_frames():
    mx.random.seed(2)
    model = tiny()
    # make the blocks non-trivial: adaLN-zero would otherwise hide every dependency
    for block in model.blocks:
        block.mod.weight = mx.random.normal(block.mod.weight.shape) * 0.05
    model.final_mod.weight = mx.random.normal(model.final_mod.weight.shape) * 0.05
    x = mx.random.normal((1, 4, 8, 8, 1))
    k = mx.full((1, 4), 0.5)
    base = np.array(model(x, k))
    later = mx.concatenate([x[:, :3], mx.random.normal((1, 1, 8, 8, 1))], axis=1)
    assert not np.allclose(np.array(model(later, k))[:, 0], base[:, 0], atol=1e-5)  # future -> past
    k2 = mx.array([[0.5, 0.5, 0.5, 0.9]])
    assert not np.allclose(np.array(model(x, k2))[:, 0], base[:, 0], atol=1e-5)  # level of frame 3


def test_loss_has_finite_gradients_and_respects_the_weight_mask():
    mx.random.seed(3)
    model = tiny()
    x0 = mx.random.normal((4, 4, 8, 8, 1))
    levels, weight = independent_levels(4, 4)
    loss, grads = nn.value_and_grad(model, lambda m: training_loss(m, x0, levels, weight))(model)
    assert np.isfinite(float(loss))
    zero = mx.zeros_like(weight)
    assert float(training_loss(model, x0, levels, zero)) == 0.0
