import mlx.core as mx
import numpy as np
import pytest

from hgd.diffusion import alpha_sigma
from hgd.guidance import (
    context_inputs,
    fractional,
    guided_v,
    plain,
    rollout,
    sample,
    temporal,
    vanilla,
)


class Oracle:
    """Knows the true clean video, so the exact v for any input and levels."""

    def __init__(self, truth):
        self.truth = truth
        self.calls = 0

    def __call__(self, x, levels):
        self.calls += 1
        a, s = alpha_sigma(levels)
        a, s = a[..., None, None, None], s[..., None, None, None]
        eps = (x - a * self.truth) / mx.maximum(s, 1e-6)
        return a * eps - s * self.truth


class Counter:
    """Returns a constant per clip from how many frames are clean vs partly noised."""

    def __call__(self, x, levels):
        clean = (levels == 0).sum(axis=1).astype(mx.float32)
        partial = ((levels > 0) & (levels < 0.6)).sum(axis=1).astype(mx.float32)
        value = clean + 0.1 * partial
        return mx.broadcast_to(value[:, None, None, None, None], x.shape)


def video(frames=6, seed=0):
    mx.random.seed(seed)
    return mx.random.uniform(-1, 1, (2, frames, 4, 4, 1))


def test_plain_guidance_with_the_oracle_recovers_the_generated_frames():
    truth = video()
    clean = mx.where(mx.arange(6)[None, :, None, None, None] < 3, truth, 0.0)
    out = sample(Oracle(truth), clean, [3, 4, 5], plain(6, {0, 1, 2}), steps=12)
    np.testing.assert_allclose(np.array(out), np.array(truth), atol=1e-4)


def test_history_frames_are_returned_untouched():
    truth = video()
    clean = mx.where(mx.arange(6)[None, :, None, None, None] < 3, truth, 0.0)
    out = sample(Oracle(truth), clean, [3, 4, 5], plain(6, {0, 1, 2}), steps=4)
    np.testing.assert_array_equal(np.array(out[:, :3]), np.array(truth[:, :3]))


def test_context_masking_levels():
    clean = mx.ones((1, 3, 2, 2, 1))
    noise = mx.full(clean.shape, 5.0)
    gen = mx.array([False, False, True])
    noisy = mx.full(clean.shape, -7.0)
    x, lv = context_inputs(clean, gen, noisy, 0.8, [0.0, 1.0, 0.3], noise)
    x = np.array(x)
    np.testing.assert_allclose(x[0, 0], 1.0, atol=1e-6)  # clean history
    np.testing.assert_allclose(x[0, 1], 5.0, atol=1e-5)  # masked history is the fixed noise
    np.testing.assert_allclose(x[0, 2], -7.0)  # generated frame passes through
    np.testing.assert_allclose(np.array(lv)[0], [0.0, 1.0, 0.8])
    # a partial level keeps a(k) of the frame and s(k) of the noise
    x2, _ = context_inputs(clean, mx.array([False, False, True]), noisy, 0.8, [0.0, 0.3, 0.3], noise)
    a, s = alpha_sigma(mx.array(0.3))
    np.testing.assert_allclose(np.array(x2)[0, 1], float(a) + 5 * float(s), atol=1e-5)


def test_vanilla_guidance_is_cfg_on_the_history():
    clean, gen, noisy = video(4), mx.array([False, False, True, True]), video(4, 1)
    ctx = mx.zeros_like(clean)
    n_hist = 2
    for omega in (0.0, 1.0, 2.5):
        v = guided_v(Counter(), clean, gen, noisy, 0.7, vanilla(4, {0, 1}, omega), ctx)
        np.testing.assert_allclose(np.array(v)[0, 0, 0, 0, 0], omega * n_hist, atol=1e-5)


def test_plain_guidance_skips_the_unconditional_pass():
    clean, gen, noisy = video(4), mx.array([False, False, True, True]), video(4, 1)
    oracle = Oracle(clean)
    guided_v(oracle, clean, gen, noisy, 0.7, plain(4, {0, 1}), mx.zeros_like(clean))
    assert oracle.calls == 1
    oracle.calls = 0
    guided_v(oracle, clean, gen, noisy, 0.7, vanilla(4, {0, 1}, 2.0), mx.zeros_like(clean))
    assert oracle.calls == 2


def test_fractional_guidance_matches_the_paper_formula():
    clean, gen, noisy = video(4), mx.array([False, False, True, True]), video(4, 1)
    omega, level = 1.5, 0.4
    v = guided_v(Counter(), clean, gen, noisy, 0.7, fractional(4, {0, 1}, omega, level), mx.zeros_like(clean))
    cond, partial, uncond = 2.0, 0.1 * 2, 0.0
    expected = cond + omega * (partial - uncond)
    np.testing.assert_allclose(np.array(v)[0, 0, 0, 0, 0], expected, atol=1e-5)


def test_temporal_guidance_composes_sub_histories_with_their_weights():
    clean, gen, noisy = video(5), mx.array([False, False, False, True, True]), video(5, 1)
    parts = [({0, 1, 2}, 0.7), ({1, 2}, 0.5), ({2}, 0.3)]
    v = guided_v(Counter(), clean, gen, noisy, 0.7, temporal(5, parts), mx.zeros_like(clean))
    expected = 0.7 * 3 + 0.5 * 2 + 0.3 * 1
    np.testing.assert_allclose(np.array(v)[0, 0, 0, 0, 0], expected, atol=1e-5)


def test_guided_v_equals_conditional_when_weights_sum_to_one():
    clean, gen, noisy = video(4), mx.array([False, False, True, True]), video(4, 1)
    ctx = mx.zeros_like(clean)
    cond = guided_v(Counter(), clean, gen, noisy, 0.7, plain(4, {0, 1}), ctx)
    split = guided_v(Counter(), clean, gen, noisy, 0.7, temporal(4, [({0, 1}, 0.5), ({0, 1}, 0.5)]), ctx)
    np.testing.assert_allclose(np.array(split), np.array(cond), atol=1e-5)


class Zero:
    def __call__(self, x, levels):
        return mx.zeros_like(x)


def test_rollout_extends_by_the_window_step_and_keeps_the_start():
    start = video(4)
    out = rollout(Zero(), start, 12, window=8, history=4, make_guidance=lambda T, h: plain(T, set(h)), steps=4)
    assert out.shape == (2, 12, 4, 4, 1)
    np.testing.assert_array_equal(np.array(out[:, :4]), np.array(start))
    assert np.isfinite(np.array(out)).all()


def test_rollout_with_a_trimmed_final_window_returns_exactly_total_frames():
    out = rollout(Zero(), video(4), 10, window=8, history=4, make_guidance=lambda T, h: plain(T, set(h)), steps=3)
    assert out.shape[1] == 10


def test_rollout_conditions_each_round_on_the_latest_frames():
    seen = []

    class Spy:
        def __call__(self, x, levels):
            seen.append(np.array(x[:, :4]))
            return mx.zeros_like(x)

    start = video(4)
    rollout(Spy(), start, 12, window=8, history=4, make_guidance=lambda T, h: plain(T, set(h)), steps=2)
    np.testing.assert_allclose(seen[0], np.array(start), atol=1e-6)  # round one sees the start
    assert not np.allclose(seen[-1], np.array(start))  # later rounds see generated frames
