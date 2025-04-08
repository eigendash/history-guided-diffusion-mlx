"""Forward process, parameterisation and training-time noise levels.

Frames carry their own noise level k_t in [0, 1], with x^k = a(k) x + s(k) eps and a
variance-preserving cosine schedule a = cos(pi k / 2), s = sin(pi k / 2). Level 0 is a
clean frame and level 1 is pure noise, so a frame's level doubles as a mask: noising a
frame fully removes it from the context.

The network predicts v = a eps - s x. That is an affine function of the score, as the
noise prediction would be, but stays well behaved near k = 1 where recovering x from
eps divides by a tiny a.
"""

import math

import mlx.core as mx


def alpha_sigma(k):
    return mx.cos(k * math.pi / 2), mx.sin(k * math.pi / 2)


def _frames(x):
    """(B, T) -> (B, T, 1, 1, 1) so a per-frame scalar broadcasts over pixels."""
    return x[..., None, None, None]


def add_noise(x0, k, eps):
    a, s = alpha_sigma(k)
    return _frames(a) * x0 + _frames(s) * eps


def v_target(x0, k, eps):
    a, s = alpha_sigma(k)
    return _frames(a) * eps - _frames(s) * x0


def x0_from_v(xk, k, v):
    a, s = alpha_sigma(k)
    return _frames(a) * xk - _frames(s) * v


def eps_from_v(xk, k, v):
    a, s = alpha_sigma(k)
    return _frames(s) * xk + _frames(a) * v


def independent_levels(batch, frames):
    """Diffusion forcing: every frame gets its own uniform noise level.

    Returns (levels (B, T), loss_weight (B, T)). All frames are supervised.
    """
    return mx.random.uniform(shape=(batch, frames)), mx.ones((batch, frames))


def binary_dropout_levels(batch, frames, drop=0.15):
    """The conventional recipe, as a baseline.

    A random-length prefix is history held at level 0 and every other frame shares one
    noise level. With probability `drop` the whole history is masked (level 1), which
    is what trains the unconditional score. Only the generated frames are supervised.
    """
    length = mx.random.randint(0, frames, (batch, 1))
    shared = mx.random.uniform(shape=(batch, 1))
    dropped = mx.random.uniform(shape=(batch, 1)) < drop
    is_history = mx.arange(frames)[None] < length
    levels = mx.where(is_history, mx.where(dropped, 1.0, 0.0), shared)
    return levels, (~is_history).astype(mx.float32)


def training_loss(model, x0, levels, weight):
    """x0: (B, T, H, W, C) in [-1, 1]. Noises each frame at its own level."""
    eps = mx.random.normal(x0.shape)
    pred = model(add_noise(x0, levels, eps), levels)
    err = ((pred - v_target(x0, levels, eps)) ** 2).mean(axis=(2, 3, 4))
    return (err * weight).sum() / mx.maximum(weight.sum(), 1.0)
