"""History guidance and sampling.

A "context" is a vector of per-frame noise levels for the history frames. Level 0 shows
a history frame clean, level 1 replaces it with pure noise (masked out), and anything in
between shows a partly noised copy, which keeps only its coarse content.

Every guided estimate is a base score plus weighted differences,

    v_hat = v(masked history) + sum_i w_i * ( v(context_i) - v(masked history) ),

which covers all the variants:

    plain       one term, weight 1                 the conditional score
    vanilla     one term, weight w                 classifier-free guidance on history
    fractional  clean context (1) + level-k context (w)
    temporal    several clean sub-histories, weights w_i

Score, noise prediction and v are affine in each other with weights that sum to one, so
combining v directly is the same as combining scores.
"""

from dataclasses import dataclass, field

import mlx.core as mx

from .diffusion import alpha_sigma, x0_from_v


@dataclass
class Term:
    weight: float
    levels: list  # per-frame level of every frame; entries at generated positions are ignored


@dataclass
class Guidance:
    terms: list = field(default_factory=list)

    def needs_base(self):
        return not (len(self.terms) == 1 and abs(self.terms[0].weight - 1.0) < 1e-9)


def _levels(frames, history, level_of):
    return [level_of(t) if t in history else 1.0 for t in range(frames)]


def plain(frames, history):
    return Guidance([Term(1.0, _levels(frames, history, lambda t: 0.0))])


def vanilla(frames, history, omega):
    return Guidance([Term(omega, _levels(frames, history, lambda t: 0.0))])


def fractional(frames, history, omega, level):
    return Guidance(
        [
            Term(1.0, _levels(frames, history, lambda t: 0.0)),
            Term(omega, _levels(frames, history, lambda t: level)),
        ]
    )


def temporal(frames, parts):
    """parts: list of (set of history frame indices, weight)."""
    return Guidance([Term(w, _levels(frames, set(h), lambda t: 0.0)) for h, w in parts])


def context_inputs(clean, generating, noisy, k, levels, context_noise):
    """Assemble one network input.

    clean: (B, T, H, W, C) frames with history filled in; generating: (T,) bool;
    noisy: current noisy frames at the generated positions; k: scalar level of those;
    levels: (T,) per-frame level for history frames; context_noise: fixed noise used to
    partly or fully mask history frames.
    """
    a, s = alpha_sigma(mx.array(levels))
    shown = a[None, :, None, None, None] * clean + s[None, :, None, None, None] * context_noise
    gen = generating[None, :, None, None, None]
    x = mx.where(gen, noisy, shown)
    per_frame = mx.where(generating, k, mx.array(levels))
    return x, mx.broadcast_to(per_frame[None], (clean.shape[0], clean.shape[1]))


def guided_v(model, clean, generating, noisy, k, guidance, context_noise):
    def run(levels):
        x, lv = context_inputs(clean, generating, noisy, k, levels, context_noise)
        return model(x, lv)

    if not guidance.needs_base():
        return run(guidance.terms[0].levels)
    frames = clean.shape[1]
    base = run([1.0] * frames)
    out = base
    for term in guidance.terms:
        out = out + term.weight * (run(term.levels) - base)
    return out


def sample(model, clean, generate, guidance, steps=30, clip=True):
    """DDIM from pure noise over the frames in `generate`, holding the rest as history.

    clean: (B, T, H, W, C); the frames at `generate` are ignored. Returns the same shape
    with those frames filled in.
    """
    B, T = clean.shape[:2]
    generating = mx.array([t in set(generate) for t in range(T)])
    context_noise = mx.random.normal(clean.shape)
    x = mx.random.normal(clean.shape)
    for i in range(steps):
        k, k_next = 1.0 - i / steps, 1.0 - (i + 1) / steps
        kv = mx.full((B, T), k)
        v = guided_v(model, clean, generating, x, k, guidance, context_noise)
        x0 = x0_from_v(x, kv, v)
        if clip:
            x0 = mx.clip(x0, -1.0, 1.0)
        a, s = alpha_sigma(mx.array(k))
        eps = (x - a * x0) / s  # k >= 1/steps, so s > 0; equals eps_from_v when x0 is unclipped
        a2, s2 = alpha_sigma(mx.array(k_next))
        x = a2 * x0 + s2 * eps
        mx.eval(x)
    return mx.where(generating[None, :, None, None, None], x, clean)


def rollout(model, start, total, window, history, make_guidance, steps=30):
    """Extend `start` (B, T0, H, W, C) to `total` frames with a sliding window.

    Each round conditions on the last `history` frames and generates the remaining
    `window - history` positions of a window.
    """
    frames = start
    step = window - history
    while frames.shape[1] < total:
        context = frames[:, -history:]
        pad = mx.zeros((frames.shape[0], step, *frames.shape[2:]))
        clean = mx.concatenate([context, pad], axis=1)
        guidance = make_guidance(window, list(range(history)))
        out = sample(model, clean, list(range(history, window)), guidance, steps)
        frames = mx.concatenate([frames, out[:, history:]], axis=1)
    return frames[:, :total]
