"""A small non-causal video transformer that takes one noise level per frame.

All frames, history or not, enter as one token sequence. Each token is modulated by
the noise level of its own frame, so the network sees which frames are clean, which
are partly noised and which are pure noise, and attends across all of them in both
directions in time.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    size: int = 16
    frames: int = 8
    patch: int = 2
    channels: int = 1
    dim: int = 128
    heads: int = 4
    depth: int = 6


def level_embedding(k, dim):
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half) / half)
    angles = (k * 1000.0)[..., None] * freqs
    return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(dim // heads)
        self.k_norm = nn.RMSNorm(dim // heads)
        self.out = nn.Linear(dim, dim)

    def __call__(self, x):
        B, N, D = x.shape
        q, k, v = (t.reshape(B, N, self.heads, -1).transpose(0, 2, 1, 3) for t in mx.split(self.qkv(x), 3, axis=-1))
        o = mx.fast.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v, scale=(D // self.heads) ** -0.5)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, affine=False)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.mod = nn.Linear(dim, 6 * dim)
        self.mod.weight = mx.zeros_like(self.mod.weight)  # adaLN-zero: start as identity
        self.mod.bias = mx.zeros_like(self.mod.bias)

    def __call__(self, h, cond):
        """h: (B, T, N, D) tokens grouped by frame; cond: (B, T, D)."""
        B, T, N, D = h.shape
        s1, b1, g1, s2, b2, g2 = (m[:, :, None] for m in mx.split(self.mod(nn.silu(cond)), 6, axis=-1))
        flat = lambda t: t.reshape(B, T * N, D)  # noqa: E731
        unflat = lambda t: t.reshape(B, T, N, D)  # noqa: E731
        h = h + g1 * unflat(self.attn(flat(self.norm1(h) * (1 + s1) + b1)))
        return h + g2 * self.mlp(self.norm2(h) * (1 + s2) + b2)


class VideoDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        p, c = cfg.patch, cfg.channels
        self.tokens = (cfg.size // p) ** 2
        self.embed = nn.Linear(p * p * c, cfg.dim)
        self.space_pos = mx.random.normal((self.tokens, cfg.dim)) * 0.02
        self.time_pos = mx.random.normal((cfg.frames, cfg.dim)) * 0.02
        self.level_mlp = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.blocks = [Block(cfg.dim, cfg.heads) for _ in range(cfg.depth)]
        self.final_norm = nn.LayerNorm(cfg.dim, affine=False)
        self.final_mod = nn.Linear(cfg.dim, 2 * cfg.dim)
        self.final_mod.weight = mx.zeros_like(self.final_mod.weight)
        self.final_mod.bias = mx.zeros_like(self.final_mod.bias)
        self.head = nn.Linear(cfg.dim, p * p * c)

    def patchify(self, x):
        B, T, H, W, C = x.shape
        p = self.cfg.patch
        x = x.reshape(B, T, H // p, p, W // p, p, C).transpose(0, 1, 2, 4, 3, 5, 6)
        return x.reshape(B, T, (H // p) * (W // p), p * p * C)

    def unpatchify(self, t):
        B, T, _, _ = t.shape
        p, C, g = self.cfg.patch, self.cfg.channels, self.cfg.size // self.cfg.patch
        t = t.reshape(B, T, g, g, p, p, C).transpose(0, 1, 2, 4, 3, 5, 6)
        return t.reshape(B, T, self.cfg.size, self.cfg.size, C)

    def __call__(self, x, levels):
        """x: (B, T, H, W, C); levels: (B, T). Returns the predicted v, same shape as x."""
        T = x.shape[1]
        h = self.embed(self.patchify(x)) + self.space_pos[None, None] + self.time_pos[None, :T, None]
        cond = self.level_mlp(level_embedding(levels, self.cfg.dim))
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = (m[:, :, None] for m in mx.split(self.final_mod(nn.silu(cond)), 2, axis=-1))
        return self.unpatchify(self.head(self.final_norm(h) * (1 + scale) + shift))
