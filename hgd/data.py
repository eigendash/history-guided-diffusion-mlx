"""A toy video domain where history matters.

One soft ball moves at constant velocity and bounces off the walls of a 16x16
canvas. A vertical band across the middle hides it completely. Two things follow:

  * velocity cannot be read from a single frame, so at least two history frames
    are needed to predict the motion;
  * while the ball is behind the band, a model that only sees the last couple of
    frames has no trace of where it was heading, so it cannot say where the ball
    comes out. Longer history carries that information.

Because the dynamics are deterministic, a continuation can be scored against the
true one whenever the history pins down the state.
"""

import numpy as np

SIZE = 16
BAND = (6, 10)  # columns hidden by the occluder
BAND_VALUE = 0.4
BLOB_SIGMA = 1.2
MARGIN = 1.5  # ball centre stays this far from the walls
DETECT_FLOOR = 0.15
VISIBLE_MASS = 0.5


class BallWorld:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def sample_states(self, n):
        pos = self.rng.uniform(MARGIN, SIZE - MARGIN, (n, 2))
        speed = self.rng.uniform(0.8, 1.6, n)
        angle = self.rng.uniform(0, 2 * np.pi, n)
        vel = np.stack([speed * np.cos(angle), speed * np.sin(angle)], axis=1)
        return pos, vel

    @staticmethod
    def trajectory(pos, vel, frames):
        """Ball centres (n, frames, 2) as (x, y), reflecting off the walls."""
        lo, hi = MARGIN, SIZE - MARGIN
        pos, vel = pos.copy(), vel.copy()
        out = np.zeros((len(pos), frames, 2))
        for t in range(frames):
            out[:, t] = pos
            pos = pos + vel
            low, high = pos < lo, pos > hi
            pos = np.where(low, 2 * lo - pos, pos)
            pos = np.where(high, 2 * hi - pos, pos)
            vel = np.where(low | high, -vel, vel)
        return out

    @staticmethod
    def render(centres):
        """Frames (n, T, SIZE, SIZE) in [0, 1] with the occluder painted on top."""
        ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        dx = xs - centres[..., 0, None, None]
        dy = ys - centres[..., 1, None, None]
        frames = np.exp(-(dx**2 + dy**2) / (2 * BLOB_SIGMA**2)).astype(np.float32)
        frames[..., :, BAND[0] : BAND[1]] = BAND_VALUE
        return frames

    def sample(self, n, frames):
        pos, vel = self.sample_states(n)
        centres = self.trajectory(pos, vel, frames)
        return self.render(centres), centres


def to_model(frames):
    """(n, T, H, W) in [0, 1] -> (n, T, H, W, 1) in [-1, 1]."""
    return frames[..., None] * 2.0 - 1.0


def from_model(x):
    return np.clip((np.asarray(x)[..., 0] + 1.0) / 2.0, 0.0, 1.0)


def detect(frames):
    """Ball centroid per frame, ignoring the occluder columns.

    Returns (centres (..., 2), visible (...,)). Where the ball is not visible the
    centre is meaningless.
    """
    img = np.clip(frames - DETECT_FLOOR, 0.0, None).copy()
    img[..., :, BAND[0] : BAND[1]] = 0.0
    mass = img.sum(axis=(-1, -2))
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    safe = np.maximum(mass, 1e-6)
    cx = (img * xs).sum(axis=(-1, -2)) / safe
    cy = (img * ys).sum(axis=(-1, -2)) / safe
    return np.stack([cx, cy], axis=-1), mass > VISIBLE_MASS
