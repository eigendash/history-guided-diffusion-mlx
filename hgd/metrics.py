"""Scoring generated continuations against the true dynamics.

All frames here are numpy arrays in [0, 1], shape (n, T, H, W).
"""

import numpy as np

from .data import BAND, DETECT_FLOOR, detect

MISS_PENALTY = 8.0  # pixels charged when the truth shows a ball and the sample does not


def ball_mass(frames):
    img = np.clip(frames - DETECT_FLOOR, 0.0, None).copy()
    img[..., :, BAND[0] : BAND[1]] = 0.0
    return img.sum(axis=(-1, -2))


def centroid_error(gen, truth):
    """Per-frame distance between detected balls, only where the truth ball is visible.

    Returns (errors (n, T) with NaN where the truth is hidden)."""
    cg, vg = detect(gen)
    ct, vt = detect(truth)
    err = np.where(vg, np.linalg.norm(cg - ct, axis=-1), MISS_PENALTY)
    return np.where(vt, np.minimum(err, MISS_PENALTY), np.nan)


def visibility_agreement(gen, truth):
    return float((detect(gen)[1] == detect(truth)[1]).mean())


def motion(frames):
    """Mean absolute change between consecutive frames."""
    return float(np.abs(np.diff(frames, axis=1)).mean())


def chunk_metrics(gen, truth, first):
    """Score frames from index `first` onward; the earlier frames were history."""
    g, t = gen[:, first:], truth[:, first:]
    err = centroid_error(g, t)
    return {
        "mse": float(((g - t) ** 2).mean()),
        "centroid": float(np.nanmean(err)) if np.isfinite(err).any() else float("nan"),
        "visible": visibility_agreement(g, t),
        # compare motion including the last history frame, so the first step counts
        "motion": motion(gen[:, first - 1 :]) / max(motion(truth[:, first - 1 :]), 1e-9),
    }


def horizon_metrics(gen, truth, start, width):
    """Rollout scores in consecutive bins of `width` frames after `start`."""
    rows = []
    for lo in range(start, gen.shape[1], width):
        hi = min(lo + width, gen.shape[1])
        err = centroid_error(gen[:, lo:hi], truth[:, lo:hi])
        shown, true = ball_mass(gen[:, lo:hi]), ball_mass(truth[:, lo:hi])
        seen = detect(truth[:, lo:hi])[1]
        rows.append(
            {
                "frames": (lo, hi),
                "centroid": float(np.nanmean(err)) if np.isfinite(err).any() else float("nan"),
                "visible": visibility_agreement(gen[:, lo:hi], truth[:, lo:hi]),
                "mass": float(shown[seen].mean() / max(true[seen].mean(), 1e-9)) if seen.any() else float("nan"),
            }
        )
    return rows
