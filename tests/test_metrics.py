import numpy as np

from hgd.data import BallWorld, detect
from hgd.metrics import MISS_PENALTY, ball_mass, centroid_error, chunk_metrics, horizon_metrics, motion


def clip(n=16, frames=8, seed=0):
    return BallWorld(seed).sample(n, frames)[0]


def test_identical_video_scores_perfectly():
    truth = clip()
    m = chunk_metrics(truth, truth, 4)
    assert m["mse"] == 0 and m["visible"] == 1.0 and abs(m["motion"] - 1.0) < 1e-9
    assert m["centroid"] == 0.0 or np.isnan(m["centroid"])


def test_static_copy_of_the_last_history_frame_is_flagged_by_the_motion_ratio():
    truth = clip()
    static = truth.copy()
    static[:, 4:] = truth[:, 3:4]
    m = chunk_metrics(static, truth, 4)
    assert m["motion"] < 0.35
    assert m["mse"] > 0


def test_a_missing_ball_costs_the_miss_penalty_only_where_the_truth_is_visible():
    truth = clip(32, 8, seed=1)
    blank = np.zeros_like(truth)
    err = centroid_error(blank, truth)
    visible = detect(truth)[1]
    assert np.isnan(err[~visible]).all()
    assert (err[visible] == MISS_PENALTY).all()


def test_shifted_ball_has_a_centroid_error_of_the_shift():
    centres = np.array([[[3.0, 8.0]]])
    a = BallWorld.render(centres)
    b = BallWorld.render(centres + np.array([0.0, 2.0]))
    assert abs(float(centroid_error(b, a)[0, 0]) - 2.0) < 0.35


def test_motion_is_zero_for_a_still_video():
    still = np.repeat(clip()[:, :1], 6, axis=1)
    assert motion(still) == 0.0


def test_horizon_bins_cover_the_rollout_and_report_mass_ratio():
    truth = clip(8, 20)
    rows = horizon_metrics(truth, truth, 4, 8)
    assert [r["frames"] for r in rows] == [(4, 12), (12, 20)]
    assert all(abs(r["mass"] - 1.0) < 1e-6 for r in rows if not np.isnan(r["mass"]))
    faded = horizon_metrics(truth * 0.5, truth, 4, 8)
    assert all(r["mass"] < 0.8 for r in faded if not np.isnan(r["mass"]))
    assert ball_mass(truth).shape == (8, 20)
