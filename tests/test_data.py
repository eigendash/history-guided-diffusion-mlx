import numpy as np

from hgd.data import BAND, BAND_VALUE, MARGIN, SIZE, BallWorld, detect, from_model, to_model


def test_trajectory_is_deterministic_and_stays_inside_the_walls():
    world = BallWorld(0)
    pos, vel = world.sample_states(200)
    a = world.trajectory(pos, vel, 60)
    b = world.trajectory(pos, vel, 60)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= MARGIN - 1e-6 and a.max() <= SIZE - MARGIN + 1e-6


def test_motion_is_constant_velocity_between_bounces():
    pos, vel = np.array([[8.0, 8.0]]), np.array([[1.0, 0.5]])
    path = BallWorld.trajectory(pos, vel, 5)[0]
    np.testing.assert_allclose(np.diff(path, axis=0), [[1.0, 0.5]] * 4, atol=1e-9)


def test_ball_reflects_at_a_wall():
    path = BallWorld.trajectory(np.array([[SIZE - MARGIN - 0.5, 8.0]]), np.array([[1.0, 0.0]]), 4)[0]
    assert path[1, 0] > path[2, 0]  # turned around after crossing the wall
    assert (path[:, 0] <= SIZE - MARGIN + 1e-9).all()


def test_occluder_hides_the_ball_completely():
    centre = np.array([[[8.0, 8.0]]])
    frame = BallWorld.render(centre)[0, 0]
    assert np.allclose(frame[:, BAND[0] : BAND[1]], BAND_VALUE)
    _, visible = detect(frame)
    assert not visible


def test_detector_recovers_a_visible_ball():
    rng = np.random.default_rng(0)
    xs = rng.uniform(2, BAND[0] - 2.5, 50)
    ys = rng.uniform(2, SIZE - 2, 50)
    centres = np.stack([xs, ys], axis=1)[:, None]
    found, visible = detect(BallWorld.render(centres))
    assert visible.all()
    np.testing.assert_allclose(found[:, 0], centres[:, 0], atol=0.35)


def test_render_range_and_model_round_trip():
    frames, _ = BallWorld(1).sample(4, 6)
    assert frames.shape == (4, 6, SIZE, SIZE)
    assert frames.min() >= 0 and frames.max() <= 1
    np.testing.assert_allclose(from_model(to_model(frames)), frames, atol=1e-6)


def test_one_frame_cannot_reveal_velocity_but_two_can():
    pos = np.array([[4.0, 8.0], [4.0, 8.0]])
    vel = np.array([[1.0, 0.0], [-1.0, 0.0]])
    frames = BallWorld.render(BallWorld.trajectory(pos, vel, 3))
    np.testing.assert_allclose(frames[0, 0], frames[1, 0])  # identical first frames
    assert not np.allclose(frames[0, 1], frames[1, 1])  # differ one step later
