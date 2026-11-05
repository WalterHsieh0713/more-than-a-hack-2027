"""src/noise_b.py on small synthetic matches. Every number here is invented."""

import copy

import numpy as np

from src import noise_b
from src.match import AWAY, HOME, Match

PITCH = (105.0, 68.0)


def make_match(T=600, n_home=4, n_away=4, ball_z=0.0, freeze=None):
    """Persons walk along distinct lanes so a position identifies its person.
    freeze=(a, b) holds the ball still on frames a..b-1, as the provider does while it is dead."""
    P = n_home + n_away
    t = np.arange(T, dtype=np.float32)
    players = np.empty((T, P, 2), np.float32)
    for k in range(P):                               # lane k: y = 8k + 4, drifting 1 cm a frame
        players[:, k, 0] = 10 + 8 * k + 0.01 * t
        players[:, k, 1] = 8 * k + 4
    ball = np.column_stack([20 + (0.05 * t) % 60, 10 + (0.02 * t) % 40, np.full(T, ball_z, np.float32)])
    if freeze:
        a, b = freeze
        ball[a:b, :2] = ball[a, :2]
    team = np.array([HOME] * n_home + [AWAY] * n_away, np.int8)
    half = np.ones(T, np.int8)
    return Match(name="toy", fps=25, pitch=PITCH, half=half, ball=ball.astype(np.float32),
                 players=players, team=team, labels=np.zeros(T, np.int8))


def person_of(m, out):
    """[T, 46] index of the input person whose position each slot shows, -1 where NaN or unmatched.
    Only meaningful at level=0 where positions are copied exactly."""
    who = np.full(out.players.shape[:2], -1)
    for s in range(46):
        xy = out.players[:, s]
        seen = np.isfinite(xy[:, 0])
        d = np.abs(m.players[seen] - xy[seen, None]).sum(-1)      # [n, P]
        hit = d.min(1) == 0
        who[np.flatnonzero(seen)[hit], s] = d.argmin(1)[hit]
    return who


def test_labels_passed_through_and_never_read():
    m = make_match()
    a = noise_b.degrade(m, seed=3)
    assert a.labels is m.labels or np.array_equal(a.labels, m.labels)
    m2 = copy.deepcopy(m)
    m2.labels = np.random.default_rng(0).permutation(m2.labels.size).astype(np.int8) % 4
    b = noise_b.degrade(m2, seed=3)
    assert a.players.tobytes() == b.players.tobytes()
    assert a.ball.tobytes() == b.ball.tobytes()
    m2.labels = None
    assert noise_b.degrade(m2, seed=3).labels is None


def test_deterministic_and_seed_sensitive():
    m = make_match()
    a, b, c = (noise_b.degrade(m, seed=s) for s in (7, 7, 8))
    assert a.players.tobytes() == b.players.tobytes() and a.ball.tobytes() == b.ball.tobytes()
    assert not np.array_equal(a.players, c.players, equal_nan=True)


def test_output_contract_and_no_mutation():
    m = make_match()
    before = copy.deepcopy(m)
    out = noise_b.degrade(m, seed=1)
    assert out.players.shape == (m.n_frames, 46, 2) and out.players.dtype == np.float32
    assert out.team.dtype == np.int8 and out.team.tolist() == [HOME] * 23 + [AWAY] * 23
    assert out.ball.shape == (m.n_frames, 3) and np.isnan(out.ball[:, 2]).all()
    assert np.isfinite(out.ball[:, :2]).any()
    assert (out.half is m.half or np.array_equal(out.half, m.half)) and out.fps == m.fps
    assert out.pitch == m.pitch
    assert np.array_equal(before.players, m.players) and np.array_equal(before.ball, m.ball)
    assert out.players is not m.players and out.ball is not m.ball


def test_level_zero_is_an_anonymised_copy():
    m = make_match(T=6000)
    out = noise_b.degrade(m, seed=5, level=0.0)
    who = person_of(m, out)
    # Every visible slot shows a real person, and the pool it sits in is that person's team.
    for s in range(46):
        ps = who[:, s][who[:, s] >= 0]
        assert (m.team[ps] == (HOME if s < 23 else AWAY)).all()
    # Every person is shown on every frame exactly once.
    for t in range(0, 6000, 97):
        assert sorted(who[t][who[t] >= 0].tolist()) == list(range(8))
    # No person keeps one slot, so no slot keeps following one person.
    for p in range(8):
        assert len(set(np.nonzero(who == p)[1].tolist())) >= 2
    assert np.isnan(out.ball[:, 2]).all()


def test_level_scales_noise():
    m = make_match(T=3000)
    lo, mid, hi = (noise_b.degrade(m, seed=2, level=lv) for lv in (0.0, 1.0, 3.0))
    assert np.allclose(np.sort(lo.ball[:, :2], 0), np.sort(m.ball[:, :2], 0))  # no jitter at 0
    assert np.isnan(lo.players[:, :, 0]).sum(1).tolist() == [46 - 8] * 3000
    dev = lambda o: np.nanmean(np.abs(o.ball[:, :2] - m.ball[:, :2]))
    missing = lambda o: np.isnan(o.ball[:, 0]).mean()
    assert dev(mid) > 0.05 and missing(mid) > 0.02
    assert dev(hi) > dev(mid) and missing(hi) > missing(mid)
    assert np.isnan(hi.players[:, :, 0]).sum() > np.isnan(mid.players[:, :, 0]).sum()


def test_frozen_ball_does_not_survive():
    m = make_match(T=1000, freeze=(300, 700))
    assert (np.diff(m.ball[300:700, :2], axis=0) == 0).all()
    out = noise_b.degrade(m, seed=4)
    still = (np.diff(out.ball[300:700, :2], axis=0) == 0).all(1)
    assert still.mean() < 0.02
    # At level 0 the frozen ball is carried by its nearest person instead of hanging in the air.
    out0 = noise_b.degrade(m, seed=4, level=0.0)
    d = np.linalg.norm(m.players[300:700] - m.ball[300:700, None, :2], axis=-1)
    carrier = d.argmin(1)
    assert np.allclose(out0.ball[300:700, :2], m.players[np.arange(300, 700), carrier])
    assert np.allclose(out0.ball[:300, :2], m.ball[:300, :2])


def test_ground_projection_of_an_aerial_ball():
    # Camera at (52.5, -30) and 20 m up; a ball 10 m up at mid-pitch is seen twice as far away.
    cam = np.array([52.5, -30.0, 20.0])
    xy = noise_b.project(np.array([[52.5, 34.0]]), np.array([10.0]), cam)
    assert np.allclose(xy, [[52.5, 98.0]])
    assert np.allclose(noise_b.project(np.array([[10.0, 5.0]]), np.array([0.0]), cam), [[10, 5]])
    # Through degrade: a ball always 5 m up lands further from the camera, never nearer.
    m = make_match(T=2000, ball_z=5.0)
    out = noise_b.degrade(m, seed=9)
    cam_y = noise_b.camera(np.random.default_rng(9), *m.pitch)[1]
    away = np.sign(out.ball[:, 1] - m.ball[:, 1]) == np.sign(m.ball[:, 1] - cam_y)
    assert np.nanmean(away[np.isfinite(out.ball[:, 1])]) > 0.9


def test_more_persons_than_slots_is_not_an_error():
    m = make_match(T=200, n_home=30, n_away=2)
    out = noise_b.degrade(m, seed=1, level=0.0)
    assert np.isfinite(out.players[:, :23, 0]).sum(1).max() <= 23


def test_ball_missing_more_when_high_and_crowded():
    low = make_match(T=4000, ball_z=0.0)
    high = make_match(T=4000, ball_z=4.0)
    miss = lambda mm: np.isnan(noise_b.degrade(mm, seed=11).ball[:, 0]).mean()
    assert miss(high) > miss(low)
    crowd = make_match(T=4000)
    crowd.players[:, :, 0] = crowd.ball[:, None, 0] + 0.5     # everyone stands on the ball
    crowd.players[:, :, 1] = crowd.ball[:, None, 1] + 0.5
    assert miss(crowd) > miss(low)


def test_slots_swap_and_ghosts_appear_with_noise():
    m = make_match(T=6000)
    out = noise_b.degrade(m, seed=6)
    seen = np.isfinite(out.players[:, :, 0]).sum(1)
    assert seen.max() > 8, "no duplicated person was ever tracked"
    assert seen.min() < 8, "nobody was ever lost"
