"""The degrade() contract, on small synthetic matches whose right answers can be read off by hand."""

import copy

import numpy as np
import pytest

from src import noise
from src.match import AWAY, HOME, Match
from src.noise import degrade

FPS = 25
ROW_M = 5.0
ROWS = (5, 10, 15, 20, 30, 35, 40, 45)   # home on the first four rows, away on the last four


def clean(T=7500, z=0.0, ys=ROWS):
    """Players parked on fixed rows far from a ball that orbits midfield and never stops moving."""
    t = np.arange(T)
    ball = np.stack([52 + 40 * np.sin(t / 50), 34 + 20 * np.cos(t / 50), np.full(T, z)], axis=1)
    team = np.array([HOME if y < 25 else AWAY for y in ys], dtype=np.int8)
    players = np.zeros((T, len(ys), 2), np.float32)
    players[:, :, 0] = np.where(team == HOME, 2, 103)
    players[:, :, 1] = ys
    return Match("synthetic", FPS, (105.0, 68.0), half=np.ones(T, np.int8),
                 ball=ball.astype(np.float32), players=players, team=team,
                 labels=(t % 3).astype(np.int8))


def frozen(*stretches, T=7500):
    """clean(), with the ball held at the start of each stretch and teleporting away at its end."""
    m = clean(T)
    for start, stop in stretches:
        m.ball[start:stop] = m.ball[start]
    return m


def rows(m):
    """Which row each object sits on, [T, slots]; NaN where the slot is empty."""
    return np.rint(m.players[:, :, 1] / ROW_M)


@pytest.fixture
def quiet(monkeypatch):
    """Every player mechanism off, so one can be switched on and checked exactly."""
    for name, value in dict(PLAYER_JITTER_M=0, PLAYER_GAP_MEAN_S=0, SWAP_TRIES_PER_MIN=0,
                            DUPLICATES_PER_MIN=0, TRACKLET_MEAN_S=1e9).items():
        monkeypatch.setattr(noise, name, value)
    return monkeypatch


# -- the contract --------------------------------------------------------------------------------

def test_labels_are_never_read_and_pass_through_unchanged():
    m = clean()
    permuted = copy.copy(m)
    permuted.labels = np.random.default_rng(0).permutation(m.labels)
    unlabelled = copy.copy(m)
    unlabelled.labels = None

    a, b, c = degrade(m, 7), degrade(permuted, 7), degrade(unlabelled, 7)

    assert a.players.tobytes() == b.players.tobytes() == c.players.tobytes()
    assert a.ball.tobytes() == b.ball.tobytes() == c.ball.tobytes()
    np.testing.assert_array_equal(a.labels, m.labels)
    np.testing.assert_array_equal(b.labels, permuted.labels)
    assert c.labels is None


def test_same_seed_same_arrays_and_another_seed_differs():
    a, b, c = degrade(clean(), 7), degrade(clean(), 7), degrade(clean(), 8)
    assert a.players.tobytes() == b.players.tobytes() and a.ball.tobytes() == b.ball.tobytes()
    assert a.players.tobytes() != c.players.tobytes() and a.ball.tobytes() != c.ball.tobytes()


@pytest.mark.parametrize("level", [0, 1])
def test_output_shape_and_passthrough(level):
    m = clean()
    out = degrade(m, 7, level)
    assert out.players.dtype == np.float32 and out.players.shape == (m.n_frames, 46, 2)
    assert out.team.dtype == np.int8
    np.testing.assert_array_equal(out.team, [HOME] * 23 + [AWAY] * 23)
    assert np.issubdtype(out.ball.dtype, np.floating) and out.ball.shape == (m.n_frames, 3)
    assert np.isnan(out.ball[:, 2]).all()
    np.testing.assert_array_equal(out.half, m.half)
    assert out.fps == m.fps and out.pitch == m.pitch


def test_input_is_not_mutated():
    m = frozen((1000, 1200))
    before = [a.copy() for a in (m.players, m.ball, m.team, m.half, m.labels)]
    degrade(m, 7)
    for was, now in zip(before, (m.players, m.ball, m.team, m.half, m.labels)):
        assert was.tobytes() == now.tobytes()


def test_level_zero_keeps_every_player_and_every_moving_ball_exactly():
    m = clean()
    out = degrade(m, 7, level=0)
    np.testing.assert_array_equal(out.ball[:, :2], m.ball[:, :2])
    np.testing.assert_array_equal(np.sort(out.players[:, :, 1], axis=1)[:, :8],
                                  np.sort(m.players[:, :, 1], axis=1))
    assert np.isfinite(out.players[:, :, 1]).sum(1).max() == 8


def test_higher_levels_are_harsher():
    m = clean(T=15000)
    runs = [degrade(m, 7, level) for level in (0, 1, 3)]
    lost = [np.isnan(r.ball[:, 0]).mean() for r in runs]
    seen = [np.isfinite(r.players[:, :, 1]).sum(1).mean() for r in runs]
    error = [np.nanstd(r.players[:, :, 1] - rows(r) * ROW_M) for r in runs]
    assert lost[0] == 0 < lost[1] < lost[2]
    assert seen[0] == 8 > seen[1] > seen[2]
    assert error[0] == 0 < error[1] < error[2]


def test_separator_frames_stay_empty():
    m = clean()
    m.half[100] = 0
    m.ball[100] = np.nan
    m.players[100] = np.nan
    out = degrade(m, 7)
    assert np.isnan(out.ball[100]).all() and np.isnan(out.players[100]).all()


# -- ball loss -----------------------------------------------------------------------------------

def lost_share(out, frames):
    return np.isnan(out.ball[frames, 0]).mean()


def test_ball_loss_rises_with_hidden_height():
    m = clean(T=30000)
    m.ball[15000:, 2] = 8.0
    out = degrade(m, 7)
    assert lost_share(out, slice(15000, None)) > 3 * lost_share(out, slice(0, 15000))


def test_ball_loss_rises_with_crowding():
    m = clean(T=30000, ys=(5, 6, 7, 8, 30, 31, 32, 33))
    m.players[15000:] = m.ball[15000:, None, :2]   # everybody on top of the ball, half way through
    out = degrade(m, 7)
    assert lost_share(out, slice(15000, None)) > 3 * lost_share(out, slice(0, 15000))


def test_ball_loss_comes_in_bursts():
    lost = np.isnan(degrade(clean(T=30000), 7).ball[:, 0])
    start, stop = noise._runs(lost)
    assert (stop - start).mean() > 10
    assert lost[1:][lost[:-1]].mean() > 0.8           # plain random dropout would give lost.mean()


def test_ball_loss_works_when_the_source_has_no_height():
    assert 0 < np.isnan(degrade(clean(z=np.nan), 7).ball[:, 0]).mean() < 1


# -- the frozen ball -----------------------------------------------------------------------------

def test_frozen_means_a_sustained_exactly_still_ball_seen_from_positions_alone():
    xy = np.arange(80.0)[:, None].repeat(2, axis=1)
    xy[10:13] = xy[10]                                # three frames: a pause, not a freeze
    xy[40:52] = xy[40]                                # twelve frames: frozen
    xy[60:75] = np.nan                                # missing is not still
    want = np.zeros(80, bool)
    want[40:52] = True
    np.testing.assert_array_equal(noise._frozen(xy, 12), want)


def test_frozen_detection_covers_a_run_that_reaches_the_last_frame():
    xy = np.arange(30.0)[:, None].repeat(2, axis=1)
    xy[20:] = xy[20]
    assert noise._frozen(xy, 5).nonzero()[0].tolist() == list(range(20, 30))


@pytest.mark.parametrize("level", [0, 1, 3])
def test_a_frozen_ball_is_no_longer_still(level):
    out = degrade(frozen((3000, 3300)), 7, level)
    xy = out.ball[3000:3300, :2]
    assert not (xy[1:] == xy[:-1]).all(1).any()
    moved = np.linalg.norm(xy[25:] - xy[:-25], axis=1)
    assert np.nanmedian(moved) > 0.5                  # smoothing over a second must not reveal it


def test_a_short_freeze_that_ends_where_it_began_still_moves():
    m = clean()
    m.ball[2999:3011] = m.ball[2998] + 1              # twelve frames: an even number on purpose
    m.ball[3011:] = m.ball[2998] + 0.1 * np.arange(4489)[:, None]
    xy = degrade(m, 7, level=0).ball[2998:3012, :2]
    assert not (xy[1:] == xy[:-1]).all(1).any()


def test_the_restart_is_not_a_teleport():
    m = frozen((3000, 3150))
    assert np.linalg.norm(m.ball[3150, :2] - m.ball[3149, :2]) > 30    # the shortcut being removed
    xy = degrade(m, 7, level=0).ball[:, :2]
    assert np.linalg.norm(np.diff(xy, axis=0), axis=1).max() < 2


def test_a_freeze_beside_a_separator_or_at_the_end_keeps_its_ball():
    m = frozen((101, 200), (7300, 7500))
    m.half[100] = 0
    m.ball[100] = np.nan
    out = degrade(m, 7, level=0)
    assert np.isfinite(out.ball[np.r_[101:200, 7300:7500], :2]).all()


def test_hiding_a_frozen_ball_does_not_become_the_new_shortcut():
    stretches = [(s, s + 150) for s in range(500, 7000, 500)]
    out = degrade(frozen(*stretches), 7)
    inside = np.concatenate([np.arange(a, b) for a, b in stretches])
    assert 0.05 < lost_share(out, inside) < 0.95     # neither always shown nor always hidden
    first = [np.isnan(out.ball[a, 0]) for a, _ in stretches]
    last = [np.isnan(out.ball[b - 1, 0]) for _, b in stretches]
    assert not all(first) and not all(last)          # gaps are not cut to the stoppage


# -- players -------------------------------------------------------------------------------------

@pytest.mark.parametrize("level", [0, 1])
def test_a_slot_does_not_follow_a_person(level):
    r = rows(degrade(clean(T=30000), 7, level))
    slots_per_person = [len(np.unique(np.nonzero(r == y / ROW_M)[1])) for y in ROWS]
    people_per_slot = [len(np.unique(r[np.isfinite(r[:, s]), s])) for s in range(46)]
    assert min(slots_per_person) > 3
    assert max(people_per_slot) > 1


@pytest.mark.parametrize("level", [0, 1])
def test_tracks_only_move_within_their_team(level):
    r = rows(degrade(clean(), 7, level))
    assert set(np.unique(r[:, :23][np.isfinite(r[:, :23])])) <= {1, 2, 3, 4}
    assert set(np.unique(r[:, 23:][np.isfinite(r[:, 23:])])) <= {6, 7, 8, 9}


def test_tracklets_have_random_lengths():
    out = degrade(clean(T=30000), 7)
    lengths = np.concatenate([np.subtract(*noise._runs(np.isfinite(out.players[:, s, 0]))[::-1])
                              for s in range(46)])
    assert lengths.std() > 0.5 * lengths.mean()


def test_people_go_missing_in_stretches():
    r = rows(degrade(clean(T=30000), 7))
    seen = (r == 1).any(1)
    assert 0.6 < seen.mean() < 0.99
    start, stop = noise._runs(~seen)
    assert (stop - start).mean() > 5


def test_close_teammates_swap_tracks(quiet):
    quiet.setattr(noise, "SWAP_TRIES_PER_MIN", 6)
    m = clean(ys=(5, 6, 15, 20, 30, 35, 40, 45))
    y = degrade(m, 7).players[:, :, 1]
    switched = [s for s in range(46) if set(np.unique(y[np.isfinite(y[:, s]), s])) == {5, 6}]
    assert len(switched) == 2                        # one slot per person, trading places
    np.testing.assert_array_equal(np.sort(y, axis=1)[:, :8], np.sort(m.players[:, :, 1], axis=1))


def test_distant_teammates_do_not_swap(quiet):
    quiet.setattr(noise, "SWAP_TRIES_PER_MIN", 60)
    y = degrade(clean(), 7).players[:, :, 1]
    assert all(len(np.unique(y[np.isfinite(y[:, s]), s])) <= 1 for s in range(46))


def test_duplicates_add_a_nearby_object_of_the_same_team(quiet):
    quiet.setattr(noise, "DUPLICATES_PER_MIN", 60)
    quiet.setattr(noise, "DUPLICATE_OFFSET_M", 0.1)
    out = degrade(clean(), 7)
    count = np.isfinite(out.players[:, :, 0]).sum(1)
    assert count.min() == 8 and count.max() > 8
    r = rows(out)
    assert set(np.unique(r[:, :23][np.isfinite(r[:, :23])])) <= {1, 2, 3, 4}
    assert set(np.unique(r[:, 23:][np.isfinite(r[:, 23:])])) <= {6, 7, 8, 9}


def test_jitter_has_the_set_size_and_is_smooth_in_time(quiet):
    quiet.setattr(noise, "PLAYER_JITTER_M", 0.5)
    m = clean(T=30000)
    err = degrade(m, 7).players[:, :, 1] - m.players[0, :, 1].mean()   # strip nothing but a constant
    err = err[np.isfinite(err)].reshape(len(m.half), 8)
    err = err - err.mean(0)
    assert 0.35 < err.std() < 0.65
    assert (err[1:] * err[:-1]).mean() / err.var() > 0.5


def test_ball_jitter_has_the_set_size(monkeypatch):
    monkeypatch.setattr(noise, "BALL_LOSS_BASE_PER_S", 0)
    monkeypatch.setattr(noise, "BALL_JITTER_M", 0.5)
    m = clean(T=30000)
    err = degrade(m, 7).ball[:, :2] - m.ball[:, :2]
    assert 0.35 < err.std() < 0.65


def test_a_team_with_more_tracks_than_slots_drops_the_excess():
    out = degrade(clean(T=200, ys=tuple(range(1, 25)) + (40,)), 7, level=0)
    assert np.isfinite(out.players[:, :23, 0]).sum(1).max() == 23
    assert np.isfinite(out.players[:, 23:, 0]).sum(1).max() == 1
