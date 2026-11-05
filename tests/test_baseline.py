import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src import barath
from src.baseline import nearest_team, smooth
from src.match import AWAY, HOME, Match, pm
from starter import baseline as organisers
from tests.fake_atd import PITCH, random_tracking, write_atd


@given(seed=st.integers(0, 2**32 - 1), n_frames=st.integers(1, 30), n_slots=st.integers(1, 8))
@settings(deadline=None, max_examples=80)
def test_nearest_team_agrees_with_the_organisers_code(seed, n_frames, n_slots):
    players, ball, team = random_tracking(seed, n_frames, n_slots, missing=0.5)
    with tempfile.TemporaryDirectory() as folder:
        meta, raw = write_atd(folder, players, ball, team)
        ours = nearest_team(barath.parse(meta, raw, n_frames, PITCH, ()))
        theirs = organisers.nearest_player_labels(
            raw, organisers.team_of_each_track(meta), n_frames, PITCH)
    np.testing.assert_array_equal(ours, theirs)


def test_the_strictly_nearer_player_wins_when_rounded_distances_tie():
    """Squared distances differ here while their square roots are equal, and the organisers'
    baseline compares the squares. Found by search; a comparison of norms picks the wrong slot."""
    ball = [99.92376897094668, 31.283069473018532]
    farther = [79.78884741096309, 33.82474329315809]
    nearer = [79.96604903921738, 27.600056146769223]
    m = Match(name="tie", fps=25, pitch=PITCH, half=np.array([1], dtype=np.int8),
              ball=np.array([ball + [np.nan]]), players=np.array([[farther, nearer]]),
              team=np.array([HOME, AWAY], dtype=np.int8))
    assert nearest_team(m).tolist() == [AWAY]


@pytest.mark.parametrize("window", [0, 1, 2, 25, 50, 125])
@given(seed=st.integers(0, 2**32 - 1), n_frames=st.integers(1, 300))
@settings(deadline=None, max_examples=25)
def test_smoothing_agrees_with_the_organisers_code(window, seed, n_frames):
    codes = np.random.default_rng(seed).integers(-1, 2, n_frames).astype(np.int8)
    np.testing.assert_array_equal(smooth(codes, window),
                                  pm.label_codes(organisers.smooth(codes, window)))
