import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.external.labels import challenge_labels, merge_flips
from src.match import A, D, H, X, pm

FPS = 25


def run(*pieces):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in pieces])


def test_a_touch_inside_a_possession_is_absorbed():
    got = merge_flips(run((H, 100), (A, 10), (H, 100)), FPS, 1)
    np.testing.assert_array_equal(got, run((H, 210)))


@pytest.mark.parametrize("frames", [25, 30])
def test_a_spell_of_the_limit_or_longer_stands(frames):
    labels = run((H, 100), (A, frames), (H, 100))
    np.testing.assert_array_equal(merge_flips(labels, FPS, 1), labels)


@pytest.mark.parametrize("labels", [
    run((H, 100), (D, 10), (H, 100)),            # a stoppage is not a flip
    run((H, 100), (A, 10), (D, 50)),             # possession changed, then the ball went dead
    run((D, 50), (A, 10), (H, 100)),             # a restart taken and lost at once
    run((H, 100), (A, 10), (X, 5), (H, 100)),    # ungraded frames between: no neighbour
], ids=["stoppage", "before-dead", "after-dead", "across-ungraded"])
def test_only_a_spell_between_two_of_the_other_team_is_a_flip(labels):
    np.testing.assert_array_equal(merge_flips(labels, FPS, 1), labels)


def test_the_shortest_flip_goes_first():
    # H 100, A 10, H 5, A 100: the 5-frame H sits inside A, leaving one handover.
    got = merge_flips(run((H, 100), (A, 10), (H, 5), (A, 100)), FPS, 1)
    np.testing.assert_array_equal(got, run((H, 100), (A, 115)))


def test_a_limit_of_zero_changes_nothing_and_the_input_is_left_alone():
    labels = run((H, 100), (A, 10), (H, 100))
    before = labels.copy()
    np.testing.assert_array_equal(merge_flips(labels, FPS, 0), before)
    merge_flips(labels, FPS, 1)
    np.testing.assert_array_equal(labels, before)


@given(seed=st.integers(0, 2**32 - 1), seconds=st.sampled_from([0.5, 1, 2]))
@settings(deadline=None, max_examples=60)
def test_dead_and_ungraded_frames_never_change_and_no_short_flip_survives(seed, seconds):
    rng = np.random.default_rng(seed)
    labels = run(*[(int(rng.integers(0, 4)), int(rng.integers(1, 60))) for _ in range(40)])
    got = merge_flips(labels, FPS, seconds)
    np.testing.assert_array_equal(got == D, labels == D)
    np.testing.assert_array_equal(got == X, labels == X)
    runs = pm.valid_runs(got)
    segs = pm.segments(got, runs)
    for one, two, three in zip(segs, segs[1:], segs[2:]):
        touching = one[1] == two[0] and two[1] == three[0]
        flip = two[2] in (H, A) and one[2] == three[2] and one[2] in (H, A)
        assert not (touching and flip and two[1] - two[0] < seconds * FPS)
    np.testing.assert_array_equal(merge_flips(got, FPS, seconds), got)


def test_the_chosen_rule_absorbs_a_touch_but_keeps_a_one_second_tackle():
    touch = run((H, 100), (A, 24), (H, 100))
    tackle = run((H, 100), (A, 25), (H, 100))
    np.testing.assert_array_equal(challenge_labels(touch, FPS), run((H, 224)))
    np.testing.assert_array_equal(challenge_labels(tackle, FPS), tackle)
