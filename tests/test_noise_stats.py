import numpy as np
import pytest

from src.external.noise_stats import noise_stats, ranges
from src.match import Match

nan = np.nan


def match(ball_x, players, half=None, fps=10):
    """players: [T, P] of x positions, NaN where unseen. y is fixed, so x alone sets the speed."""
    players = np.array(players, dtype=float)
    T, P = players.shape
    xy = np.stack([players, np.where(np.isnan(players), nan, 1.0)], axis=2)
    ball = np.column_stack([ball_x, np.ones(T), np.full(T, nan)])
    return Match("toy", fps, (105.0, 68.0), np.ones(T, np.int8) if half is None else np.array(half, np.int8),
                 ball, xy, np.zeros(P, np.int8))


def test_ball_share_and_gaps_are_counted_by_hand():
    m = match([0, nan, nan, 3, nan, 5, 6, nan, nan, nan], np.zeros((10, 1)))
    s = noise_stats(m)
    assert s["ball_visible"] == pytest.approx(0.4)
    assert s["ball_gaps"] == 3
    assert s["ball_gap_median_s"] == pytest.approx(0.2)        # gaps of 1, 2 and 3 frames
    assert s["ball_gap_p90_s"] == pytest.approx(0.28)


def test_gaps_are_reported_in_seconds_whatever_the_frame_rate():
    ball = [0, nan, nan, nan, nan, 5]
    assert noise_stats(match(ball, np.zeros((6, 1)), fps=25))["ball_gap_median_s"] == pytest.approx(0.16)


def test_frames_outside_the_halves_do_not_count_and_a_gap_stops_at_the_half():
    ball = [nan, 1, nan, nan, nan, 5, nan]
    half = [0, 1, 1, 0, 2, 2, 0]
    s = noise_stats(match(ball, np.zeros((7, 1)), half=half))
    assert s["ball_visible"] == pytest.approx(0.5)              # 4 frames are in a half
    assert s["ball_gaps"] == 2                                  # one on each side of the break
    assert s["ball_gap_median_s"] == pytest.approx(0.1)


def test_objects_per_frame_count_players_and_ball():
    players = [[1, 2], [1, nan], [nan, nan], [nan, nan]]
    s = noise_stats(match([0, 0, 0, nan], players), full=2)
    assert s["objects_mean"] == pytest.approx((3 + 2 + 1 + 0) / 4)
    assert s["objects_mean_seen"] == pytest.approx((3 + 2 + 1) / 3)
    assert s["players_mean_seen"] == pytest.approx((2 + 1 + 0) / 3)
    assert s["empty_share"] == pytest.approx(0.25)


def test_a_frame_misses_a_player_when_fewer_than_full_are_visible():
    players = [[1, 2], [1, nan], [nan, nan], [1, 2]]
    assert noise_stats(match([0] * 4, players), full=2)["player_missing_share"] == pytest.approx(0.5)


def test_continuity_is_the_run_length_of_each_slot():
    players = [[1, 1], [1, nan], [1, 1], [nan, 1], [1, 1]]
    s = noise_stats(match([0] * 5, players), full=2)
    assert s["track_run_median_s"] == pytest.approx(0.2)        # runs of 3, 1 frames and 1, 3 frames
    assert s["track_run_p90_s"] == pytest.approx(0.3)


def test_a_run_stops_at_the_half():
    s = noise_stats(match([0] * 5, np.ones((5, 1)), half=[1, 1, 0, 2, 2]), full=1)
    assert s["track_run_median_s"] == pytest.approx(0.2)


def test_jumps_are_steps_no_player_could_run():
    players = [[0, 0], [1, 0], [3, 0], [3.5, 0]]                # 10, 20 and 5 m/s at 10 fps
    s = noise_stats(match([0] * 4, players), full=2)
    assert s["jump_share"] == pytest.approx(1 / 6)


def test_steps_are_not_taken_across_a_gap_or_the_half():
    players = [[0], [nan], [50], [50]]                          # the 50 m is across a gap
    assert noise_stats(match([0] * 4, players), full=1)["jump_share"] == 0
    assert noise_stats(match([0] * 3, [[0], [50], [50]], half=[1, 2, 2]), full=1)["jump_share"] == 0


def test_an_all_seen_match_has_no_gaps_and_nothing_breaks():
    s = noise_stats(match([1, 2], [[1], [1]]), full=1)
    assert s["ball_visible"] == 1 and s["ball_gaps"] == 0
    assert np.isnan(s["ball_gap_median_s"])


def test_ranges_are_low_and_high_per_key():
    assert ranges([{"a": 1.0, "b": 5.0}, {"a": 3.0, "b": 4.0}]) == {"a": (1.0, 3.0), "b": (4.0, 5.0)}
