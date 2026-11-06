import numpy as np
import pytest
from dataclasses import replace

from src import features
from src.features import NAMES, attack_sign, build, expand, flip_pitch, half_spans, swap_teams
from src.match import A, AWAY, D, H, HOME, Match, X

L, W = 105.0, 68.0
FPS = 25


def scene(n=500, seed=0, half=None):
    """Two tidy teams on their own sides, ball moving among them, some of it unseen."""
    rng = np.random.default_rng(seed)
    home = np.array([30.0, 34.0]) + rng.normal(0, 5, (6, 2))
    away = np.array([75.0, 34.0]) + rng.normal(0, 5, (6, 2))
    base = np.concatenate([home, away])
    wander = np.cumsum(rng.normal(0, 0.03, (n, 12, 2)), axis=0)
    players = (base[None] + wander).astype(np.float64)
    players[rng.random((n, 12)) < 0.1] = np.nan
    ball = np.column_stack([52 + 30 * np.sin(np.arange(n) / 40), 34 + 10 * np.cos(np.arange(n) / 30),
                            np.full(n, np.nan)])
    ball[100:160, :2] = np.nan
    return Match(name="scene", fps=FPS, pitch=(L, W),
                 half=np.ones(n, dtype=np.int8) if half is None else half, ball=ball,
                 players=players, team=np.array([HOME] * 6 + [AWAY] * 6, dtype=np.int8),
                 labels=np.full(n, H, dtype=np.int8))


def col(x, name):
    return x[:, NAMES.index(name)]


def test_one_row_per_fifth_of_a_second_on_graded_frames_only():
    half = np.array([0] * 7 + [1] * 23 + [0] * 3 + [2] * 10 + [0] * 2, dtype=np.int8)
    m = scene(45, half=half)
    assert half_spans(m) == [(1, 7, 30), (2, 33, 43)]
    X_, index = build(m)
    assert index.tolist() == [7, 12, 17, 22, 27, 33, 38]
    assert X_.shape == (7, len(NAMES))


def test_expanding_rows_back_to_frames_marks_everything_else_ungraded():
    half = np.array([0] * 7 + [1] * 12 + [0] * 2, dtype=np.int8)
    m = scene(21, half=half)
    got = expand(np.array([H, A, D], dtype=np.int8), build(m)[1], m)
    want = [X] * 7 + [H] * 5 + [A] * 5 + [D] * 2 + [X] * 2
    assert got.tolist() == want


def test_at_ten_frames_a_second_a_row_is_two_frames():
    m = replace(scene(20), fps=10)
    assert build(m)[1].tolist() == list(range(0, 20, 2))


def test_attack_direction_follows_where_each_team_stands():
    m = scene(100)
    assert attack_sign(m, 0, 100) == 1
    assert attack_sign(flip_pitch(m), 0, 100) == -1


def test_attack_direction_is_worked_out_separately_for_each_half():
    half = np.array([1] * 50 + [2] * 50, dtype=np.int8)
    m = scene(100, half=half)
    players = m.players.copy()
    players[50:, :, 0] = L - players[50:, :, 0]
    m = replace(m, players=players)
    assert attack_sign(m, 0, 50) == 1 and attack_sign(m, 50, 100) == -1


def test_no_feature_depends_on_which_slot_a_player_sits_in():
    m = scene()
    order = np.array([3, 0, 5, 1, 2, 4, 9, 11, 6, 10, 7, 8])
    shuffled = replace(m, players=m.players[:, order])
    np.testing.assert_allclose(build(shuffled)[0], build(m)[0], atol=1e-5)


def test_no_feature_reads_the_labels():
    m = scene()
    other = replace(m, labels=np.random.default_rng(1).permutation(m.labels))
    np.testing.assert_array_equal(build(other)[0], build(m)[0])


def test_features_do_not_care_which_way_the_teams_attack():
    m = scene()
    np.testing.assert_allclose(build(flip_pitch(m))[0], build(m)[0], atol=1e-4)


@pytest.mark.parametrize("home,away", [("dist_home", "dist_away"), ("speed_home", "speed_away"),
                                       ("depth_home", "depth_away")])
def test_swapping_the_teams_swaps_their_columns(home, away):
    m = scene()
    a, b = build(m)[0], build(swap_teams(m))[0]
    np.testing.assert_allclose(col(b, home), col(a, away), atol=1e-5)


def test_swapping_the_teams_reverses_who_is_nearest():
    m = scene()
    a, b = build(m)[0], build(swap_teams(m))[0]
    np.testing.assert_allclose(col(b, "near_past_1s"), -col(a, "near_past_1s"), atol=1e-5)


def test_swapping_the_teams_swaps_their_labels_too():
    m = replace(scene(4), labels=np.array([H, A, D, X], dtype=np.int8))
    assert swap_teams(m).labels.tolist() == [A, H, D, X]


def test_the_ball_near_a_home_player_counts_for_home():
    m = scene(200)
    players = m.players.copy()
    players[:] = np.nan
    players[:, 0] = [10.0, 10.0]
    players[:, 6] = [90.0, 60.0]
    ball = np.tile([10.5, 10.0, np.nan], (200, 1))
    m = replace(m, players=players, ball=ball)
    x = build(m)[0]
    assert (col(x, "control")[2:-2] == 1).all()
    assert col(x, "dist_home")[5] == pytest.approx(0.5, abs=1e-4)


def test_a_past_window_looks_only_back_and_a_future_window_only_ahead():
    assert features.past(np.array([1.0, 2, 3, 4]), 2)[3] == pytest.approx(2.5)
    assert features.future(np.array([1.0, 2, 3, 4]), 2)[0] == pytest.approx(2.5)
    assert np.isnan(features.past(np.array([1.0, 2]), 2)[0])


def test_the_last_control_carries_through_a_gap_in_both_directions():
    sign = np.array([0, 1, 0, 0, -1, 0], dtype=np.float32)
    last, age = features.carry(sign)
    assert last.tolist()[1:] == [1, 1, 1, -1, -1] and age.tolist()[1:] == [0, 1, 2, 0, 1]
    assert np.isnan(last[0])


def test_gap_columns_report_how_long_the_ball_has_been_unseen():
    m = scene()
    x = build(m)[0]
    row = 130 // 5
    # Row covers frames 130-134, the ball was last seen on 99 and returns on 160.
    assert col(x, "gap_age")[row] == pytest.approx(33 / FPS, abs=1e-5)
    assert col(x, "gap_ahead")[row] == pytest.approx(28 / FPS, abs=1e-5)


def test_a_half_with_no_ball_at_all_still_gives_rows():
    m = scene()
    ball = m.ball.copy()
    ball[:, :2] = np.nan
    x = build(replace(m, ball=ball))[0]
    assert x.shape[0] == 100 and np.isfinite(col(x, "speed_all")).all()


def alone(player_path, ball_path, n=100):
    """One home and one far-away away player, with the ball on a given path."""
    m = scene(n)
    players = np.full_like(m.players, np.nan)
    players[:, 0] = player_path
    players[:, 6] = [100.0, 60.0]
    return replace(m, players=players, ball=np.column_stack([ball_path, np.full(n, np.nan)]))


def test_a_ball_far_from_everyone_is_nearer_one_team_without_being_controlled():
    m = alone(np.tile([10.0, 10.0], (100, 1)), np.tile([20.0, 10.0], (100, 1)))
    x = build(m)[0]
    assert (col(x, "near_past_1s")[6:] == 1).all()
    assert (col(x, "control") == 0).all()


def test_a_fast_ball_is_not_controlled_even_right_beside_a_player():
    ball = np.column_stack([np.arange(100) * 0.8 + 5, np.full(100, 30.0)])
    m = alone(ball + [0.5, 0.0], ball)
    x = build(m)[0]
    assert col(x, "ball_speed")[10] == pytest.approx(20.0, abs=0.1)
    assert (col(x, "control")[2:-2] == 0).all()


def test_a_tracker_jump_does_not_count_as_running():
    path = np.tile([10.0, 10.0], (100, 1))
    path[50:] += [5.0, 0.0]
    m = alone(path, np.tile([50.0, 30.0], (100, 1)))
    assert (col(build(m)[0], "speed_home") == 0).all()


def pair_at(first, second, n=200):
    """One home and one away player on given paths, with the ball out of the way."""
    m = scene(n)
    players = np.full_like(m.players, np.nan)
    players[:, 0], players[:, 6] = first, second
    return replace(m, players=players, ball=np.tile([52.0, 34.0, np.nan], (n, 1)))


def still(x, y, n=200):
    return np.tile([x, y], (n, 1))


def test_steady_running_counts_as_speed_over_a_whole_second():
    run_path = np.column_stack([np.arange(200) * 0.2 + 5, np.full(200, 20.0)])    # 5 m/s
    x = build(pair_at(run_path, still(90.0, 40.0)))[0]
    assert col(x, "pace")[20] == pytest.approx(2.5, abs=0.05)        # one runs, one stands
    assert col(x, "movers")[20] == 1


def test_jitter_on_the_spot_does_not_look_like_running():
    rng = np.random.default_rng(0)
    shaky = still(30.0, 30.0) + rng.normal(0, 0.3, (200, 2))
    x = build(pair_at(shaky, still(90.0, 40.0)))[0]
    assert col(x, "pace")[20] < col(x, "speed_all")[20] / 3
    assert col(x, "movers")[20] == 0


@pytest.mark.parametrize("spot,want", [((30.0, 0.5), 1), ((0.5, 30.0), 1), ((30.0, 30.0), 0),
                                       ((30.0, -2.0), 1)])
def test_players_on_or_past_the_lines_are_counted(spot, want):
    x = build(pair_at(still(*spot), still(90.0, 40.0)))[0]
    assert col(x, "on_lines")[20] == want


def test_a_kickoff_shape_has_everyone_in_their_own_half():
    x = build(pair_at(still(40.0, 34.0), still(60.0, 34.0)))[0]
    assert col(x, "own_half")[20] == 1
    x = build(pair_at(still(70.0, 34.0), still(60.0, 34.0)))[0]
    assert col(x, "own_half")[20] == 0.5


def test_players_in_the_box_are_counted_at_the_goal_each_team_attacks():
    x = build(pair_at(still(100.0, 34.0), still(103.0, 34.0)))[0]
    assert col(x, "box_attacked_by_home")[20] == 2 and col(x, "box_attacked_by_away")[20] == 0


def test_the_gap_between_the_teams_is_measured_in_metres():
    x = build(pair_at(still(40.0, 34.0), still(60.0, 34.0)))[0]
    assert col(x, "team_gap")[20] == pytest.approx(20.0)


def test_long_windows_reach_twenty_seconds_either_way():
    assert "pace_past_20s" in NAMES and "pace_next_20s" in NAMES and "movers_next_10s" in NAMES


def test_player_only_columns_never_use_the_ball():
    m = scene()
    ball = m.ball.copy()
    ball[:, :2] = np.random.default_rng(3).random((len(ball), 2)) * 60
    a, b = build(m)[0], build(replace(m, ball=ball))[0]
    keep = [NAMES.index(n) for n in features.PLAYER_ONLY]
    np.testing.assert_array_equal(a[:, keep], b[:, keep])
    assert "pace" in features.PLAYER_ONLY and "seen" not in features.PLAYER_ONLY


def test_features_are_rounded_so_a_tiny_float_difference_cannot_reach_the_model():
    m = scene()
    nudged = replace(m, players=m.players * (1 + 1e-13))
    np.testing.assert_array_equal(build(nudged)[0], build(m)[0])
