import numpy as np
import pytest

from src import tags
from src.match import A, D, H, X


def event(code, team, start, end):
    return {"code": code, "start_frame": start, "end_frame": end, "tags": {"Team": team}}


INDEX = np.arange(0, 100, 5)


def signs(*events):
    return tags.row_signs(list(events), INDEX, 5, "Barcelona", "Atletic Club").tolist()


def test_an_attacking_phase_points_to_the_team_it_names():
    got = signs(event("BUILD UP", "Barcelona", 10, 20))
    assert got[2:4] == [1, 1] and got[1] == 0 and got[4] == 0


def test_a_defending_phase_points_to_the_other_team():
    got = signs(event("DEFENDING IN DEFENSIVE THIRD", "Barcelona", 10, 20))
    assert got[2:4] == [-1, -1]


def test_a_defensive_transition_also_points_to_the_other_team():
    assert signs(event("DEFENSIVE TRANSITION", "Atletic Club", 10, 15))[2] == 1


@pytest.mark.parametrize("code", ["Build Up", "build up", " BUILD UP "])
def test_codes_match_whatever_their_case(code):
    assert signs(event(code, "Barcelona", 10, 15))[2] == 1


def test_a_tag_with_no_team_says_nothing():
    assert set(signs(event("SET PIECES", "N/A", 10, 50))) == {0}


def test_an_unknown_team_name_is_an_error_rather_than_silence():
    with pytest.raises(ValueError):
        signs(event("BUILD UP", "Real Madrid", 10, 15))


def test_a_code_with_no_meaning_for_possession_says_nothing():
    assert set(signs(event("GOALS", "Barcelona", 10, 50))) == {0}


def test_tags_that_disagree_cancel_out():
    got = signs(event("BUILD UP", "Barcelona", 10, 20), event("PROGRESSION", "Atletic Club", 10, 20))
    assert got[2] == 0


def test_tags_that_agree_count_once_each_row():
    got = signs(event("BUILD UP", "Barcelona", 10, 20), event("PROGRESSION", "Barcelona", 10, 20))
    assert got[2] == 1


def test_evidence_shifts_the_team_odds_and_leaves_dead_ball_alone():
    p = np.array([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2], [0.4, 0.4, 0.2]])
    got = tags.apply(p, np.array([1, -1, 0]), 1.0)
    assert got[0, H] > got[0, A] and got[1, A] > got[1, H]
    np.testing.assert_allclose(got[2], p[2])
    np.testing.assert_allclose(got[:, D], p[:, D])
    np.testing.assert_allclose(got.sum(axis=1), 1.0)
    assert got[0, H] / got[0, A] == pytest.approx(np.e)


def test_a_weight_of_zero_changes_nothing():
    p = np.array([[0.5, 0.3, 0.2]])
    np.testing.assert_allclose(tags.apply(p, np.array([1]), 0.0), p)


def test_simulated_tags_name_the_right_team_as_often_as_asked():
    truth = np.array([H] * 30000 + [A] * 30000 + [D] * 2000, dtype=np.int8)
    rng = np.random.default_rng(0)
    sign = tags.simulated(truth, reliability=0.8, coverage=1.0, rng=rng)
    tagged = (truth != D) & (sign != 0)
    want = np.where(truth == H, 1, -1)
    assert (sign[tagged] == want[tagged]).mean() == pytest.approx(0.8, abs=0.03)


def test_simulated_tags_cover_about_the_share_asked_for():
    truth = np.array([H] * 20000, dtype=np.int8)
    sign = tags.simulated(truth, 1.0, 0.5, np.random.default_rng(1))
    assert (sign != 0).mean() == pytest.approx(0.5, abs=0.1)


def test_simulated_tags_never_use_ungraded_rows():
    truth = np.array([X] * 500 + [H] * 500, dtype=np.int8)
    sign = tags.simulated(truth, 1.0, 1.0, np.random.default_rng(2))
    assert (sign[:400] == 0).all()
