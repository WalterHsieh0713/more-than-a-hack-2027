import numpy as np
import pytest

from src.decode import Costs, decode, viterbi
from src.match import A, D, H

FREE = Costs(turnover=0, stop=0, restart_other=0, restart_same=0)
STICKY = Costs(turnover=10, stop=10, restart_other=10, restart_same=10)


def emit(*states):
    """Scores with 1 for the named state and 0 elsewhere, one row per step."""
    rows = np.zeros((len(states), 3), dtype=np.int64)
    rows[np.arange(len(states)), states] = 1
    return rows


def test_with_no_switching_cost_every_step_follows_its_evidence():
    assert viterbi(emit(H, A, D, H), FREE).tolist() == [H, A, D, H]


def test_a_blip_too_short_to_pay_for_two_switches_is_ignored():
    assert viterbi(emit(H, H, A, H, H), STICKY).tolist() == [H] * 5


def test_a_long_enough_run_pays_for_its_switches():
    # Split: 55 evidence minus 20 for two switches = 35. All H: 30. All A: 25.
    steps = (H,) * 15 + (A,) * 25 + (H,) * 15
    assert viterbi(emit(*steps), STICKY).tolist() == list(steps)


def test_the_cheapest_switch_point_is_where_the_evidence_changes():
    steps = (H,) * 30 + (A,) * 30
    assert viterbi(emit(*steps), STICKY).tolist() == list(steps)


def test_after_a_stoppage_the_decoder_remembers_who_had_the_ball():
    """Equal evidence for both teams after the stop: the cheaper restart decides."""
    rows = np.zeros((5, 3), dtype=np.int64)
    rows[0, H], rows[1, D], rows[2, D] = 100, 100, 100
    rows[3:, H] = rows[3:, A] = 50
    other = Costs(turnover=1000, stop=0, restart_other=1, restart_same=9)
    same = Costs(turnover=1000, stop=0, restart_other=9, restart_same=1)
    assert viterbi(rows, other).tolist()[-1] == A
    assert viterbi(rows, same).tolist()[-1] == H


def test_a_likely_boundary_makes_a_switch_cheaper_there():
    steps = (H,) * 3 + (A,) * 3 + (H,) * 3
    no_help = np.zeros(9, dtype=np.int64)
    assert viterbi(emit(*steps), STICKY, no_help).tolist() == [H] * 9
    discount = np.zeros(9, dtype=np.int64)
    discount[[3, 6]] = 9
    assert viterbi(emit(*steps), STICKY, discount).tolist() == list(steps)


def test_a_tie_is_settled_the_same_way_every_time():
    rows = np.zeros((6, 3), dtype=np.int64)
    first = viterbi(rows, STICKY).tolist()
    assert first == viterbi(rows, STICKY).tolist() and len(set(first)) == 1


def test_decode_quantises_so_a_tiny_float_change_cannot_move_a_label():
    rng = np.random.default_rng(0)
    p = rng.dirichlet((1, 1, 1), 400)
    boundary = rng.random(400) * 0.9 + 0.05
    costs = Costs(turnover=2000, stop=2000, restart_other=1500, restart_same=2500)
    base = decode(p, boundary, costs)
    assert set(base.tolist()) <= {H, A, D}
    noisy = np.clip(p * (1 + rng.normal(0, 1e-7, p.shape)), 1e-9, 1)
    np.testing.assert_array_equal(decode(noisy, boundary, costs), base)


def test_decode_follows_confident_probabilities():
    p = np.array([[0.98, 0.01, 0.01]] * 30 + [[0.01, 0.98, 0.01]] * 30)
    got = decode(p, np.full(60, 0.5), Costs(1000, 1000, 1000, 1000))
    assert got.tolist() == [H] * 30 + [A] * 30


@pytest.mark.parametrize("field", ["turnover", "stop", "restart_other", "restart_same"])
def test_costs_must_be_whole_numbers_for_exact_arithmetic(field):
    values = dict(turnover=1, stop=1, restart_other=1, restart_same=1)
    values[field] = 1.5
    with pytest.raises(TypeError):
        Costs(**values)


def test_probabilities_are_rounded_so_a_near_tie_cannot_go_either_way():
    """Without rounding the 1e-8 nudge would hand this row to A on one machine and H on another."""
    free = Costs(0, 0, 0, 0)
    nudged = np.array([[0.5 - 1e-8, 0.5 + 1e-8, 1e-6]])
    assert decode(nudged, np.ones(1), free).tolist() == [H]


def test_the_boundary_head_can_be_switched_off_with_gain_zero():
    p = np.array([[0.9, 0.05, 0.05]] * 3 + [[0.05, 0.9, 0.05]] * 3)
    unlikely = np.full(6, 1e-5)
    free = Costs(0, 0, 0, 0)
    assert decode(p, unlikely, free, gain=0).tolist() == [H] * 3 + [A] * 3
    assert decode(p, unlikely, free, gain=1).tolist() == [H] * 6
