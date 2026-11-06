import numpy as np

from src import tune
from src.decode import Costs
from src.match import A, H, X


def test_rows_are_spread_back_over_their_frames_with_ungraded_frames_left_alone():
    truth = np.array([X, X] + [H] * 10 + [X] * 3, dtype=np.int8)
    got = tune.frames(np.array([H, A], dtype=np.int8), np.array([2, 7]), 5, truth)
    assert got.tolist() == [X, X] + [H] * 5 + [A] * 5 + [X] * 3


def saved(chance, half, truth, step=5):
    n = len(chance)
    return dict(chance=np.array(chance, dtype=np.float32), boundary=np.ones(n, dtype=np.float32),
                index=np.arange(n) * step, half=np.array(half, dtype=np.int8), truth=truth)


def test_each_half_is_decoded_on_its_own():
    """A switch cost too high to pay inside one half must not stop the second half differing."""
    chance = [[0.9, 0.05, 0.05]] * 4 + [[0.05, 0.9, 0.05]] * 4
    truth = np.array([H] * 20 + [A] * 20, dtype=np.int8)
    got = tune.predicted(saved(chance, [1] * 4 + [2] * 4, truth), Costs(10 ** 6, 10 ** 6, 10 ** 6,
                                                                        10 ** 6), 0)
    assert got.tolist() == truth.tolist()
