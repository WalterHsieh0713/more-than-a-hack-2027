import numpy as np

from src import submit
from src.match import A, D, H, X


def test_frames_before_the_second_half_kickoff_are_always_dead():
    half = np.array([0, 1, 1, 0, 2, 2, 2, 2, 0], dtype=np.int8)
    codes = np.array([X, H, A, X, H, A, H, A, X], dtype=np.int8)
    got = submit.finish(codes, half, kickoff=6)
    assert got.tolist() == [X, H, A, X, D, D, H, A, X]


def test_the_input_is_left_alone():
    half = np.array([2, 2], dtype=np.int8)
    codes = np.array([H, A], dtype=np.int8)
    submit.finish(codes, half, kickoff=1)
    assert codes.tolist() == [H, A]


def test_the_file_is_named_after_the_team():
    assert submit.OUTPUT.endswith("827.csv")


def test_the_model_sees_the_second_half_from_its_kickoff_and_the_frames_before_are_dead():
    from dataclasses import replace
    from test_stack import play
    seen = {}

    class Finalist:
        def predict(self, m, costs, gain):
            seen["half"] = m.half.copy()
            return np.where(m.half > 0, H, X).astype(np.int8)
    m = play()
    first, kickoff = m.n_frames // 2, m.n_frames // 2 + 50
    half = np.repeat(np.array([1, 2], dtype=np.int8), m.n_frames // 2)
    half[:10] = 0
    got = submit.labels(replace(m, half=half), Finalist(), kickoff=kickoff)
    assert (seen["half"][first:kickoff] == 0).all() and (seen["half"][kickoff:] == 2).all()
    assert (seen["half"][:first] == half[:first]).all()
    assert (got[first:kickoff] == D).all() and (got[kickoff:] == H).all()
    assert (got[:10] == X).all() and (got[10:first] == H).all()


def test_the_submission_decodes_the_finalist_with_its_own_costs_and_no_tags():
    from dataclasses import replace
    from src import stack
    from test_stack import play
    seen = {}

    class Finalist:
        def predict(self, m, costs, gain, signs=None, weight=0.0):
            seen.update(costs=costs, gain=gain, signs=signs, weight=weight)
            return np.full(m.n_frames, H, dtype=np.int8)
    m = play()
    submit.labels(replace(m, half=np.repeat(np.array([1, 2], dtype=np.int8), m.n_frames // 2)),
                  Finalist())
    assert seen == dict(costs=stack.COSTS, gain=stack.GAIN, signs=None, weight=0.0)
    assert submit.MODEL.endswith("v2")
