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


def test_the_submission_decodes_the_finalist_with_its_own_costs_and_tags(monkeypatch):
    from dataclasses import replace
    from src import stack, tags
    from test_stack import play
    seen = {}

    class Finalist:
        def predict(self, m, costs, gain, signs, weight):
            seen.update(costs=costs, gain=gain, weight=weight)
            return np.full(m.n_frames, H, dtype=np.int8)
    monkeypatch.setattr(submit.load_smart_tagging, "load", lambda: [])
    m = play()
    submit.labels(replace(m, half=np.repeat(np.array([1, 2], dtype=np.int8), m.n_frames // 2)),
                  Finalist())
    assert seen == dict(costs=stack.COSTS, gain=stack.GAIN, weight=tags.WEIGHT)
    assert submit.MODEL.endswith("v2")
