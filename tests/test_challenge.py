import numpy as np
import pytest

from src import challenge
from src.match import A, D, H, HOME, Match


def match(name, n=400):
    labels = np.array(([H] * 100 + [A] * 100) * (n // 200), dtype=np.int8)
    return Match(name=name, fps=25, pitch=(105.0, 68.0), half=np.ones(n, dtype=np.int8),
                 ball=np.full((n, 3), np.nan), players=np.full((n, 1, 2), np.nan),
                 team=np.array([HOME], dtype=np.int8), labels=labels)


GAMES = {name: match(name) for name in ("a", "b", "c")}


def same(m, seed, level):
    return m


def perfect(train, threads):
    return lambda m: m.labels.copy()


def test_a_held_out_match_is_never_in_its_own_training_set():
    trained_on = []

    def fit(train, threads):
        trained_on.append(sorted(train))
        return lambda m: m.labels.copy()
    challenge.held_out(fit, GAMES, ("a", "b"), {"s": same}, threads=1)
    assert trained_on == [["b", "c"], ["a", "c"]]


def test_each_fold_is_tested_through_every_simulator():
    got = challenge.held_out(perfect, GAMES, ("a", "b"), {"one": same, "two": same}, threads=1)
    assert sorted(got) == ["one", "two"] and len(got["one"]) == 2


def test_the_confirmation_matches_are_refused_unless_unlocked():
    with pytest.raises(ValueError):
        challenge.check_folds(challenge.CONFIRM)
    challenge.check_folds(challenge.CONFIRM, unlocked=True)
    challenge.check_folds(challenge.DEVELOPMENT)


def test_development_and_confirmation_share_no_match_and_cover_all_nine():
    dev, confirm = set(challenge.DEVELOPMENT), set(challenge.CONFIRM)
    assert not dev & confirm and len(dev | confirm) == 9


def test_final_test_data_is_nowhere_in_the_folds():
    assert "M3" not in challenge.DEVELOPMENT + challenge.CONFIRM


def pair(flip):
    gt = np.array(([H] * 200 + [A] * 200 + [D] * 100) * 6, dtype=np.int8)
    pred = gt.copy()
    for start in range(0, len(gt), flip):
        pred[start:start + 20] = D
    return gt, pred


def test_a_clear_winner_passes_the_rule():
    frozen = {"s": [pair(100)], "real": [pair(100)]}
    better = {"s": [pair(100000)], "real": [pair(100000)]}
    report = challenge.compare(better, frozen, {"s": 25, "real": 25}, draws=200)
    assert report["passes"]


def test_winning_on_simulators_but_losing_on_real_tracking_fails_the_rule():
    frozen = {"s": [pair(400)], "real": [pair(100000)]}
    worse_real = {"s": [pair(100000)], "real": [pair(150)]}
    report = challenge.compare(worse_real, frozen, {"s": 25, "real": 25}, draws=200)
    assert not report["passes"]


def test_a_small_gain_on_the_simulators_fails_the_rule():
    frozen = {"s": [pair(100000)], "real": [pair(100000)]}
    report = challenge.compare(frozen, frozen, {"s": 25, "real": 25}, draws=100)
    assert not report["passes"]


def test_saved_results_come_back_unchanged(tmp_path):
    results = {"s": [pair(100), pair(300)], "real": [pair(200)]}
    challenge.save(results, tmp_path / "r.npz")
    back = challenge.load(tmp_path / "r.npz")
    assert sorted(back) == ["real", "s"] and len(back["s"]) == 2
    np.testing.assert_array_equal(back["s"][1][1], results["s"][1][1])


def test_metrica_stays_out_of_training_but_is_still_tested():
    trained_on = []

    def fit(train, threads):
        trained_on.append(sorted(train))
        return lambda m: m.labels.copy()
    games = {name: match(name) for name in ("J", "K", "M1", "M2")}
    out = challenge.held_out(challenge.without_metrica(fit), games, ("J", "M1"), {"s": same},
                             threads=1)
    assert trained_on == [["K"], ["J", "K"]]
    assert len(out["s"]) == 2
