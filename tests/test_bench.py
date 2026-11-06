import json

import numpy as np
import pytest

from src import bench
from src.match import A, D, H, HOME, Match, X, pm
from src.scoring import pooled


def run(*pieces):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in pieces])


def match(name, labels):
    n = len(labels)
    return Match(name=name, fps=25, pitch=(105.0, 68.0), half=np.ones(n, dtype=np.int8),
                 ball=np.full((n, 3), np.nan), players=np.full((n, 1, 2), np.nan),
                 team=np.array([HOME], dtype=np.int8), labels=labels)


LONG = run((H, 300), (A, 200), (D, 100), (H, 300))
THREE = {name: match(name, LONG) for name in "abc"}


def truth(m):
    return m.labels


def test_the_dusseldorf_games_never_sit_on_both_sides_of_a_split():
    dusseldorf = {"J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"}
    for test, train in bench.splits(bench.GROUPED):
        assert not (set(test) & dusseldorf and set(train) & dusseldorf)
    assert len(bench.GROUPED) == 3


def test_every_match_is_tested_once_and_never_trains_on_itself():
    seen = []

    def fit(train):
        names = [m.name for m in train]

        def predict(m):
            assert m.name not in names
            seen.append(m.name)
            return m.labels
        return predict
    bench.evaluate(fit, THREE, [["a"], ["b", "c"]])
    assert sorted(seen) == ["a", "b", "c"]


def test_perfect_predictions_score_one():
    assert bench.evaluate(lambda train: truth, THREE, [["a"], ["b"], ["c"]]).score[
        "spell_f1_mean"] == 1.0


def test_scoring_uses_the_challenge_labels_rather_than_the_raw_flips():
    flippy = run((H, 100), (A, 10), (H, 100), (D, 50), (A, 300))
    merged = run((H, 210), (D, 50), (A, 300))
    matches = {"a": match("a", flippy), "b": match("b", flippy)}
    groups = [["a"], ["b"]]
    assert bench.evaluate(lambda t: lambda m: merged, matches, groups).score[
        "spell_f1_mean"] == 1.0
    assert bench.evaluate(lambda t: lambda m: flippy, matches, groups).score[
        "spell_f1_mean"] < 1.0


def test_both_sides_of_a_split_see_degraded_tracking_and_true_labels_are_kept():
    calls = []

    def degrade(m, seed, level):
        calls.append((m.name, seed, level))
        return Match(name=m.name + "-noisy", fps=m.fps, pitch=m.pitch, half=m.half, ball=m.ball,
                     players=m.players, team=m.team, labels=m.labels)

    def fit(train):
        assert all(m.name.endswith("-noisy") for m in train)

        def predict(m):
            assert m.name.endswith("-noisy")
            return m.labels
        return predict
    result = bench.evaluate(fit, THREE, [["a"], ["b"], ["c"]], degrade, seed=3, level=0.5)
    assert result.score["spell_f1_mean"] == 1.0
    assert len({seed for _, seed, _ in calls}) == 3 and {c[2] for c in calls} == {0.5}


def test_the_same_seed_degrades_a_match_the_same_way_in_every_fold():
    seeds = {}

    def degrade(m, seed, level):
        seeds.setdefault(m.name, set()).add(seed)
        return m
    bench.evaluate(lambda t: truth, THREE, [["a"], ["b"], ["c"]], degrade)
    assert all(len(s) == 1 for s in seeds.values())


def noisy_pair(seed):
    rng = np.random.default_rng(seed)
    gt = run(*[(int(rng.integers(0, 3)), int(rng.integers(20, 400))) for _ in range(120)],
             (X, 5))
    pred = gt.copy()
    for start in rng.integers(0, len(gt) - 60, 40):
        pred[start:start + 30] = rng.integers(0, 3)
    pred[gt == X] = X
    return gt, pred


def test_block_counts_reproduce_the_official_pooled_score_exactly():
    pairs = [noisy_pair(1), noisy_pair(2), noisy_pair(3)]
    assert bench.spell_f1(pairs) == pytest.approx(pooled(pairs)["spell_f1_mean"], abs=1e-12)


def test_labels_on_ungraded_frames_are_ignored_as_the_official_scorer_does():
    gt, pred = noisy_pair(10)
    busy = np.where(gt == X, H, pred).astype(np.int8)
    assert bench.spell_f1([(gt, busy)]) == pytest.approx(pooled([(gt, busy)])["spell_f1_mean"],
                                                         abs=1e-12)


def test_an_abstention_inside_a_possession_is_filled_as_the_official_scorer_does():
    gt = run((H, 300), (A, 300), (X, 5))
    pred = gt.copy()
    pred[100:200] = X
    assert bench.spell_f1([(gt, pred)]) == 1.0


def test_the_interval_is_repeatable_and_brackets_the_estimate():
    pairs = [noisy_pair(4), noisy_pair(5)]
    first = bench.interval(pairs, fps=25, draws=300)
    assert first == bench.interval(pairs, fps=25, draws=300)
    assert first[0] < bench.spell_f1(pairs) < first[1]


def test_comparing_a_system_with_itself_gives_exactly_zero():
    pairs = [noisy_pair(6)]
    assert bench.difference(pairs, pairs, fps=25, draws=200) == (0.0, 0.0)


def test_a_clearly_better_system_has_an_interval_above_zero():
    pairs = [noisy_pair(7), noisy_pair(8)]
    perfect = [(gt, gt) for gt, _ in pairs]
    low, high = bench.difference(perfect, pairs, fps=25, draws=300)
    assert 0 < low < high


def test_pairing_narrows_the_interval_compared_with_scoring_each_alone():
    a = [noisy_pair(9)]
    gt, pred = a[0]
    almost = pred.copy()
    almost[10:40] = gt[10:40]
    low, high = bench.difference([(gt, almost)], a, fps=25, draws=400)
    alone_low, alone_high = bench.interval(a, fps=25, draws=400)
    assert high - low < alone_high - alone_low


def test_the_ledger_appends_and_stops_at_the_config_limit(tmp_path):
    path = tmp_path / "ledger.jsonl"
    for i in range(bench.MAX_CONFIGS):
        bench.record(path, "stage", "config{}".format(i), {"spell_f1": 0.5})
    with pytest.raises(RuntimeError):
        bench.record(path, "stage", "one-too-many", {"spell_f1": 0.5})
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(entries) == bench.MAX_CONFIGS and entries[0]["config"] == "config0"


def test_rerunning_a_known_config_is_allowed_once_the_stage_is_full(tmp_path):
    path = tmp_path / "ledger.jsonl"
    for i in range(bench.MAX_CONFIGS):
        bench.record(path, "one", "config{}".format(i), {"spell_f1": 0.5})
    bench.record(path, "one", "config0", {"spell_f1": 0.6})
    assert len(path.read_text().splitlines()) == bench.MAX_CONFIGS + 1


def test_a_full_stage_does_not_use_up_the_next_stages_tries(tmp_path):
    path = tmp_path / "ledger.jsonl"
    for i in range(bench.MAX_CONFIGS):
        bench.record(path, "one", "config{}".format(i), {"spell_f1": 0.5})
    bench.record(path, "two", "first", {"spell_f1": 0.5})
    assert len(path.read_text().splitlines()) == bench.MAX_CONFIGS + 1
