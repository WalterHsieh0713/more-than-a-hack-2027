import numpy as np
from dataclasses import replace

from src import features, model
from src.match import A, AWAY, D, H, HOME, Match, X
from src.decode import Costs


def play(n=3000, seed=0):
    """Invented play where the ball sits with a home player, then away, then is dead."""
    rng = np.random.default_rng(seed)
    labels = np.empty(n, dtype=np.int8)
    players = np.zeros((n, 12, 2)) + rng.normal(0, 6, (1, 12, 2))
    players[:, :6] += [30, 34]
    players[:, 6:] += [75, 34]
    ball = np.zeros((n, 3))
    ball[:, 2] = np.nan
    start, turn = 0, int(rng.integers(0, 3))
    while start < n:
        stop = min(n, start + int(rng.integers(80, 260)))
        state = turn % 3
        labels[start:stop] = state
        slot = {H: 2, A: 8, D: 2}[state]
        ball[start:stop, :2] = players[start, slot]
        players[start:stop] += rng.normal(0, 0.02, (stop - start, 12, 2)) * (state != D)
        start, turn = stop, turn + 1
    return Match(name="play{}".format(seed), fps=25, pitch=(105.0, 68.0),
                 half=np.ones(n, dtype=np.int8), ball=ball, players=players,
                 team=np.array([HOME] * 6 + [AWAY] * 6, dtype=np.int8), labels=labels)


COSTS = Costs(turnover=2000, stop=2000, restart_other=2000, restart_same=2000)
QUICK = dict(rounds=40)


def test_row_targets_take_the_label_at_the_middle_of_each_row():
    m = replace(play(20), labels=np.array([H] * 7 + [A] * 3 + [D] * 10, dtype=np.int8))
    index = np.arange(0, 20, 5)
    assert model.row_labels(m, index).tolist() == [H, A, D, D]


def test_a_boundary_target_marks_rows_near_a_change():
    labels = np.array([H] * 10 + [A] * 10, dtype=np.int8)
    assert model.near_change(labels, 2).tolist() == [0] * 8 + [1] * 4 + [0] * 8


def test_ungraded_rows_are_left_out_of_training():
    m = play(2000, 1)
    labels = m.labels.copy()
    labels[:500] = X
    x, y, b = model.table(replace(m, labels=labels))
    assert len(x) == len(y) == len(b) and len(y) < 400 and (y != X).all()


def test_each_training_copy_appears_as_itself_and_with_teams_swapped():
    m = play(1000, 2)
    x, y, _ = model.training_rows([m], [lambda c: c])
    assert len(y) == 2 * len(features.build(m)[1])
    assert (y == H).sum() == (y == A).sum()


def test_a_model_learns_to_tell_the_teams_apart_on_unseen_play():
    trained = model.fit([play(seed=s) for s in (1, 2, 3, 4)], [lambda c: c], **QUICK)
    test = play(seed=9)
    got = trained.predict(test, COSTS)
    graded = test.half > 0
    assert (got[graded] == test.labels[graded]).mean() > 0.9


def test_training_is_repeatable_to_the_bit():
    games = [play(seed=s) for s in (1, 2)]
    a = model.fit(games, [lambda c: c], rounds=10).proba(games[0])
    b = model.fit(games, [lambda c: c], rounds=10).proba(games[0])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_a_saved_model_predicts_identically_after_loading(tmp_path):
    games = [play(seed=s) for s in (1, 2)]
    trained = model.fit(games, [lambda c: c], rounds=10)
    trained.save(tmp_path / "m")
    back = model.Model.load(tmp_path / "m")
    np.testing.assert_array_equal(trained.proba(games[0])[0], back.proba(games[0])[0])


def test_predictions_cover_every_graded_frame_and_only_those():
    half = np.array([0] * 100 + [1] * 2800 + [0] * 100, dtype=np.int8)
    trained = model.fit([play(seed=1)], [lambda c: c], rounds=10)
    test = replace(play(seed=2), half=half)
    got = trained.predict(test, COSTS)
    assert (got[half == 0] == X).all() and (got[half > 0] != X).all()


def test_a_missing_ball_never_decides_on_its_own_when_it_is_hidden_in_training():
    """Metrica's corrected tracking blanks the ball while dead. Filling it from the ball alone,
    without labels, removes that giveaway."""
    n = 100
    ball = np.tile([50.0, 30.0, np.nan], (n, 1))
    ball[:, 0] += np.arange(n) * 0.1
    ball[40:60, :2] = np.nan
    m = replace(play(n), ball=ball, labels=np.full(n, H, dtype=np.int8))
    filled = model.fill_blanks(m)
    assert np.isfinite(filled.ball[:, :2]).all()
    assert filled.ball[50, 0] == 50.0 + 5.0


def test_both_halves_are_decoded_and_kept_apart():
    half = np.array([1] * 1500 + [0] * 100 + [2] * 1400, dtype=np.int8)
    trained = model.fit([play(seed=1)], [lambda c: c], rounds=10)
    got = trained.predict(replace(play(seed=2), half=half), COSTS)
    assert (got[half == 1] != X).all() and (got[half == 2] != X).all()
    assert (got[half == 0] == X).all()


def test_the_dead_ball_head_cannot_see_the_ball():
    games = [play(seed=s) for s in (1, 2)]
    trained = model.fit(games, [lambda c: c], rounds=10)
    test = play(seed=3)
    scrambled = test.ball.copy()
    scrambled[:, :2] = np.random.default_rng(0).random((len(test.ball), 2)) * 60
    a, _ = trained.proba(test)
    b, _ = trained.proba(replace(test, ball=scrambled))
    np.testing.assert_array_equal(a[:, D], b[:, D])
    assert not np.array_equal(a[:, H], b[:, H])


def test_the_three_chances_add_up_to_one():
    trained = model.fit([play(seed=1)], [lambda c: c], rounds=10)
    p, b = trained.proba(play(seed=2))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert ((b >= 0) & (b <= 1)).all()


def test_the_team_head_learns_only_from_live_play():
    x = np.zeros((6, 2))
    y = np.array([H, H, A, D, D, D], dtype=np.int8)
    keep, target = model.team_rows(x, y)
    assert keep.tolist() == [True, True, True, False, False, False]
    assert target.tolist() == [0, 0, 1]


def test_tag_evidence_changes_the_team_call_and_no_evidence_changes_nothing():
    trained = model.fit([play(seed=1)], [lambda c: c], rounds=10)
    test = play(seed=2)
    rows = len(features.build(test)[1])
    plain = trained.predict(test, COSTS)
    np.testing.assert_array_equal(trained.predict(test, COSTS, signs=np.zeros(rows)), plain)
    pushed = trained.predict(test, COSTS, signs=np.full(rows, -1.0), weight=8.0)
    assert (pushed == A).sum() > (plain == A).sum()
