import numpy as np

from src import features, model, stack
from src.decode import Costs
from src.match import A, AWAY, D, H, HOME, Match, X

COSTS = Costs(turnover=2000, stop=2000, restart_other=1600, restart_same=2400)


def wave(n=400, seed=0):
    """Invented chances, in dead, away, change, then question order, with one row per step."""
    rng = np.random.default_rng(seed)
    return np.round(rng.random((n, 3 + len(stack.QUESTIONS))), 4).astype(np.float32)


def test_inner_groups_hold_every_match_exactly_once():
    names = ["M1", "b", "a", "c", "d", "e", "f", "g", "h"]
    held = stack.inner_groups(names)
    assert len(held) == stack.GROUPS and sorted(sum(held, [])) == sorted(names)
    assert all(held)


def test_a_match_is_never_scored_by_a_first_stage_that_saw_it():
    trained_on = {}

    def first(names):
        names = tuple(sorted(names))
        return lambda name, table: trained_on.setdefault(name, names) and table
    tables = {name: ["table of " + name] for name in "abcdefgh"}
    chances = stack.out_of_fold(tables, first)
    assert sorted(chances) == sorted(tables)
    for name, names in trained_on.items():
        assert name not in names
    assert chances["c"] == ["table of c"]


def test_each_column_is_a_function_of_its_own_half_only():
    chances, half = wave(600), np.repeat([1, 2], 300)
    changed = chances.copy()
    changed[300:] = wave(300, seed=9)
    a, b = stack.columns(chances, half), stack.columns(changed, half)
    np.testing.assert_array_equal(a[:300], b[:300])
    assert not np.array_equal(a[300:], b[300:])


def test_columns_are_finite_and_rounded_so_machines_cannot_disagree():
    x = stack.columns(wave(), np.ones(400, dtype=np.int8))
    assert x.shape == (400, len(stack.COLUMNS)) and np.isfinite(x).all()
    np.testing.assert_array_equal(x, np.round(x, stack.DECIMALS))


def test_future_columns_see_what_is_coming_and_past_columns_do_not():
    chances = np.zeros((200, 3 + len(stack.QUESTIONS)), dtype=np.float32)
    chances[100:, 1] = 1
    x = stack.columns(chances, np.ones(200, dtype=np.int8))
    ahead, behind = stack.COLUMNS.index("a_next_5"), stack.COLUMNS.index("a_past_5")
    assert x[99, ahead] == 1 and x[99, behind] == 0
    assert x[105, behind] == 1 and x[94, ahead] == 0


def test_each_kind_of_change_is_asked_about_separately():
    labels = np.array([H] * 10 + [A] * 10 + [D] * 10 + [H] * 10, dtype=np.int8)
    half = np.ones(40, dtype=np.int8)
    turn, stop, start = (stack.question(name, labels, half)[1] for name in ("turn", "stop", "start"))
    assert turn[9] == 1 and turn[19] == 0 and turn[29] == 0
    assert stop[19] == 1 and stop[9] == 0 and stop[29] == 0
    assert start[29] == 1 and start[9] == 0 and start[19] == 0


def test_a_shifted_question_asks_who_has_the_ball_a_second_from_now():
    labels = np.array([H] * 10 + [A] * 10, dtype=np.int8)
    half = np.ones(20, dtype=np.int8)
    keep, away = stack.question("away_soon", labels, half)
    assert keep[:15].all() and not keep[15:].any() and away.tolist() == [0] * 5 + [1] * 10
    keep, away = stack.question("away_was", labels, half)
    assert keep[5:].all() and not keep[:5].any() and away.tolist() == [0] * 10 + [1] * 5


def test_a_shifted_question_skips_dead_ball_and_the_other_half():
    labels = np.array([H] * 10 + [D] * 10 + [A] * 10, dtype=np.int8)
    keep, _ = stack.question("away_soon", labels, np.ones(30, dtype=np.int8))
    assert not keep[5:15].any() and keep[:5].all()
    keep, _ = stack.question("away_soon", np.full(10, H, np.int8), np.repeat([1, 2], 5))
    assert not keep[:5].any() and keep[5:].tolist() == [False] * 5


def test_ungraded_rows_are_never_asked_about():
    labels = np.array([X] * 10 + [H] * 10, dtype=np.int8)
    half = np.ones(20, dtype=np.int8)
    for name in stack.QUESTIONS:
        assert not stack.question(name, labels, half)[0][:10].any()


def test_the_dead_head_reads_no_team_series():
    names = [stack.COLUMNS[i] for i in stack.DEAD_COLUMNS]
    assert "d" in names and "c" in names and "stop" in names and "away_soon" in names
    assert not any(name == "a" or name.startswith("a_") or "team_flip" in name for name in names)


def play(name="play", n=1500, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.repeat(np.array([H, A, D], dtype=np.int8), 100), n // 300 + 1)[:n]
    players = rng.normal(50, 5, (n, 4, 2))
    ball = np.column_stack([rng.normal(50, 5, (n, 2)), np.full(n, np.nan)])
    return Match(name=name, fps=25, pitch=(105.0, 68.0), half=np.ones(n, dtype=np.int8),
                 ball=ball, players=players, team=np.array([HOME, HOME, AWAY, AWAY], dtype=np.int8),
                 labels=labels)


def tiny_stack():
    """A first and second stage fitted on an invented match, for what can go wrong in saving."""
    m = play()
    x, index = features.build(m)
    table = dict(x=x, labels=model.row_labels(m, index), half=m.half[index])
    first = stack.train_first([table], rounds=3)
    rows = stack.head_rows(first.chances(x), table["half"], table["labels"])
    return stack.Stack(first, *stack.train_heads(*rows, rounds=3))


def test_the_first_stage_answers_every_question_and_every_head():
    m = play()
    x, index = features.build(m)
    first = stack.train_first([dict(x=x, labels=model.row_labels(m, index), half=m.half[index])],
                              rounds=2)
    chances = first.chances(x)
    assert chances.shape == (len(x), 3 + len(stack.QUESTIONS))
    np.testing.assert_array_equal(chances, np.round(chances, stack.CHANCE_DECIMALS))


def test_saved_weights_give_the_same_labels(tmp_path):
    fitted, m = tiny_stack(), play(seed=1)
    fitted.save(tmp_path)
    again = stack.Stack.load(tmp_path)
    np.testing.assert_array_equal(fitted.predict(m, COSTS), again.predict(m, COSTS))


def test_prediction_gives_one_label_per_frame_in_the_scorer_codes():
    m = play(seed=2)
    codes = tiny_stack().predict(m, COSTS)
    assert len(codes) == m.n_frames and set(np.unique(codes)) <= {H, A, D, X}


def rows_of(m):
    return len(features.build(m)[1])


def test_tags_with_no_weight_change_nothing():
    fitted, m = tiny_stack(), play(seed=3)
    np.testing.assert_array_equal(fitted.predict(m, COSTS),
                                  fitted.predict(m, COSTS, signs=np.ones(rows_of(m)), weight=0.0))


def test_tags_pull_frames_towards_the_team_they_name():
    fitted, m = tiny_stack(), play(seed=3)
    home = fitted.predict(m, COSTS, signs=np.ones(rows_of(m)), weight=5.0)
    away = fitted.predict(m, COSTS, signs=-np.ones(rows_of(m)), weight=5.0)
    assert (home == H).sum() > (away == H).sum()
