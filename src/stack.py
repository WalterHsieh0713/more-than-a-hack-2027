"""Second stage: LightGBM reads the first stage's chances across time.

The first stage judges one moment. This one sees the shape around it: every chance at the row, at
lags either side, over past and future windows, with slopes and the time since and until the last
flip. Beside the frozen three heads the first stage answers seven more questions from the same
features: is a stop, a start or a turnover near, is the ball dead judged with the ball in view, who
has it judged from player shape alone, and who has it a second from now and had it a second ago.

Most of the gain is better dead-ball calls. The frozen dead head sees players only, because a dead
ball looks different in each tracker. Here ball-reading chances reach it through a model trained
to weigh them across simulators, and that held up on real tracking. The team questions gain on
simulators only and split real possessions too finely, so turnovers cost twice what stops do.

Chances for training are out of fold, split by match, so the second stage cannot learn to trust a
model that has already seen the play. Inner fits use half the noisy copies, which costs little
because the copies repeat each other, and the first stage used for prediction uses them all.
Everything is rounded before it reaches a model, and decode rounds what comes out.

The dead chance used is the plain mean of this stack's and a ball-free partner's, weight fixed at
1/2. The partner's dead side reads only questions answered from players, because reading the ball
helps on every simulator yet failed badly when trained on one simulator and tested on the other.
"""

import os

import lightgbm as lgb
import numpy as np

from . import experiment, features, model, tags
from .decode import Costs, decode
from .match import A, D, H, X

GROUPS = 4
SIMULATED = ("noise", "noise_b")
CHANCE_DECIMALS = 4
DECIMALS = 3
ROUNDS = 150
COSTS = Costs(turnover=2400, stop=1200, restart_other=960, restart_same=1440)
GAIN = 1
SERIES = "dac"              # dead, away, change: the columns of a chance array
LAGS = (1, 2, 3, 5, 8, 13, 25)
WINDOWS = (5, 10, 25, 50, 100)
SPANS = (5, 10, 25)
PEAK_FLOOR, PEAK_REACH = 0.3, 3
HEADS = ("dead", "team", "change")
QUESTIONS = ("stop", "start", "turn", "dead_any", "team_shape", "away_soon", "away_was")
SHIFT_ROWS = 5              # one second
QUESTION_LAGS = (2, 5, 13)
QUESTION_WINDOWS = (5, 25)
FREE_QUESTIONS = ("stop", "start", "dead_any", "team_shape")   # from players only when ball-free


def _lagged(values, rows):
    return values[np.clip(np.arange(len(values)) + rows, 0, len(values) - 1)]


def _span(values, first, stop, kind):
    """Max or min of values[t + first:t + stop] for each t, with the ends repeated."""
    n, width = len(values), stop - first
    padded = np.concatenate([np.full(max(0, -first), values[0]), values,
                             np.full(max(0, stop), values[-1])])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    start = max(0, -first) + first
    return getattr(windows, kind)(axis=1)[start:start + n]


def _since(flags):
    """Rows since flags was last set, counted from before the start if it never was."""
    i = np.arange(len(flags))
    last = np.maximum.accumulate(np.where(flags, i, -1))
    return np.where(last >= 0, i - last, i + 1).astype(np.float32)


def _flips(flags):
    return np.concatenate(([False], flags[1:] != flags[:-1]))


def _half_columns(chances):
    """{name: values} for one half of chances [rows, 3 + questions], SERIES then QUESTIONS."""
    series = {s: chances[:, k].astype(np.float64) for k, s in enumerate(SERIES)}
    dead, away, change = series["d"], series["a"], series["c"]
    out = dict(series)
    for k, name in enumerate(QUESTIONS):
        v = chances[:, len(SERIES) + k].astype(np.float64)
        out[name] = v
        for lag in QUESTION_LAGS:
            out[f"{name}_before_{lag}"] = _lagged(v, -lag)
            out[f"{name}_after_{lag}"] = _lagged(v, lag)
        for w in QUESTION_WINDOWS:
            before, after = features.past(v, w), features.future(v, w)
            before, after = np.where(np.isnan(before), v, before), np.where(np.isnan(after), v, after)
            out[f"{name}_past_{w}"] = before
            out[f"{name}_next_{w}"] = after
            out[f"{name}_rise_{w}"] = after - before
    for s, v in series.items():
        for k in LAGS:
            out[f"{s}_before_{k}"] = _lagged(v, -k)
            out[f"{s}_after_{k}"] = _lagged(v, k)
    for s, v in series.items():
        for w in WINDOWS:
            before, after = features.past(v, w), features.future(v, w)
            before, after = np.where(np.isnan(before), v, before), np.where(np.isnan(after), v, after)
            out[f"{s}_past_{w}"] = before
            out[f"{s}_next_{w}"] = after
            out[f"{s}_rise_{w}"] = after - before
    for w in SPANS:
        out[f"a_high_past_{w}"] = _span(away, -w, 0, "max")
        out[f"a_low_past_{w}"] = _span(away, -w, 0, "min")
        out[f"a_high_next_{w}"] = _span(away, 1, w + 1, "max")
        out[f"a_low_next_{w}"] = _span(away, 1, w + 1, "min")
        out[f"c_high_past_{w}"] = _span(change, -w, 0, "max")
        out[f"c_high_next_{w}"] = _span(change, 1, w + 1, "max")
        out[f"d_high_past_{w}"] = _span(dead, -w, 0, "max")
        out[f"d_high_next_{w}"] = _span(dead, 1, w + 1, "max")
    peaks = (change >= _span(change, -PEAK_REACH, PEAK_REACH + 1, "max")) & (change > PEAK_FLOOR)
    for name, flags in (("team_flip", _flips(away > 0.5)), ("dead_flip", _flips(dead > 0.5)),
                        ("peak", peaks)):
        out["since_" + name] = _since(flags)
        out["until_" + name] = _since(flags[::-1])[::-1]
    return out


COLUMNS = list(_half_columns(np.zeros((2, len(SERIES) + len(QUESTIONS)))))
DEAD_COLUMNS = [i for i, name in enumerate(COLUMNS)
                if name != "a" and not name.startswith("a_") and not name.endswith("team_flip")]
FREE_DEAD_COLUMNS = [i for i, name in enumerate(COLUMNS)
                     if name.split("_")[0] in ("d", "stop", "start", "dead")]


def dead_columns(free):
    return FREE_DEAD_COLUMNS if free else DEAD_COLUMNS


def columns(chances, half):
    """Second-stage features [rows, len(COLUMNS)], each from its own half alone."""
    parts = [_half_columns(chances[half == h]) for h in np.unique(half)]
    x = np.vstack([np.column_stack([part[name] for name in COLUMNS]) for part in parts])
    return np.round(x.astype(np.float32), DECIMALS)


def _near_kind(labels, kind):
    """1 for rows near a change of one kind: stop (live to dead), start, or turn (team to team)."""
    before, after = labels[:-1], labels[1:]
    live_before, live_after = np.isin(before, (H, A)), np.isin(after, (H, A))
    picked = {"turn": (before != after) & live_before & live_after,
              "stop": live_before & (after == D), "start": (before == D) & live_after}[kind]
    out = np.zeros(len(labels), dtype=np.int8)
    for change in np.flatnonzero(picked) + 1:
        out[max(0, change - model.NEAR_ROWS):change + model.NEAR_ROWS] = 1
    return out


def question(name, labels, half):
    """(rows to ask about, the 0/1 answer on those rows) for one first-stage question."""
    graded, live = labels != X, np.isin(labels, (H, A))
    if name in ("stop", "start", "turn"):
        return graded, _near_kind(labels, name)[graded]
    if name == "dead_any":
        return graded, (labels[graded] == D).astype(np.int8)
    if name == "team_shape":
        return live, (labels[live] == A).astype(np.int8)
    step = SHIFT_ROWS if name == "away_soon" else -SHIFT_ROWS
    there = np.arange(len(labels)) + step
    keep = graded & (there >= 0) & (there < len(labels))
    keep[keep] = (half[there[keep]] == half[keep]) & live[there[keep]]
    return keep, (labels[there[keep]] == A).astype(np.int8)


def _question_columns(name, free=False):
    players = name == "team_shape" or (free and name in FREE_QUESTIONS)
    return model.PLAYER_COLUMNS if players else slice(None)


class First:
    """The frozen recipe's three heads and the extra questions, asked of the same features.
    free: the stop, start and dead questions see players only."""

    def __init__(self, heads, questions, free=False):
        self.heads, self.questions, self.free = heads, questions, free

    def chances(self, x):
        """Rounded chances [rows, 3 + questions]: dead, away, change, then QUESTIONS."""
        parts = [self.heads.dead.predict(x[:, model.PLAYER_COLUMNS], num_threads=1),
                 self.heads.team.predict(x, num_threads=1),
                 self.heads.change.predict(x, num_threads=1)]
        parts += [self.questions[name].predict(x[:, _question_columns(name, self.free)],
                                               num_threads=1)
                  for name in QUESTIONS]
        return np.round(np.column_stack(parts), CHANCE_DECIMALS).astype(np.float32)

    def save(self, folder):
        self.heads.save(folder)
        for name in QUESTIONS:
            self.questions[name].save_model(os.path.join(folder, name + ".txt"))

    @classmethod
    def load(cls, folder, free=False):
        return cls(model.Model.load(folder),
                   {name: model.booster(os.path.join(folder, name + ".txt"))
                    for name in QUESTIONS}, free)


def train_first(tables, rounds=model.ROUNDS, threads=1, free=False):
    """The first stage on tables of dict(x, labels, half), one per noisy copy."""
    keep = [t["labels"] != X for t in tables]
    x = np.concatenate([t["x"][k] for t, k in zip(tables, keep)])
    y = np.concatenate([t["labels"][k] for t, k in zip(tables, keep)])
    near = np.concatenate([model.near_change(t["labels"], model.NEAR_ROWS)[k]
                           for t, k in zip(tables, keep)])
    heads = model.train(x, y, near, rounds=rounds, threads=threads)
    del x, y, near
    params = dict(model.PARAMS, num_threads=threads, objective="binary")
    questions = {}
    for name in QUESTIONS:
        answers = [question(name, t["labels"], t["half"]) for t in tables]
        rows = np.concatenate([t["x"][a[0]] for t, a in zip(tables, answers)])
        data = lgb.Dataset(rows[:, _question_columns(name, free)],
                           np.concatenate([a[1] for a in answers]), free_raw_data=False)
        questions[name] = lgb.train(params, data, rounds)
    return First(heads, questions, free)


def inner_groups(names):
    names = sorted(names)
    return [names[i::GROUPS] for i in range(GROUPS)]


def out_of_fold(tables, first):
    """{name: [score of each table]}. first(names) trains on those matches and returns a scorer
    score(name, table), and a match is only ever scored by what was trained without it."""
    out = {}
    for held in inner_groups(tables):
        score = first([name for name in tables if name not in held])
        for name in held:
            out[name] = [score(name, table) for table in tables[name]]
    return out


def head_rows(chances, half, labels):
    """(columns, labels, near-a-change) on one table's graded rows."""
    keep = labels != X
    return (columns(chances, half)[keep], labels[keep],
            model.near_change(labels, model.NEAR_ROWS)[keep])


def train_heads(x, y, near, rounds=ROUNDS, threads=1, free=False):
    params = dict(model.PARAMS, num_threads=threads, objective="binary")
    dead = lgb.train(params, lgb.Dataset(x[:, dead_columns(free)], (y == D).astype(np.int8),
                                         free_raw_data=False), rounds)
    keep, away = model.team_rows(x, y)
    team = lgb.train(params, lgb.Dataset(x[keep], away, free_raw_data=False), rounds)
    change = lgb.train(params, lgb.Dataset(x, near, free_raw_data=False), rounds)
    return dead, team, change


class Stack:
    """partner: a ball-free Stack whose dead chance is averaged with this one's."""

    def __init__(self, first, dead, team, change, partner=None):
        self.first, self.dead, self.team, self.change = first, dead, team, change
        self.partner = partner

    def dead_chance(self, x, half):
        """(this stack's own dead chance per row, its second-stage features)."""
        row = columns(self.first.chances(x), half)
        return self.dead.predict(row[:, dead_columns(self.first.free)], num_threads=1), row

    def _one_view(self, m):
        x, index = features.build(m)
        half = m.half[index]
        dead, row = self.dead_chance(x, half)
        if self.partner is not None:
            dead = (dead + self.partner.dead_chance(x, half)[0]) / 2
        away = self.team.predict(row, num_threads=1)
        change = self.change.predict(row, num_threads=1)
        p = np.empty((len(row), 3))
        p[:, H], p[:, A], p[:, D] = (1 - dead) * (1 - away), (1 - dead) * away, dead
        return p, change, half, index

    def chances(self, m, both_sides=False):
        """(H, A, D chances per row, change chance per row, half per row, row indices).
        both_sides averages in a second pass with the teams swapped, so that neither team's
        answer depends on which one the file calls home."""
        p, change, half, index = self._one_view(m)
        if both_sides:
            q, other = self._one_view(features.swap_teams(m))[:2]
            p, change = (p + q[:, [A, H, D]]) / 2, (change + other) / 2
        return p, change, half, index

    def predict(self, m, costs=COSTS, gain=GAIN, signs=None, weight=0.0, both_sides=False):
        """One label per frame, decoded separately within each half. signs and weight add
        Smart Tagging evidence, one sign per row."""
        p, change, half, index = self.chances(m, both_sides)
        if signs is not None:
            p = tags.apply(p, signs, weight)
        codes = np.concatenate([decode(p[half == h], change[half == h], costs, gain)
                                for h in np.unique(half)])
        return features.expand(codes, index, m)

    def save(self, folder):
        self.first.save(os.path.join(folder, "first"))
        for name in HEADS:
            getattr(self, name).save_model(os.path.join(folder, name + ".txt"))
        if self.partner is not None:
            self.partner.save(os.path.join(folder, "partner"))

    @classmethod
    def load(cls, folder, free=False):
        """A partner folder, if there is one, always holds the ball-free side."""
        partner = os.path.join(folder, "partner")
        return cls(First.load(os.path.join(folder, "first"), free),
                   *[model.booster(os.path.join(folder, name + ".txt")) for name in HEADS],
                   partner=cls.load(partner, free=True) if os.path.isdir(partner) else None)


def noisy_tables(m):
    """Every noisy copy of a match as rows, tagged with its level and team order."""
    out = []
    for name in SIMULATED:
        for level, copy in zip(experiment.TRAIN_LEVELS, experiment.copies(experiment.SIMULATORS[name])):
            seen = copy(m)
            for swapped, view in enumerate((seen, features.swap_teams(seen))):
                x, index = features.build(view)
                out.append(dict(x=x, labels=model.row_labels(view, index), half=view.half[index],
                                inner=(level == experiment.TRAIN_LEVELS[0]) == (swapped == 0)))
    return out


def train(matches, threads, free=False):
    """Both stages on these matches' noisy copies."""
    tables = {name: noisy_tables(m) for name, m in matches.items()}
    inner = {name: [t for t in copies if t["inner"]] for name, copies in tables.items()}

    def first(names):
        trained = train_first([t for name in names for t in inner[name]], threads=threads,
                              free=free)

        def score(name, table):
            return head_rows(trained.chances(table["x"]), table["half"], table["labels"])
        return score
    held = out_of_fold(inner, first)
    parts = [part for name in sorted(held) for part in held[name]]
    x, y, near = (np.concatenate([part[k] for part in parts]) for k in range(3))
    del held, parts
    heads = train_heads(x, y, near, threads=threads, free=free)
    del x, y, near
    everything = [t for copies in tables.values() for t in copies]
    return Stack(train_first(everything, threads=threads, free=free), *heads)


def train_averaged(matches, threads):
    """The finalist: a stack with a ball-free partner, trained the same way on the same matches."""
    full = train(matches, threads)
    full.partner = train(matches, threads, free=True)
    return full


def fit(matches, threads):
    """A challenger for src.challenge: fit(train_matches, threads) -> predict(match)."""
    return train(matches, threads).predict


def fit_averaged(matches, threads):
    return train_averaged(matches, threads).predict
