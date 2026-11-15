"""The possession model: three gradient-boosted heads, then the decoder.

One head says whether the ball is dead, from the players alone, because how a ball looks when dead
depends on the tracker. One says which team has it, from live play. One says whether a change is
near, which sharpens boundaries. Trained only on other matches, each seen through noise copies and with the
teams swapped. Single-threaded and deterministic, and scored from saved files, because LightGBM
promises repeatable training only on one system.
"""

import os

import lightgbm as lgb
import numpy as np

from . import features, tags
from .decode import decode
from .match import A, D, H, X

ROUNDS = 150
PARAMS = {"learning_rate": 0.06, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.7,
          "lambda_l2": 5.0, "seed": 7, "deterministic": True, "force_col_wise": True,
          "verbose": -1}
NEAR_ROWS = 3               # 0.6 s either side counts as near a change
HEADS = ("dead", "team", "change")
PLAYER_COLUMNS = [features.NAMES.index(name) for name in features.PLAYER_ONLY]


def booster(path):
    """Weights from a LightGBM text file. Read in text mode, so CRLF line endings, which Git on
    Windows may add on checkout and LightGBM cannot parse, come back as plain newlines."""
    with open(path, encoding="utf-8") as handle:
        return lgb.Booster(model_str=handle.read())


def row_labels(m, index):
    """The label at the middle frame of each row."""
    step = max(1, m.fps // features.HZ)
    return m.labels[np.minimum(index + step // 2, len(m.labels) - 1)]


def near_change(labels, reach):
    """1 for rows within reach of a label change, 0 elsewhere."""
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    out = np.zeros(len(labels), dtype=np.int8)
    for change in changes:
        out[max(0, change - reach):change + reach] = 1
    return out


def table(m):
    """(features, labels, near-a-change targets) for a match's graded rows."""
    x, index = features.build(m)
    labels = row_labels(m, index)
    keep = labels != X
    return x[keep], labels[keep], near_change(labels, NEAR_ROWS)[keep]


def fill_blanks(m):
    """Bridge ball gaps in a straight line, from the ball alone. For tracking where a blank ball
    marks dead time, as in Metrica's corrected samples, so that noise can then be added fairly."""
    ball = m.ball.copy()
    frames = np.arange(len(ball))
    seen = np.isfinite(ball[:, 0])
    if seen.any():
        for k in (0, 1):
            ball[:, k] = np.interp(frames, frames[seen], ball[seen, k])
    return features.replace(m, ball=ball)


def training_rows(matches, copies):
    """Rows from every match through every noise copy, each with the teams swapped as well."""
    parts = []
    for m in matches:
        for copy in copies:
            seen = copy(m)
            parts.append(table(seen))
            parts.append(table(features.swap_teams(seen)))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def team_rows(x, y):
    """(rows of live play, 1 where away has the ball) for the team head."""
    keep = y != D
    return keep, (y[keep] == A).astype(np.int8)


class Model:
    def __init__(self, dead, team, change):
        self.dead, self.team, self.change = dead, team, change

    def _chances(self, x):
        dead = self.dead.predict(x[:, PLAYER_COLUMNS], num_threads=1)
        away = self.team.predict(x, num_threads=1)
        p = np.empty((len(x), 3))
        p[:, H], p[:, A], p[:, D] = (1 - dead) * (1 - away), (1 - dead) * away, dead
        return p, self.change.predict(x, num_threads=1)

    def proba(self, m):
        """(H, A, D chances [rows, 3], chance of a change [rows]) for a match's rows."""
        return self._chances(features.build(m)[0])

    def predict(self, m, costs, gain=1, signs=None, weight=0.0):
        """One label per frame: decoded separately within each half. signs and weight add
        Smart Tagging evidence, one sign per row."""
        x, index = features.build(m)
        p, b = self._chances(x)
        if signs is not None:
            p = tags.apply(p, signs, weight)
        half = m.half[index]
        codes = np.concatenate([decode(p[half == h], b[half == h], costs, gain)
                                for h in np.unique(half)])
        return features.expand(codes, index, m)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        for name in HEADS:
            getattr(self, name).save_model(os.path.join(folder, name + ".txt"))

    @classmethod
    def load(cls, folder):
        return cls(*[booster(os.path.join(folder, name + ".txt")) for name in HEADS])


def fit(matches, copies, rounds=ROUNDS, threads=1):
    return train(*training_rows(matches, copies), rounds=rounds, threads=threads)


def train(x, y, near, rounds=ROUNDS, threads=1):
    data = dict(PARAMS, num_threads=threads, objective="binary")
    dead = lgb.train(data, lgb.Dataset(x[:, PLAYER_COLUMNS], (y == D).astype(np.int8),
                                       free_raw_data=False), rounds)
    keep, away = team_rows(x, y)
    team = lgb.train(data, lgb.Dataset(x[keep], away, free_raw_data=False), rounds)
    change = lgb.train(data, lgb.Dataset(x, near, free_raw_data=False), rounds)
    return Model(dead, team, change)
