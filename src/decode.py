"""Turn per-step probabilities into whole possessions.

Plain Viterbi, in integers so that the same probabilities give the same labels on any machine.
Dead ball has two hidden states, one for each team that last had the ball, which lets a restart
cost differ for the same team and the other team. Both come out as D.
"""

from dataclasses import dataclass

import numpy as np

from .match import A, D, H

SCALE = 1000                # log-probabilities kept to three decimals
_H, _A, _DEAD_AFTER_H, _DEAD_AFTER_A = 0, 1, 2, 3
_LABEL = np.array([H, A, D, D], dtype=np.int8)
_FORBIDDEN = 10 ** 12       # dead ball cannot change who last had the ball


@dataclass(frozen=True)
class Costs:
    turnover: int
    stop: int
    restart_other: int
    restart_same: int

    def __post_init__(self):
        for value in (self.turnover, self.stop, self.restart_other, self.restart_same):
            if not isinstance(value, (int, np.integer)):
                raise TypeError("costs must be whole numbers")

    def matrix(self):
        """Cost of moving from row state to column state."""
        t, s, other, same, no = (self.turnover, self.stop, self.restart_other, self.restart_same,
                                 _FORBIDDEN)
        return np.array([[0, t, s, no],
                         [t, 0, no, s],
                         [same, other, 0, no],
                         [other, same, no, 0]], dtype=np.int64)


def viterbi(emit, costs, discount=None):
    """Best label path for integer scores [n, 3]. discount[t] is taken off every switch at step t,
    so a negative discount, such as a log-probability, makes switching there dearer."""
    n = len(emit)
    scores = emit[:, [H, A, D, D]].astype(np.int64)
    base = costs.matrix()
    switching = ~np.eye(4, dtype=bool)
    best = scores[0].copy()
    back = np.zeros((n, 4), dtype=np.int8)
    for t in range(1, n):
        step = base - switching * (0 if discount is None else int(discount[t]))
        options = best[:, None] - step
        back[t] = options.argmax(axis=0)
        best = options[back[t], np.arange(4)] + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = best.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return _LABEL[path]


def _scaled_log(values):
    return np.rint(np.log(np.clip(values, 1e-6, 1)) * SCALE).astype(np.int64)


def decode(proba, boundary, costs, gain=1):
    """Labels for probabilities [n, 3] in H, A, D order and boundary chances [n]."""
    return viterbi(_scaled_log(proba)[:, [H, A, D]], costs, gain * _scaled_log(boundary))
