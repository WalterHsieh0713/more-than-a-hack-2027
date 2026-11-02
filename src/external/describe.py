"""What the open games' labels and tracking look like. OTHER GAMES ONLY -- never run on BAR-ATH.

    python -m src.external.describe

Prints per match: label shares, possession count and length, how many possessions are short
(the touch-flips that DFL counts and the challenge does not), dead-ball stretch length, and how
the ball behaves while dead (DFL freezes it, which a model would learn as a shortcut).
"""

import numpy as np

from ..match import A, D, H, X, pm
from . import idsse


def runs_of(codes, value, valid):
    """Lengths of maximal runs of `value`, inside the scorer's valid runs."""
    return np.array([b - a for a, b, k in pm.segments(codes, valid) if k == value])


def describe(m):
    lab = m.labels
    valid = pm.valid_runs(lab)
    graded = lab != X
    spells = pm.possession_spells(lab, valid)
    lengths = np.array([b - a for a, b, _ in spells]) / m.fps
    dead = runs_of(lab, D, valid) / m.fps
    minutes = graded.sum() / m.fps / 60

    # Ball behaviour by state.
    step = np.linalg.norm(np.diff(m.ball[:, :2], axis=0), axis=1) * m.fps
    speed = np.concatenate([[np.nan], step])
    L, W = m.pitch
    x, y = m.ball[:, 0], m.ball[:, 1]
    outside = (x < 0) | (x > L) | (y < 0) | (y > W)
    is_dead, is_alive = lab == D, (lab == H) | (lab == A)

    return {
        "match": m.name,
        "min": minutes,
        "H%": 100 * np.mean(lab[graded] == H),
        "A%": 100 * np.mean(lab[graded] == A),
        "D%": 100 * np.mean(lab[graded] == D),
        "poss": len(spells),
        "poss/min": len(spells) / minutes,
        "mean_s": lengths.mean(),
        "med_s": np.median(lengths),
        "<1s%": 100 * np.mean(lengths < 1),
        "<2s%": 100 * np.mean(lengths < 2),
        "sandwich<2s": sandwiches(lab, valid, 2 * m.fps),
        "dead_runs": len(dead),
        "dead_mean_s": dead.mean(),
        "still_dead%": 100 * np.nanmean(speed[is_dead] < 0.1),
        "still_alive%": 100 * np.nanmean(speed[is_alive] < 0.1),
        "out_dead%": 100 * np.mean(outside[is_dead]),
        "out_alive%": 100 * np.mean(outside[is_alive]),
        "air_alive%": 100 * np.mean(m.ball[is_alive, 2] > 2.0),
    }


def sandwiches(codes, valid, max_frames):
    """X->Y->X spells of the other team shorter than max_frames, with no dead ball around them."""
    segs = pm.segments(codes, valid)
    count = 0
    for i in range(1, len(segs) - 1):
        (a0, b0, k0), (a1, b1, k1), (a2, b2, k2) = segs[i - 1], segs[i], segs[i + 1]
        if k1 in (H, A) and k0 == k2 and k0 in (H, A) and k0 != k1 and b1 - a1 < max_frames:
            count += 1
    return count


if __name__ == "__main__":
    rows = [describe(idsse.load(mid)) for mid in idsse.MATCHES]
    cols = list(rows[0])
    print("  ".join("{:>11}".format(c) for c in cols))
    for r in rows:
        print("  ".join("{:>11}".format(v if isinstance(v, str) else
                                        ("{:.1f}".format(v) if isinstance(v, float) else v))
                        for v in r.values()))
