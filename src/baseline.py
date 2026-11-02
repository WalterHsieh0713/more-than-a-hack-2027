"""The organisers' baseline (starter/baseline.py), re-expressed on a Match.

Same two steps: the team of the nearest tracked player wherever the ball is seen, D elsewhere,
then a centred majority vote. Re-expressed rather than imported because starter/baseline.py
streams the raw ATD file, while the open games arrive as a Match. Checked against the shipped
baseline files on BAR-ATH by tests, so the two cannot drift apart unnoticed.
"""

import numpy as np

from .match import A, AWAY, D, H, HOME

NOBODY = -1


def nearest_team(m):
    """Per frame: HOME/AWAY of the player nearest the ball, NOBODY where the ball is not seen."""
    d = np.linalg.norm(m.players - m.ball[:, None, :2], axis=2)        # [T, P]
    seen = ~np.isnan(m.ball[:, 0]) & ~np.all(np.isnan(d), axis=1)
    out = np.full(m.n_frames, NOBODY, dtype=np.int8)
    nearest = np.nanargmin(np.where(np.isnan(d[seen]), np.inf, d[seen]), axis=1)
    out[seen] = m.team[nearest]
    return out


def smooth(codes, window):
    """Centred majority vote over +/- window frames; D where neither team is seen. Ties -> H."""
    if window <= 0:
        return np.where(codes == HOME, H, np.where(codes == AWAY, A, D)).astype(np.int8)
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    nh, na = home[hi] - home[lo], away[hi] - away[lo]
    return np.where(nh + na == 0, D, np.where(nh >= na, H, A)).astype(np.int8)


def predict(m, window=50):
    return smooth(nearest_team(m), window)
