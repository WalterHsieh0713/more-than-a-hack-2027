"""Degrade clean tracking until it resembles automatic, uncorrected tracking.

A model trained on clean open data (fixed identities, ball nearly always present, ball height known,
ball frozen by the provider while dead) will not transfer, so training sees it through degrade().

Every setting below is a judgement call, deliberately wide, and nothing here is measured from the
competition match. Re-centre them from external automatic tracking. All are read at call time.

    uv run python -m src.noise J03WMX     # smoke run on one open match, with its runtime
"""

import dataclasses
import math
import sys
import time

import numpy as np
import pandas as pd

from .match import AWAY, HOME, Match

SLOTS_PER_TEAM = 23

# Ball loss starts as a burst, with a hazard per second that grows with how hard the moment is.
BALL_LOSS_BASE_PER_S = 0.05
BALL_LOSS_HEIGHT_PER_S = 1.0      # added in full once the ball reaches BALL_HEIGHT_REF_M
BALL_LOSS_CROWD_PER_S = 0.5       # added in full with BALL_CROWD_REF players around the ball
BALL_HEIGHT_REF_M = 3.0
BALL_CROWD_REF = 6
BALL_CROWD_RADIUS_M = 3.0
BALL_GAP_MEAN_S = 1.5

# A provider-frozen ball is replaced by a slow walk through waypoints to where play restarts.
FROZEN_MIN_S = 0.5                # exactly still for this long counts as frozen
FROZEN_WANDER_M = 3.0             # spread of the waypoints around the straight line
FROZEN_WAYPOINT_S = 2.0
FROZEN_LOSS_PER_S = 0.5           # replaces the hazard above: a stale ball says nothing of height

TRACKLET_MEAN_S = 20.0            # shorter above level 1; not level-scaled below, as this is identity
PLAYER_GAP_MEAN_S = 2.0           # time unseen between one tracklet and the next
SWAP_TRIES_PER_MIN = 6.0          # match-wide; a try only swaps if a pair is within the radius
SWAP_RADIUS_M = 4.0
DUPLICATES_PER_MIN = 2.0          # match-wide
DUPLICATE_MEAN_S = 2.0
DUPLICATE_OFFSET_M = 1.0

PLAYER_JITTER_M = 0.3
BALL_JITTER_M = 0.3
JITTER_TAU_S = 0.2
DURATION_SIGMA = 1.0              # lognormal shape of every random duration


def _runs(mask):
    """(starts, stops) of the True runs in a boolean array, stops exclusive."""
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _frozen(xy, min_frames):
    """Frames in a run of at least min_frames exactly equal positions. NaN is never still."""
    same = np.r_[False, (xy[1:] == xy[:-1]).all(1)]
    frozen = np.zeros(len(xy), bool)
    for start, stop in zip(*_runs(same)):
        if stop - start + 1 >= min_frames:
            frozen[start - 1:stop] = True
    return frozen


def _spread(rng, size=None):
    """Random multipliers of mean 1, heavy-tailed like real track and gap durations."""
    return rng.lognormal(-DURATION_SIGMA ** 2 / 2, DURATION_SIGMA, size)


def _jitter(rng, shape, tau_frames):
    """Unit-variance AR(1) noise along the first axis: tracker error drifts, it does not flicker."""
    alpha = 1 - math.exp(-1 / tau_frames)
    white = rng.standard_normal((shape[0], int(np.prod(shape[1:]))))
    white[0] *= math.sqrt(alpha / (2 - alpha))
    red = pd.DataFrame(white).ewm(alpha=alpha, adjust=False).mean().to_numpy(np.float32)
    return (red * math.sqrt((2 - alpha) / alpha)).reshape(shape)


def _walk(rng, start, stop, n, fps, wander):
    """n positions strictly between start and stop, through random waypoints."""
    inner = 1 + int(n / (FROZEN_WAYPOINT_S * fps))   # at least one: start may equal stop
    # One waypoint in the middle of each time cell: no clockwork rhythm, and none can bunch up.
    u = np.r_[0, (np.arange(inner) + rng.uniform(0.25, 0.75, inner)) / inner, 1]
    way = start + u[:, None] * (stop - start) + rng.normal(0, wander, (inner + 2, 2))
    way[0], way[-1] = start, stop
    t = np.arange(1, n + 1) / (n + 1)
    return np.column_stack([np.interp(t, u, way[:, axis]) for axis in (0, 1)])


def _degrade_ball(m, rng, level):
    T, fps = m.n_frames, m.fps
    xy = m.ball[:, :2].astype(np.float32)
    frozen = _frozen(xy, round(FROZEN_MIN_S * fps))

    height = np.nan_to_num(m.ball[:, 2])
    crowd = (((m.players - xy[:, None]) ** 2).sum(2) < BALL_CROWD_RADIUS_M ** 2).sum(1)
    hazard = (BALL_LOSS_BASE_PER_S
              + BALL_LOSS_HEIGHT_PER_S * np.clip(height / BALL_HEIGHT_REF_M, 0, 1)
              + BALL_LOSS_CROWD_PER_S * np.minimum(crowd / BALL_CROWD_REF, 1))
    hazard[frozen] = FROZEN_LOSS_PER_S
    starts = np.flatnonzero(rng.random(T) < 1 - np.exp(-level * hazard / fps))
    lengths = 1 + (BALL_GAP_MEAN_S * fps * _spread(rng, len(starts))).astype(np.int64)
    depth = np.zeros(T + 1, np.int64)
    np.add.at(depth, starts, 1)
    np.add.at(depth, np.minimum(starts + lengths, T), -1)
    lost = depth.cumsum()[:-1] > 0

    # The freeze ends with a jump to the restart spot, so jitter alone would leave a teleport that
    # marks the exact frame play resumes. Hiding the stretch would only make missing mean dead.
    out = xy.copy()
    for start, stop in zip(*_runs(frozen)):
        before = xy[start - 1] if start and np.isfinite(xy[start - 1, 0]) else xy[start]
        after = xy[stop] if stop < T and np.isfinite(xy[stop, 0]) else xy[stop - 1]
        out[start:stop] = _walk(rng, before, after, stop - start, fps,
                                FROZEN_WANDER_M * max(level, 1))

    out += level * BALL_JITTER_M * _jitter(rng, out.shape, JITTER_TAU_S * fps)
    out[lost] = np.nan
    return np.column_stack([out, np.full(T, np.nan, np.float32)])


def _degrade_players(m, rng, level):
    T, P, fps = m.n_frames, len(m.team), m.fps
    minutes = T / fps / 60

    # Swaps first: a column of `people` stops being one person, whatever slots it visits later.
    people = m.players.copy()
    teammates = m.team[:, None] == m.team
    for t in rng.integers(T, size=rng.poisson(level * SWAP_TRIES_PER_MIN * minutes)):
        apart = np.linalg.norm(people[t, :, None] - people[t, None], axis=2)
        apart[~teammates | np.isnan(apart)] = np.inf
        np.fill_diagonal(apart, np.inf)
        i, j = np.unravel_index(apart.argmin(), apart.shape)
        if apart[i, j] < SWAP_RADIUS_M:
            people[t:, [i, j]] = people[t:, [j, i]]

    tracklets = []                               # (first frame, stop frame, column, position offset)
    for k in range(P):
        t = 0
        while t < T:
            stop = t + 1 + int(TRACKLET_MEAN_S / max(level, 1) * fps * _spread(rng))
            tracklets.append((t, min(stop, T), k, np.zeros(2)))
            t = stop + int(level * PLAYER_GAP_MEAN_S * fps * _spread(rng))
    for t in rng.integers(T, size=rng.poisson(level * DUPLICATES_PER_MIN * minutes)):
        stop = t + 1 + int(DUPLICATE_MEAN_S * fps * _spread(rng))
        tracklets.append((t, min(stop, T), rng.integers(P), rng.normal(0, DUPLICATE_OFFSET_M, 2)))

    out = np.full((T, 2 * SLOTS_PER_TEAM, 2), np.nan, np.float32)
    free_from = np.zeros(2 * SLOTS_PER_TEAM, np.int64)
    for start, stop, k, offset in sorted(tracklets, key=lambda tracklet: tracklet[0]):
        first = int(m.team[k]) * SLOTS_PER_TEAM
        free = first + np.flatnonzero(free_from[first:first + SLOTS_PER_TEAM] <= start)
        if free.size:                            # none free: the tracker dropped this one
            slot = rng.choice(free)
            free_from[slot] = stop
            out[start:stop, slot] = people[start:stop, k] + offset

    return out + level * PLAYER_JITTER_M * _jitter(rng, out.shape, JITTER_TAU_S * fps)


def degrade(m: Match, seed: int, level: float = 1.0) -> Match:
    """A noisy copy of m. Never looks at m.labels, which pass through unchanged.

    level scales the stochastic noise. At level=0 ball height is still gone, slots are still
    anonymous, and the frozen ball still moves.
    """
    rng = np.random.default_rng(seed)
    ball = _degrade_ball(m, rng, level)
    players = _degrade_players(m, rng, level)
    team = np.repeat(np.array([HOME, AWAY], np.int8), SLOTS_PER_TEAM)
    return dataclasses.replace(m, ball=ball, players=players, team=team)


if __name__ == "__main__":
    from .external import idsse

    source = idsse.load(sys.argv[1] if len(sys.argv) > 1 else idsse.MATCHES[0])
    began = time.perf_counter()
    noisy = degrade(source, seed=0)
    seconds = time.perf_counter() - began
    played = source.half > 0
    objects = np.isfinite(noisy.players[played, :, 0]).sum(1)
    print(source.name, "seconds", round(seconds, 1),
          "ball lost", round(float(np.isnan(noisy.ball[played, 0]).mean()), 3),
          "objects per frame", round(float(objects.mean()), 1))
