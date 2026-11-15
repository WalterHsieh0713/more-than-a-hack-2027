"""Make clean optical tracking look like an uncorrected single-wide-camera tracker's output.

degrade(m, seed, level) reasons from how such a tracker fails, not from any measurement of the
competition match. Every setting is a module constant with a wide, plausible default; nothing is
fitted. `level` scales all noise amplitudes and rates; what is not noise but a change of
representation -- 46 anonymous slots, no ball height, no frozen dead ball -- happens at every level.

Failure modes, and what on the pitch drives each:
  ball        single camera, so an aerial ball is projected onto the grass away from the camera
              (parallax); lost more when high, fast or in a crowd, in bursts not single frames;
              snaps to a nearby player's feet; unseen far outside the pitch; jitter grows with
              distance from the camera.
  frozen ball the provider holds the ball still while it is dead. A camera sees whoever has it,
              so a frozen ball rides its nearest person instead (this is why labels are never read:
              dead ball is found from the ball alone).
  players     occlusion loses people, more in crowds, in bursts; a lost person comes back in a new
              slot; two close team-mates swap tracks; a person is duplicated (ghost) for a moment;
              a track lands in the other team's slots; jitter grows with distance from the camera.
"""

import numpy as np

from .match import AWAY, HOME, Match

SLOTS = 46                      # 23 per team, HOME first
CAM_DIST = 35.0                 # m from the near touchline to the camera
CAM_HEIGHT = 18.0               # m
PARALLAX = 1.0                  # fraction of the single-camera ground projection applied
FAR_GAIN = 1.0                  # jitter at the far touchline is (1 + FAR_GAIN) times the near one
PLAYER_JITTER = 0.3             # m, s.d. at the near touchline
BALL_JITTER = 0.5               # m
PLAYER_MISS = 0.02              # per-frame chance a lone person is unseen
CROWD_RADIUS = 2.0              # m; each neighbour this close adds CROWD_MISS
CROWD_MISS = 0.04
GAP_SECONDS = 1.0               # time scale of a dropout burst
BREAK_PER_MINUTE = 2.0          # per person: track re-identified in a fresh slot (not scaled)
TEAM_FLIP = 0.03                # chance a track segment is filed under the other team
GHOST_PER_MINUTE = 3.0          # duplicated persons per minute
GHOST_SECONDS = 2.0             # mean ghost life
GHOST_OFFSET = 1.0              # m, s.d. of a ghost's fixed offset from its person
SWAP_PER_MINUTE = 3.0           # track swaps between two close team-mates
SWAP_RADIUS = 1.5               # m
BALL_MISS = 0.15                # per-frame chance a ball on the ground in open space is unseen
BALL_MISS_HEIGHT = 0.10         # extra per metre of height
BALL_MISS_CROWD = 0.10          # extra per person within BALL_CROWD_RADIUS
BALL_CROWD_RADIUS = 2.0
BALL_MISS_SPEED = 0.01          # extra per m/s
BALL_GAP_SECONDS = 1.5
BALL_SNAP = 0.05                # per-frame chance the ball is read off a player within SNAP_RADIUS
SNAP_RADIUS = 3.0
OUT_MARGIN = 8.0                # m outside the pitch beyond which the ball is out of frame
FREEZE_MIN_FRAMES = 5           # identical ball positions this long mean the provider froze it


def camera(rng, length, width):
    """(x, y, z) of the wide camera, on a random touchline side so neither side is learnable."""
    y = -CAM_DIST if rng.random() < 0.5 else width + CAM_DIST
    return np.array([length / 2, y, CAM_HEIGHT])


def project(xy, z, cam):
    """Where a single camera at cam sees an object at (xy, height z) once flattened to the grass."""
    z = np.minimum(z, 0.9 * cam[2])
    scale = (cam[2] / (cam[2] - z))[:, None]
    return cam[:2] + (xy - cam[:2]) * scale


def _correlated_uniform(rng, shape, fps, seconds):
    """Uniform(0, 1) per entry with AR(1) memory along axis 0, so chance events come in bursts.
    ponytail: logistic approximation of the normal CDF (max error 0.01) to avoid scipy."""
    rho = np.exp(-1.0 / (fps * seconds))
    e = rng.standard_normal(shape)
    x = np.empty(shape)
    x[0] = e[0]
    k = np.sqrt(1 - rho * rho)
    for t in range(1, shape[0]):
        x[t] = rho * x[t - 1] + k * e[t]
    return 1.0 / (1.0 + np.exp(-1.702 * x))


def _neighbours(players, radius):
    """[T, P] count of other persons within radius."""
    T, P, _ = players.shape
    out = np.zeros((T, P))
    for a in range(0, T, 4096):
        chunk = players[a:a + 4096]
        d = np.linalg.norm(chunk[:, :, None] - chunk[:, None], axis=-1)
        close = d < radius
        close[:, np.arange(P), np.arange(P)] = False
        out[a:a + 4096] = close.sum(-1)
    return out


def _far_gain(y, cam, width):
    return 1.0 + FAR_GAIN * (np.abs(y - cam[1]) - CAM_DIST) / width


def _segments(visible, cuts):
    """(person, start, stop) for each run of visible frames, split where cuts is True."""
    T, P = visible.shape
    prev = np.vstack([np.zeros((1, P), bool), visible[:-1]])
    nxt = np.vstack([visible[1:], np.zeros((1, P), bool)])
    cut_next = np.vstack([cuts[1:], np.zeros((1, P), bool)])
    starts = visible & (~prev | cuts)
    stops = visible & (~nxt | cut_next)
    out = []
    for p in range(P):
        s, e = np.flatnonzero(starts[:, p]), np.flatnonzero(stops[:, p]) + 1
        out.extend((p, a, b) for a, b in zip(s, e))
    return out


def degrade(m: Match, seed: int, level: float = 1.0) -> Match:
    rng = np.random.default_rng(seed)
    length, width = m.pitch
    T, P, fps = m.n_frames, len(m.team), m.fps
    cam = camera(rng, length, width)
    minute = 60 * fps
    players = m.players.astype(np.float64)
    ball = m.ball[:, :2].astype(np.float64)
    z = np.nan_to_num(m.ball[:, 2].astype(np.float64))
    seen_p = np.isfinite(players[:, :, 0])
    dist_pb = np.linalg.norm(np.where(seen_p[:, :, None], players, np.inf) - ball[:, None], axis=-1)
    nearest = dist_pb.argmin(1)
    nearest_xy = players[np.arange(T), nearest]
    has_near = np.isfinite(dist_pb.min(1))

    # --- ball -------------------------------------------------------------------------------
    same_prev = np.r_[False, (np.diff(ball, axis=0) == 0).all(1) & np.isfinite(ball[1:, 0])]
    frozen = same_prev | np.r_[same_prev[1:], False]
    run_start = np.flatnonzero(frozen & ~np.r_[False, frozen[:-1]])
    run_stop = np.flatnonzero(frozen & ~np.r_[frozen[1:], False]) + 1
    for a, b in zip(run_start, run_stop):
        if b - a < FREEZE_MIN_FRAMES:
            frozen[a:b] = False
    carried = frozen & has_near
    ball = np.where(carried[:, None], nearest_xy, ball)

    speed = np.r_[0, np.linalg.norm(np.diff(m.ball[:, :2].astype(np.float64), axis=0), axis=1)] * fps
    speed = np.nan_to_num(speed)
    ball = ball + level * PARALLAX * (project(ball, z, cam) - ball)
    snap = _correlated_uniform(rng, (T,), fps, BALL_GAP_SECONDS) < level * BALL_SNAP
    snap &= has_near & (dist_pb.min(1) < SNAP_RADIUS)
    ball = np.where(snap[:, None], nearest_xy, ball)
    ball = ball + level * BALL_JITTER * _far_gain(ball[:, 1], cam, width)[:, None] * rng.standard_normal((T, 2))
    crowd_b = (dist_pb < BALL_CROWD_RADIUS).sum(1)
    hazard = level * (BALL_MISS + BALL_MISS_HEIGHT * z + BALL_MISS_CROWD * crowd_b + BALL_MISS_SPEED * speed)
    lost = _correlated_uniform(rng, (T,), fps, BALL_GAP_SECONDS) < np.clip(hazard, 0, 0.98)
    out_of_frame = ((ball < -OUT_MARGIN) | (ball > np.array([length, width]) + OUT_MARGIN)).any(1)
    ball[lost | out_of_frame] = np.nan
    ball_out = np.full((T, 3), np.nan, dtype=np.float32)
    ball_out[:, :2] = ball

    # --- players: who is seen when ------------------------------------------------------------
    hazard = level * (PLAYER_MISS + CROWD_MISS * _neighbours(players, CROWD_RADIUS))
    visible = seen_p & (_correlated_uniform(rng, (T, P), fps, GAP_SECONDS) >= np.clip(hazard, 0, 0.98))
    cuts = rng.random((T, P)) < BREAK_PER_MINUTE / minute
    segments = [(p, a, b, None) for p, a, b in _segments(visible, cuts)]
    for _ in range(rng.poisson(level * GHOST_PER_MINUTE * T / minute)):
        a = rng.integers(T)
        b = min(T, a + 1 + rng.poisson(GHOST_SECONDS * fps))
        who = np.flatnonzero(visible[a])
        if len(who):
            segments.append((rng.choice(who), a, b, rng.normal(0, GHOST_OFFSET, 2)))
    segments.sort(key=lambda s: (s[1], s[0]))

    # --- players: slot allocation, one fresh random free slot per segment -----------------------
    pool_of = np.where(m.team == HOME, 0, 1)
    flip = rng.random(len(segments)) < level * TEAM_FLIP
    free_at = np.zeros(SLOTS, dtype=np.int64)
    last_slot = np.full(P, -1)                              # a re-acquired track gets a new id
    out = np.full((T, SLOTS, 2), np.nan)
    for (p, a, b, offset), f in zip(segments, flip):
        pool = pool_of[p] ^ int(f)
        free = np.flatnonzero(free_at[pool * 23:(pool + 1) * 23] <= a) + pool * 23
        if len(free) > 1:
            free = free[free != last_slot[p]]
        if not len(free):
            continue                                        # tracker has no slot left: lost
        slot = rng.choice(free)
        free_at[slot] = b
        if offset is None:
            last_slot[p] = slot
        xy = players[a:b, p] + (0 if offset is None else offset)
        xy = xy + level * PLAYER_JITTER * _far_gain(xy[:, 1], cam, width)[:, None] * rng.standard_normal(xy.shape)
        out[a:b, slot] = xy

    # --- players: two close team-mates swap tracks ---------------------------------------------
    for t in rng.integers(T, size=rng.poisson(level * SWAP_PER_MINUTE * T / minute)):
        row = out[t]
        d = np.linalg.norm(row[:, None] - row[None], axis=-1)
        same_pool = (np.arange(SLOTS) < 23)[:, None] == (np.arange(SLOTS) < 23)[None]
        i, j = np.nonzero(np.triu(d < SWAP_RADIUS, 1) & same_pool)
        if len(i):
            k = rng.integers(len(i))
            out[t:, [i[k], j[k]]] = out[t:, [j[k], i[k]]]

    return Match(name=m.name, fps=fps, pitch=m.pitch, half=m.half, ball=ball_out,
                 players=out.astype(np.float32),
                 team=np.array([HOME] * 23 + [AWAY] * 23, dtype=np.int8),
                 labels=m.labels, source_frame=m.source_frame, meta=dict(m.meta, noise_b=(seed, level)))
