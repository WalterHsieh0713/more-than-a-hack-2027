"""Team-level features, five rows a second, with no track identity and no labels.

Everything is summed over a team, because a slot does not follow a person. Directions are measured
towards the goal each team attacks, worked out per half from where the teams stand. Possession
persists through loose balls, so the last and next moments of apparent control are carried across
gaps. Windows look back and ahead separately, which lets a model place a change between them.
"""

from dataclasses import replace

import numpy as np

from .match import A, AWAY, H, HOME, X

HZ = 5
WINDOWS_S = (0.4, 1, 2, 4)
CONTROL_M = 2.0             # a player this close to a ball moving slower than the next line
CONTROL_SPEED = 8.0         # m/s
NEAR_M = 3.0
TOP_SPEED = 12.0            # m/s: faster steps are tracker jumps
SLOW = 1.5                  # m/s
MARGIN = 1.0                # m beyond the line that counts as out
CAP_S = 30.0
PACE_SECONDS = 1            # speed over this long, because tracker jitter cancels out over it
MOVING = 3.0                # m/s
LINE_M = 1.5
BOX_DEPTH, BOX_WIDTH = 16.5, 40.3

ROLLED = ("near", "control", "seen", "speed_all")
ROLLED_LONG = ("pace", "movers", "on_lines")     # stoppages last far longer than possessions
LONG_WINDOWS_S = (1, 4, 10, 20)
NAMES = (["seen", "gap_age", "gap_ahead", "ball_depth", "ball_wide", "ball_speed", "ball_out",
          "dist_home", "dist_away", "close_home", "close_away", "speed_home", "speed_away",
          "speed_all", "slow_share", "depth_home", "depth_away", "spread_home", "spread_away",
          "count_home", "count_away", "push_home", "push_away", "control",
          "last_control", "last_age", "next_control", "next_age", "since_half",
          "pace", "movers", "on_lines", "own_half", "box_attacked_by_home",
          "box_attacked_by_away", "team_gap"]
         + ["{}_{}_{:g}s".format(name, side, w)
            for name in ROLLED for side in ("past", "next") for w in WINDOWS_S]
         + ["{}_{}_{:g}s".format(name, side, w)
            for name in ROLLED_LONG for side in ("past", "next") for w in LONG_WINDOWS_S])

BALL_FREE = ("speed_home", "speed_away", "speed_all", "slow_share", "depth_home", "depth_away",
             "spread_home", "spread_away", "count_home", "count_away", "push_home", "push_away",
             "since_half", "pace", "movers", "on_lines", "own_half", "box_attacked_by_home",
             "box_attacked_by_away", "team_gap")
PLAYER_ONLY = [name for name in NAMES if name in BALL_FREE
               or name.split("_past_")[0].split("_next_")[0] in ROLLED_LONG + ("speed_all",)]


def half_spans(m):
    """[(half, first_frame, stop_frame)] for each half present."""
    spans = []
    for half in (1, 2):
        frames = np.flatnonzero(m.half == half)
        if len(frames):
            spans.append((half, int(frames[0]), int(frames[-1]) + 1))
    return spans


def attack_sign(m, start, stop):
    """+1 if home attacks towards high x in this stretch, else -1."""
    x = m.players[start:stop, :, 0]
    home = np.nanmean(x[:, m.team == HOME])
    away = np.nanmean(x[:, m.team == AWAY])
    return 1 if home < away else -1


def flip_pitch(m):
    """The same play seen from the other side of the ground."""
    length, width = m.pitch
    players, ball = m.players.copy(), m.ball.copy()
    players[..., 0], players[..., 1] = length - players[..., 0], width - players[..., 1]
    ball[:, 0], ball[:, 1] = length - ball[:, 0], width - ball[:, 1]
    return replace(m, players=players, ball=ball)


def swap_teams(m):
    """The same play with home and away exchanged, labels included. Doubles the training data
    and stops a model learning that home teams have the ball more."""
    labels = None
    if m.labels is not None:
        labels = m.labels.copy()
        labels[m.labels == H], labels[m.labels == A] = A, H
    return replace(m, team=(AWAY - m.team + HOME).astype(m.team.dtype), labels=labels)


def past(values, window):
    """Mean of the window rows before each row, NaN where they hold nothing."""
    return _mean_over(values, -window, 0)


def future(values, window):
    """Mean of the window rows after each row, NaN where they hold nothing."""
    return _mean_over(values, 1, window + 1)


def _mean_over(values, first, stop):
    known = np.isfinite(values)
    totals = np.concatenate(([0.0], np.cumsum(np.where(known, values, 0.0))))
    counts = np.concatenate(([0.0], np.cumsum(known)))
    i = np.arange(len(values))
    low, high = np.clip(i + first, 0, len(values)), np.clip(i + stop, 0, len(values))
    n = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (totals[high] - totals[low]) / n, np.nan)


def carry(sign):
    """(latest non-zero sign at or before each row, rows since it), NaN before the first."""
    i = np.arange(len(sign))
    last = np.maximum.accumulate(np.where(sign != 0, i, -1))
    value = np.where(last >= 0, sign[np.maximum(last, 0)], np.nan)
    return value, np.where(last >= 0, (i - last).astype(np.float32), np.nan)


def _team(values, mask):
    """Per frame mean of [T, P] values over a team's seen slots, NaN if none are seen."""
    picked = values[:, mask]
    seen = np.isfinite(picked)
    count = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, np.where(seen, picked, 0.0).sum(axis=1) / count, np.nan), count


def _rows(values, step):
    """Mean of each step frames, ignoring NaN, with a shorter last row."""
    index = np.arange(len(values)) // step
    known = np.isfinite(values)
    totals = np.bincount(index, np.where(known, values, 0.0))
    counts = np.bincount(index, known)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / counts, np.nan)


def _half(m, start, stop, step):
    length, width = m.pitch
    players = np.asarray(m.players[start:stop], dtype=np.float64)
    home, away = m.team == HOME, m.team == AWAY
    n, frames = stop - start, np.arange(stop - start)
    sign = attack_sign(m, start, stop)

    ball = m.ball[start:stop, :2]
    seen = np.isfinite(ball[:, 0])
    if seen.any():
        path = np.column_stack([np.interp(frames, frames[seen], ball[seen, k]) for k in (0, 1)])
    else:
        path = np.full((n, 2), np.nan)
    last_seen = np.maximum.accumulate(np.where(seen, frames, -1))
    next_seen = np.minimum.accumulate(np.where(seen, frames, n)[::-1])[::-1]
    age = np.minimum(np.where(last_seen >= 0, frames - last_seen, n) / m.fps, CAP_S)
    ahead = np.minimum(np.where(next_seen < n, next_seen - frames, n) / m.fps, CAP_S)

    step_m = np.linalg.norm(np.diff(path, axis=0, prepend=path[:1]), axis=1)
    ball_speed = _smooth_speed(step_m, m.fps)
    depth = path[:, 0] / length if sign > 0 else 1 - path[:, 0] / length
    wide = np.abs(path[:, 1] / width - 0.5) * 2
    out = (path[:, 0] < -MARGIN) | (path[:, 0] > length + MARGIN) | (path[:, 1] < -MARGIN) \
        | (path[:, 1] > width + MARGIN)

    dist = np.sqrt(((players - path[:, None, :]) ** 2).sum(axis=2))
    d_home, d_away = np.fmin.reduce(dist[:, home], axis=1), np.fmin.reduce(dist[:, away], axis=1)
    close_home, close_away = (dist[:, home] < NEAR_M).sum(axis=1), (dist[:, away] < NEAR_M).sum(axis=1)
    with np.errstate(invalid="ignore"):
        near = np.where(~seen | np.isnan(np.fmin(d_home, d_away)), 0.0,
                        np.sign(np.nan_to_num(d_away, nan=1e9) - np.nan_to_num(d_home, nan=1e9)))
        control = np.where(np.fmin(d_home, d_away) < CONTROL_M, near, 0.0)
        control = np.where(ball_speed < CONTROL_SPEED, control, 0.0)

    velocity = np.diff(players, axis=0, prepend=players[:1]) * m.fps
    speed = np.sqrt((velocity ** 2).sum(axis=2))
    fast = speed > TOP_SPEED
    speed[fast] = np.nan
    forward = np.where(fast, np.nan, velocity[:, :, 0])

    speed_home, count_home = _team(speed, home)
    speed_away, count_away = _team(speed, away)
    all_seen = home | away
    speed_all, _ = _team(speed, all_seen)
    slow, _ = _team(np.where(np.isnan(speed), np.nan, (speed < SLOW).astype(float)), all_seen)

    x = players[:, :, 0]
    x_home, _ = _team(x, home)
    x_away, _ = _team(x, away)
    square_home, _ = _team(x ** 2, home)
    square_away, _ = _team(x ** 2, away)
    spread_home = np.sqrt(np.maximum(square_home - x_home ** 2, 0))
    spread_away = np.sqrt(np.maximum(square_away - x_away ** 2, 0))
    depth_home = x_home / length if sign > 0 else 1 - x_home / length
    depth_away = 1 - x_away / length if sign > 0 else x_away / length
    push_home, _ = _team(forward * sign, home)
    push_away, _ = _team(forward * -sign, away)

    stride = max(1, int(PACE_SECONDS * m.fps))
    moved = np.full(players.shape[:2], np.nan)
    if n > stride:
        half_way = stride // 2
        moved[half_way:half_way + n - stride] = np.sqrt(
            ((players[stride:] - players[:-stride]) ** 2).sum(axis=2)) / PACE_SECONDS
    moved[moved > TOP_SPEED] = np.nan
    pace, _ = _team(moved, all_seen)
    with np.errstate(invalid="ignore"):
        movers = (moved > MOVING).sum(axis=1).astype(float)
        y = players[:, :, 1]
        on_lines = ((x < LINE_M) | (x > length - LINE_M) | (y < LINE_M)
                    | (y > width - LINE_M)).sum(axis=1).astype(float)
        home_side = (x[:, home] < length / 2) if sign > 0 else (x[:, home] > length / 2)
        away_side = (x[:, away] > length / 2) if sign > 0 else (x[:, away] < length / 2)
        own = (home_side.sum(axis=1) + away_side.sum(axis=1)) / np.maximum(
            np.isfinite(x[:, all_seen]).sum(axis=1), 1)
        central = np.abs(y - width / 2) < BOX_WIDTH / 2
        low_box, high_box = (x < BOX_DEPTH) & central, (x > length - BOX_DEPTH) & central
    low, high = low_box.sum(axis=1).astype(float), high_box.sum(axis=1).astype(float)

    per_row = {
        "pace": pace, "movers": movers, "on_lines": on_lines, "own_half": own,
        "box_attacked_by_home": high if sign > 0 else low,
        "box_attacked_by_away": low if sign > 0 else high,
        "team_gap": np.abs(x_home - x_away),
        "seen": seen.astype(float), "gap_age": age, "gap_ahead": ahead, "ball_depth": depth,
        "ball_wide": wide, "ball_speed": ball_speed, "ball_out": out.astype(float),
        "dist_home": d_home, "dist_away": d_away, "close_home": close_home.astype(float),
        "close_away": close_away.astype(float), "speed_home": speed_home,
        "speed_away": speed_away, "speed_all": speed_all, "slow_share": slow,
        "depth_home": depth_home, "depth_away": depth_away, "spread_home": spread_home,
        "spread_away": spread_away, "count_home": count_home.astype(float),
        "count_away": count_away.astype(float), "push_home": push_home, "push_away": push_away,
        "control": control, "near": near,
    }
    table = {name: _rows(values, step) for name, values in per_row.items()}
    rows = len(table["seen"])
    rounded = np.sign(table["control"])
    table["last_control"], last_age = carry(rounded)
    table["last_age"] = last_age / HZ
    following, next_age = carry(rounded[::-1])
    table["next_control"], table["next_age"] = following[::-1], next_age[::-1] / HZ
    table["since_half"] = np.minimum(np.arange(rows) / HZ, CAP_S * 2)
    for name in ROLLED:
        for w in WINDOWS_S:
            width_rows = max(1, int(round(w * HZ)))
            table["{}_past_{:g}s".format(name, w)] = past(table[name], width_rows)
            table["{}_next_{:g}s".format(name, w)] = future(table[name], width_rows)
    for name in ROLLED_LONG:
        for w in LONG_WINDOWS_S:
            table["{}_past_{:g}s".format(name, w)] = past(table[name], int(w * HZ))
            table["{}_next_{:g}s".format(name, w)] = future(table[name], int(w * HZ))
    return np.column_stack([table[name] for name in NAMES]).astype(np.float32), \
        start + np.arange(rows) * step


def _smooth_speed(step_m, fps, frames=5):
    """Ball speed in m/s over a short window, because single-frame steps are mostly jitter."""
    return np.convolve(np.nan_to_num(step_m), np.ones(frames) / frames, mode="same") * fps


def build(m):
    """(features [rows, len(NAMES)], first frame of each row) over both halves."""
    step = max(1, m.fps // HZ)
    parts = [_half(m, start, stop, step) for _, start, stop in half_spans(m)]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def expand(codes, index, m):
    """Row labels back to one label per frame, X outside the graded frames."""
    out = np.full(len(m.half), X, dtype=np.int8)
    step = max(1, m.fps // HZ)
    for code, first in zip(codes, index):
        stop = first + step
        out[first:stop] = np.where(m.half[first:stop] > 0, code, X)
    return out
