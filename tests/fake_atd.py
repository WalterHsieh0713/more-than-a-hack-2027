"""Synthetic files in the ATD's on-disk format. Every number is invented; no match data."""

import os

import numpy as np

PITCH = (105.3, 68.0)


def random_tracking(seed, n_frames, n_slots, missing=0.3):
    """(players [T, P, 2], ball [T, 2], team [P]) normalised 0-1, NaN where unseen."""
    rng = np.random.default_rng(seed)
    players, ball = rng.random((n_frames, n_slots, 2)), rng.random((n_frames, 2))
    players[rng.random((n_frames, n_slots)) < missing] = np.nan
    ball[rng.random(n_frames) < missing] = np.nan
    return players, ball, rng.integers(0, 2, n_slots).astype(np.int8)


def _group(x, y):
    return "NaN,NaN,NaN" if np.isnan(x) else "{!r},{!r},0.0".format(float(x), float(y))


def records(players, ball):
    """One raw line per frame, numbered from 1 as the ATD is."""
    return ["{}:{}:{}\n".format(i + 1, ";".join(_group(x, y) for x, y in row), _group(*b))
            for i, (row, b) in enumerate(zip(players, ball))]


def write_meta(folder, team):
    path = os.path.join(folder, "meta.xml")
    slots = "".join('<Player teamId="{}"/>'.format("away" if t else "home") for t in team)
    with open(path, "w") as handle:
        handle.write('<Meta><Teams><Team id="home"><Name>Home FC</Name></Team>'
                     '<Team id="away"><Name>Away FC</Name></Team></Teams>'
                     '<Players>{}</Players></Meta>'.format(slots))
    return path


def write_raw(folder, lines):
    path = os.path.join(folder, "raw.txt")
    with open(path, "w") as handle:
        handle.writelines(lines)
    return path


def write_atd(folder, players, ball, team):
    return write_meta(folder, team), write_raw(folder, records(players, ball))
