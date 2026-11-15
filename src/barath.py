"""BAR-ATH's automatic tracking as a Match: row i is submission frame i.

The raw file is "<frame_id>:<player>;<player>;...:<ball>", each group "x,y,speed" normalised 0-1
with NaN for a missing object. Malformed input is refused rather than skipped, because a dropped
record would shift every later frame and the score would be quietly wrong.

Positions stay float64: they get the same arithmetic as starter/baseline.py, so the shipped
baselines can be reproduced frame for frame.
"""

import sys

import numpy as np

from .match import Match, ROOT, X

sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402


def blocks():
    """[(half, first_frame, last_frame)], the graded video frames."""
    return half_video_blocks(GAME)


def half_array(n_frames, graded):
    """int8 [T]: 1 or 2 on graded frames, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in graded:
        half[first:last + 1] = number
    return half


def _xy(group):
    fields = group.split(",")
    return float(fields[0]), float(fields[1])


def _read(raw_path, n_frames, n_slots):
    """(players [T, P, 2], ball [T, 2]), normalised, exactly one record per frame."""
    players = np.full((n_frames, n_slots, 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    count = 0
    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            parts = line.split(":")
            if len(parts) != 3:
                raise ValueError("record {} does not have three sections".format(count))
            if count > n_frames or int(parts[0]) != count:
                raise ValueError("record {} is not the next frame".format(count))
            groups = parts[1].split(";")
            if len(groups) != n_slots:
                raise ValueError("record {} has the wrong number of slots".format(count))
            row = count - 1
            for slot, group in enumerate(groups):
                if group[0] != "N":
                    players[row, slot] = _xy(group)
            if parts[2][0] != "N":
                ball[row] = _xy(parts[2])
    if count != n_frames:
        raise ValueError("file has {} records, expected {}".format(count, n_frames))
    return players, ball


def parse(meta_path, raw_path, n_frames, pitch, graded):
    team = team_of_each_track(meta_path)
    players, ball = _read(raw_path, n_frames, len(team))
    scale = np.array(pitch)
    return Match(name=GAME, fps=GAMES[GAME]["fps"], pitch=tuple(pitch),
                 half=half_array(n_frames, graded), team=team, players=players * scale,
                 ball=np.column_stack([ball * scale, np.full(n_frames, np.nan)]),
                 source_frame=np.arange(1, n_frames + 1, dtype=np.int64))


def load(data_dir=None):
    # ponytail: re-parses every call, add a cache when the wait starts to hurt.
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    return parse(meta_path, raw_path, source_frames(), GAMES[GAME]["pitch"], blocks())
