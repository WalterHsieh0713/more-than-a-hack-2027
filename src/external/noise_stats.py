"""How noisy automatic tracking is, measured on a Match. OTHER GAMES ONLY -- never run on BAR-ATH.

    python -m src.external.noise_stats

Every number is over frames inside a half (half > 0), whether or not they hold any data, because a
blackout is the noise. Anything counted in runs is cut at the half. A position is seen if it is not
NaN, so the numbers depend on the loader's NaN rule: SkillCorner's extrapolated objects count as
unseen.

  ball_visible          share of frames with a ball position
  ball_gaps             number of maximal runs of frames without one
  ball_gap_median_s     their median length, seconds; ball_gap_p90_s the 90th percentile
  objects_mean          visible players plus ball per frame, blank frames counted as 0
  objects_mean_seen     the same over frames with at least one object
  players_mean_seen     visible players per frame, over frames with at least one object
  empty_share           share of frames with no object at all
  player_missing_share  share of frames with fewer than `full` visible players (22, so a sent-off
                        player counts as missing). A broadcast view never holds all 22, so this is
                        1 on SkillCorner: use players_mean_seen to set a dropout rate
  track_run_median_s    median length of a run of frames in which one slot stays visible;
                        track_run_p90_s the 90th percentile. A slot here is whatever the source
                        gives: a re-identified player in SkillCorner, a tracklet in an ATD. So on
                        SkillCorner this measures staying in view, and is an upper bound on how
                        long a raw tracker keeps hold of someone
  jump_share            share of consecutive visible steps of one slot faster than JUMP_SPEED, the
                        one trace identity swaps leave behind. It is speed per step, so it grows
                        with frame rate: compare sources only after resampling to the same fps.
                        SkillCorner's is near 0 yet its README admits about 3% wrong identities, so
                        its post-processing hides swaps: do not set a swap rate from this number

Lengths are in seconds so sources compare, but a 10 fps source cannot show a gap under 0.1 s.
"""

import numpy as np

from ..label_stats import label_stats
from ..match import pm
from . import metrica, skillcorner

FULL = 22
JUMP_SPEED = 12.0  # m/s: above any footballer's top speed, so not a real step


def _runs(mask):
    """Lengths of the maximal runs of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return edges[1::2] - edges[::2]


def _percentile(values, q, scale=1.0):
    return float(np.percentile(values, q)) / scale if len(values) else np.nan


def noise_stats(m, full=FULL):
    players_seen = np.isfinite(m.players[:, :, 0])
    ball_seen = np.isfinite(m.ball[:, 0])
    halves = [m.half == h for h in np.unique(m.half[m.half > 0])]
    in_half = m.half > 0

    gaps = np.concatenate([_runs(~ball_seen[h]) for h in halves])
    tracks = np.concatenate([_runs(players_seen[h, k]) for h in halves
                             for k in range(players_seen.shape[1])])

    players = players_seen[in_half].sum(axis=1)
    objects = players + ball_seen[in_half]
    some = objects > 0

    same_half = in_half[1:] & (m.half[1:] == m.half[:-1])
    speed = np.linalg.norm(np.diff(m.players, axis=0), axis=2)[same_half] * m.fps
    speed = speed[np.isfinite(speed)]

    return {
        "ball_visible": float(ball_seen[in_half].mean()),
        "ball_gaps": len(gaps),
        "ball_gap_median_s": _percentile(gaps, 50, m.fps),
        "ball_gap_p90_s": _percentile(gaps, 90, m.fps),
        "objects_mean": float(objects.mean()),
        "objects_mean_seen": float(objects[some].mean()),
        "players_mean_seen": float(players[some].mean()),
        "empty_share": float((~some).mean()),
        "player_missing_share": float((players < full).mean()),
        "track_run_median_s": _percentile(tracks, 50, m.fps),
        "track_run_p90_s": _percentile(tracks, 90, m.fps),
        "jump_share": float((speed > JUMP_SPEED).mean()) if len(speed) else np.nan,
    }


def ranges(rows):
    """{key: (lowest, highest)} across a list of stats dicts."""
    return {key: (min(r[key] for r in rows), max(r[key] for r in rows)) for key in rows[0]}


def _print_ranges(title, rows):
    print(title)
    for key, (low, high) in ranges(rows).items():
        print("  %-24s%10.4g%10.4g" % (key, low, high))


def _label_row(labels, m):
    """label_stats, plus each label as a share of the frames inside a half."""
    shares = {"share_" + pm.LABELS[k]: float(np.mean(labels[m.half > 0] == k)) for k in range(4)}
    return {**shares, **label_stats(labels, m.fps)}


def _all_gaps_dead(m):
    """SkillCorner's labels with unseen gaps counted as dead, see skillcorner.py."""
    return np.where((m.half > 0) & (m.labels == pm.X), pm.D, m.labels)


if __name__ == "__main__":
    skill = [skillcorner.load(match_id) for match_id in skillcorner.MATCHES]
    meta = [metrica.load(game) for game in metrica.GAMES]
    _print_ranges("SkillCorner noise, " + str(len(skill)) + " matches", [noise_stats(m) for m in skill])
    _print_ranges("SkillCorner labels", [_label_row(m.labels, m) for m in skill])
    _print_ranges("SkillCorner labels, unseen gaps as D", [_label_row(_all_gaps_dead(m), m) for m in skill])
    _print_ranges("Metrica labels", [_label_row(m.labels, m) for m in meta])
