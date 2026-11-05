"""SkillCorner: twenty A-League 2024/25 matches of automatic tracking from broadcast video, 10 fps.

https://github.com/SkillCorner/opendata (MIT). The provider asks to be credited when this is used.

    python -m src.external.skillcorner download   # ~1.8 GB into external_data/skillcorner
    python -m src.external.skillcorner cache      # parse each match once into a .npz Match

The tracking files are in Git LFS: git holds a pointer naming their sha256, the bytes come from the
media host, and each is checked against that sha256. The README says ten matches; the repository
and data/matches.json hold twenty.

SEEN. The file lists all 22 players on every frame that has data and extrapolates those the camera
cannot see, so is_detected, not presence, decides what was observed: anything else is NaN here.
19-34% of the frames in a half have no data at all (replays) and a further 9-15% have data with
nothing detected (close-ups). All percentages below are the range over the 20 matches.

LABELS, which come from phases_of_play.csv (rows [frame_start, frame_end), their duration confirms
the end is exclusive) and are SkillCorner's own, not the challenge's:
  H/A  inside a phase: its team_in_possession_id
  D    in a half, outside every phase, with at least one object detected
  X    in a half, outside every phase, with nothing detected; and every frame outside the halves

Why D is the complement of the phases: the README says there is no phase while the ball is out of
play. That gives 78-155 gaps a match, mean 104, against 106-107 stoppages in Metrica, and 90% of
the 2011 gaps of 2 s or more line up with a game_interruption in dynamic_events.csv, which this
loader does not read. The other way round, 85-94% of those interruptions fall on a gap, so about one in ten
stoppages is inside a phase and labelled live. Gap ends rest on the phase start, so trust them to a
second or two at best. Null possession is the worse proxy: 8-16% of seen frames outside phases
still name a team, and 1-3% of those inside them do not.

Why X and not D where nothing was seen: this is what can be shown, not what is likely. Phases take
up 45-63% of a half, a normal effective playing time, so most of these frames probably are dead.
But 91% of gaps contain an unseen stretch, and in 63% of those the phase starts within a second of
sight returning, so the restart cannot be placed: it was hidden, or the director cut back for it.
Calling these frames D would also let a model score by reading "no objects" as dead, an artefact
of broadcast replays that a tactical camera does not have. The cost is that X cuts stoppages up,
so label_stats' stoppages and restart counts mean little here, and its per-minute rates are per
labelled minute. X frames keep their half, as they are still match time. To count every gap as
dead instead: labels[(half > 0) & (labels == X)] = D.

Why the phase team and not the per-frame possession.group: on frames where dynamic_events names a
player in possession, the phase team agrees with that player's team 99-100% of the time and group
81-91%. It is not a lag (on six matches, the best time shift gains about a point), and group flips
about 1.5 times as often, nearer the touch-by-touch flips CHALLENGE.md does not count as possession.
The cost is that spells come out longer and short ones rarer than Metrica's.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

from ..match import A, AWAY, D, EXTERNAL_DIR, H, HOME, Match, X
from .idsse import _sha256

DIR = os.path.join(EXTERNAL_DIR, "skillcorner")
REVISION = "4340d274572876239c154c90bc507a9b3250a656"  # pinned: labels must not follow a new push
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/" + REVISION + "/"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/" + REVISION + "/"
MATCHES = ("1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691",
           "1996435", "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166",
           "2013725", "2015213", "2016236", "2017461")
TRACKING, INFO, PHASES = "_tracking_extrapolated.jsonl", "_match.json", "_phases_of_play.csv"
FPS = 10


def _lfs_sha256(name):
    """The sha256 in the pointer file git keeps in place of an LFS object."""
    with urllib.request.urlopen(RAW + name) as r:
        fields = dict(line.split(" ", 1) for line in r.read().decode().splitlines())
    return fields["oid"].split(":")[1]


def download():
    os.makedirs(DIR, exist_ok=True)
    for match_id in MATCHES:
        stem = "data/matches/" + match_id + "/" + match_id
        for suffix in (INFO, PHASES):
            urllib.request.urlretrieve(RAW + stem + suffix, os.path.join(DIR, match_id + suffix))
        path, sha = os.path.join(DIR, match_id + TRACKING), _lfs_sha256(stem + TRACKING)
        if not (os.path.isfile(path) and _sha256(path) == sha):
            urllib.request.urlretrieve(MEDIA + stem + TRACKING, path)
        if _sha256(path) != sha:
            raise IOError("sha256 mismatch: " + path)
        print("ok", match_id)


def _labels(half, seen, phases, teams):
    labels = np.where(seen, D, X).astype(np.int8)
    for start, stop, team in phases:
        labels[start:stop] = teams[team]
    labels[half == 0] = X
    return labels


def parse(match_id, folder=None):
    """One match as a Match, labelled as the module docstring says."""
    base = os.path.join(folder or DIR, match_id)
    with open(base + INFO) as handle:
        info = json.load(handle)
    with open(base + TRACKING) as handle:
        rows = [json.loads(line) for line in handle]
    T = len(rows)

    n = np.array([r["frame"] for r in rows], dtype=np.int64)
    if not np.array_equal(n, np.arange(n[0], n[0] + T)):
        raise ValueError(match_id + ": frames are not contiguous")
    half = np.array([r["period"] or 0 for r in rows], dtype=np.int8)
    if np.any((half[:-1] > 0) & (half[1:] > 0) & (half[:-1] != half[1:])):
        raise ValueError(match_id + ": no separator frame between the halves")

    home, away = info["home_team"]["id"], info["away_team"]["id"]
    side = {p["id"]: HOME if p["team_id"] == home else AWAY for p in info["players"]}
    persons = sorted({p["player_id"] for r in rows for p in r["player_data"]})
    slot = {person: k for k, person in enumerate(persons)}

    ball = np.full((T, 3), np.nan, dtype=np.float32)
    players = np.full((T, len(persons), 2), np.nan, dtype=np.float32)
    for t, r in enumerate(rows):
        b = r["ball_data"]
        if b["is_detected"]:
            ball[t] = b["x"], b["y"], b["z"]
        for p in r["player_data"]:
            if p["is_detected"]:
                players[t, slot[p["player_id"]]] = p["x"], p["y"]

    pitch = float(info["pitch_length"]), float(info["pitch_width"])
    centre = np.array(pitch, dtype=np.float32) / 2
    ball[:, :2] += centre
    players += centre
    seen = np.isfinite(ball[:, 0]) | np.isfinite(players[:, :, 0]).any(axis=1)

    phases = pd.read_csv(base + PHASES)
    spans = zip(phases.frame_start, phases.frame_end, phases.team_in_possession_id)
    return Match(name="SkillCorner-" + match_id, fps=FPS, pitch=pitch, half=half, ball=ball,
                 players=players, team=np.array([side[p] for p in persons], dtype=np.int8),
                 labels=_labels(half, seen, spans, {home: H, away: A}), source_frame=n)


def cache_path(match_id):
    return os.path.join(DIR, "cache", match_id + ".npz")


def load(match_id):
    """The cached Match, parsing it first if needed."""
    path = cache_path(match_id)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        parse(match_id).save(path)
    return Match.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("download", "cache"))
    parser.add_argument("--match", nargs="+", default=list(MATCHES))
    args = parser.parse_args()
    if args.command == "download":
        download()
    else:
        for match_id in args.match:
            m = load(match_id)
            print(m.name, "frames", m.n_frames, "players", len(m.team), "pitch", m.pitch)
    sys.exit(0)
