"""IDSSE: seven Bundesliga matches with DFL tracking and per-frame ball status and possession.

Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and
event data in elite soccer. Scientific Data 12, 195. CC BY 4.0.
Mirror: https://huggingface.co/datasets/pysport/idsse-data (approved by the organisers, issue #16).

    python -m src.external.idsse download     # ~2.5 GB into external_data/idsse, sha256-checked
    python -m src.external.idsse cache        # parse each match once into a .npz Match

Why our own parser rather than kloppy's: kloppy builds a Python object per frame, which for
seven ~400 MB files is slow and memory-hungry. The format is simple -- one <FrameSet> per person
(or the ball) per half, one <Frame N=... X=... Y=...> per 25 Hz tick -- so streaming it straight
into numpy is a page of code. The attribute semantics follow kloppy's sportec deserializer:
BallStatus 1 = alive, BallPossession 1 = home, 2 = away.

These labels are DFL's, not the challenge's. Converting them to the challenge's convention is a
separate, explicit step (src/external/labels.py), never done silently here.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np
from lxml import etree

from ..match import A, AWAY, D, EXTERNAL_DIR, H, HOME, Match, X

DIR = os.path.join(EXTERNAL_DIR, "idsse")
REPO = "https://huggingface.co/datasets/pysport/idsse-data"
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"  # pinned: training must not follow a new push
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
FPS = 25


def _files():
    """{match_id: {kind: (filename, sha256 or None)}} from the mirror's file listing."""
    with urllib.request.urlopen(REPO.replace("/datasets/", "/api/datasets/") + "/tree/" + REVISION) as r:
        listing = json.load(r)
    out = {m: {} for m in MATCHES}
    kinds = {"matchinformation": "meta", "events_raw": "events", "positions_raw": "positions"}
    for entry in listing:
        path = entry["path"]
        for m in MATCHES:
            if path.endswith(m + ".xml"):
                kind = next(v for k, v in kinds.items() if k in path)
                out[m][kind] = (path, (entry.get("lfs") or {}).get("oid"))
    return out


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download():
    os.makedirs(DIR, exist_ok=True)
    for match_id, kinds in _files().items():
        for kind, (name, sha) in sorted(kinds.items()):
            path = os.path.join(DIR, name)
            if not (os.path.isfile(path) and (sha is None or _sha256(path) == sha)):
                urllib.request.urlretrieve("{}/resolve/{}/{}".format(REPO, REVISION, name), path)
            if sha is not None and _sha256(path) != sha:
                raise IOError("sha256 mismatch: {}".format(name))
            print("ok", match_id, kind)


def path_of(match_id, kind):
    tag = {"meta": "matchinformation", "events": "events_raw", "positions": "positions_raw"}[kind]
    names = [n for n in os.listdir(DIR) if tag in n and n.endswith(match_id + ".xml")]
    if len(names) != 1:
        raise IOError("expected one {} file for {} in {}".format(kind, match_id, DIR))
    return os.path.join(DIR, names[0])


def _teams(meta_path):
    """(home_team_id, pitch (L, W), {person_id: HOME/AWAY}) from the match information file."""
    root = etree.parse(meta_path).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    home_id = general.get("HomeTeamId")
    side = {}
    for team in root.iter("Team"):
        code = HOME if team.get("TeamId") == home_id else AWAY
        for player in team.iter("Player"):
            side[player.get("PersonId")] = code
    return home_id, (float(env.get("PitchX")), float(env.get("PitchY"))), side


def _stream(positions_path):
    """{(half, object): {'N': [...], 'X': [...], ...}} -- every FrameSet, as lists of strings."""
    sets, current = {}, None
    for event, elem in etree.iterparse(positions_path, events=("start", "end"), huge_tree=True):
        if event == "start" and elem.tag == "FrameSet":
            half = SECTIONS.get(elem.get("GameSection"))
            obj = "ball" if elem.get("TeamId") == "BALL" else elem.get("PersonId")
            current = None
            if half is not None and elem.get("TeamId") != "referee":
                current = sets.setdefault((half, obj), {})
        elif event == "end" and elem.tag == "Frame":
            if current is not None:
                for key, value in elem.attrib.items():
                    current.setdefault(key, []).append(value)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    return sets


def parse(match_id):
    """One IDSSE match as a Match, labelled with DFL's own ball status and possession."""
    _, pitch, side = _teams(path_of(match_id, "meta"))
    sets = _stream(path_of(match_id, "positions"))
    length, width = pitch
    persons = sorted({obj for (_, obj) in sets if obj != "ball" and obj in side})

    blocks = []
    for half in (1, 2):
        ball = sets[(half, "ball")]
        n = np.array(ball["N"], dtype=np.int64)
        if not np.array_equal(n, np.arange(n[0], n[0] + len(n))):
            raise ValueError("{} half {}: ball frames are not contiguous".format(match_id, half))
        first, T = n[0], len(n)

        b = np.full((T, 3), np.nan, dtype=np.float32)
        b[:, 0] = np.array(ball["X"], dtype=np.float32) + length / 2
        b[:, 1] = np.array(ball["Y"], dtype=np.float32) + width / 2
        b[:, 2] = np.array(ball["Z"], dtype=np.float32)
        status = np.array(ball["BallStatus"], dtype=np.int8)
        owner = np.array(ball["BallPossession"], dtype=np.int8)
        labels = np.where(status != 1, D, np.where(owner == 1, H, A)).astype(np.int8)

        p = np.full((T, len(persons), 2), np.nan, dtype=np.float32)
        for k, person in enumerate(persons):
            frames = sets.get((half, person))
            if not frames:
                continue
            rows = np.array(frames["N"], dtype=np.int64) - first
            keep = (rows >= 0) & (rows < T)
            p[rows[keep], k, 0] = np.array(frames["X"], dtype=np.float32)[keep] + length / 2
            p[rows[keep], k, 1] = np.array(frames["Y"], dtype=np.float32)[keep] + width / 2
        blocks.append((np.full(T, half), n, b, p, labels))

    # One X separator frame between the halves, so no possession straddles halftime.
    def joined(column, separator):
        return np.concatenate([blocks[0][column], separator, blocks[1][column]])

    return Match(
        name="IDSSE-" + match_id, fps=FPS, pitch=pitch,
        half=joined(0, np.array([0])).astype(np.int8),
        source_frame=joined(1, np.array([-1], np.int64)),
        ball=joined(2, np.full((1, 3), np.nan, np.float32)),
        players=joined(3, np.full((1, len(persons), 2), np.nan, np.float32)),
        labels=joined(4, np.array([X], np.int8)),
        team=np.array([side[p] for p in persons], dtype=np.int8),
    )


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
            print("{}  frames {}  players {}  pitch {}".format(
                m.name, m.n_frames, len(m.team), m.pitch))
    sys.exit(0)
