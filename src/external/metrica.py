"""Metrica Sports: three anonymised matches of optical tracking with events, 25 fps.

https://github.com/metrica-sports/sample-data. Data courtesy of Metrica Sports. The repository has
no licence file; its README only asks for responsible use and for the source to be acknowledged.
So it is fetched by script, never committed, and anything public must credit Metrica.

    python -m src.external.metrica download   # ~180 MB into external_data/metrica
    python -m src.external.metrica cache      # parse each game once into a .npz Match

Games 1 and 2 are CSV: one tracking file per team, with the same frames and the same ball. Game 3
is FIFA/EPTS: tracking and metadata as text and XML, events as JSON. barath._read parses its raw
lines but barath.parse does not apply: the metadata lists 35 players against 22 columns, and each
DataFormatSpecification block says which player a column holds for its frame range. A substitute
inherits the column of the player they replace, so reading columns positionally would merge two
people into one slot. Across all nine in-half boundaries nobody moves more than 0.14 m.

Positions are normalised 0-1 with (0, 0) at the top left, so y runs down the pitch. That is still a
corner origin. There is no z. The halves are contiguous in the source, so an X frame goes between.

POSSESSION AND DEAD BALL, derived from events only (documentation/events-definitions.pdf):
  Owner: the team of the latest on-ball event, from its Start Frame. On-ball means SET PIECE, PASS,
    CARRY, SHOT, RECOVERY, BALL LOST or BALL OUT. CHALLENGE, FAULT RECEIVED and CARD never change
    the owner; a duel only changes hands when a RECOVERY or the next on-ball event says so.
  Ball goes dead ("the ball went out of bounds or the referee has stopped play"):
    - at the End Frame of BALL OUT, and of any on-ball event tagged OUT, GOAL, OFFSIDE, FORCED,
      VOLUNTARY or END HALF
    - at the Start Frame of FAULT RECEIVED, or of a CHALLENGE tagged FAULT. ADVANTAGE is not FAULT.
  Ball comes alive:
    - at the Start Frame of a SET PIECE or of an on-ball event tagged GOAL KICK, which is not a
      SET PIECE in Metrica's scheme. Those also prove it was dead, so with no stop recorded (a
      handball, a save turned behind) it counts as dead since the latest on-ball End Frame
    - otherwise at the first on-ball event other than a CARRY that starts at or after the stop:
      "RECOVERY ... brings the ball back into play via a dead-ball situation that is not a SET
      PIECE". Game 3 writes a short CARRY with the same Start Frame as nearly every BALL OUT, on
      the next row, so an event that began before play stopped must not count as restarting it
  Each half is dead before its first on-ball event and after its last, as in the challenge's truth.

These are Metrica's definitions and they are close to, but not, the challenge's: their possession
ends on a shot deflected off another player, and a duel is credited at the RECOVERY.
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np
import pandas as pd

from ..barath import _read
from ..match import A, AWAY, D, EXTERNAL_DIR, H, HOME, Match, X

DIR = os.path.join(EXTERNAL_DIR, "metrica")
REVISION = "e706dd506b360d69d9d123d5b8026e7294b13996"  # pinned: labels must not follow a new push
RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/" + REVISION + "/data/"
GAMES = (1, 2, 3)
CSV_FILES = ("RawEventsData.csv", "RawTrackingData_Home_Team.csv", "RawTrackingData_Away_Team.csv")
EPTS_FILES = ("events.json", "metadata.xml", "tracking.txt")
FPS = 25
PITCH = (105.0, 68.0)

ON_BALL = {"SET PIECE", "PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"}
STOPS_PLAY = {"OUT", "GOAL", "OFFSIDE", "FORCED", "VOLUNTARY", "END HALF"}

# stop is the event's last frame: End Frame, or Start Frame where End Frame is earlier (a SET PIECE
# can have End Frame 0).
Event = namedtuple("Event", "period start stop team kind tags")


def _stem(game, folder=None):
    return os.path.join(folder or DIR, "Sample_Game_" + str(game) + "_")


def download():
    os.makedirs(DIR, exist_ok=True)
    for game in GAMES:
        for name in EPTS_FILES if game == 3 else CSV_FILES:
            url = RAW + "Sample_Game_" + str(game) + "/Sample_Game_" + str(game) + "_" + name
            urllib.request.urlretrieve(url, _stem(game) + name)
        print("ok", game)


def _dead_spells(events, first, last):
    """[(start, stop)) frames of dead ball in one half, under the rules in the module docstring."""
    spells, dead_from, previous = [], None, first
    for e in events:
        on_ball = e.kind in ON_BALL
        restart = e.kind == "SET PIECE" or "GOAL KICK" in e.tags
        resumes = on_ball and e.kind != "CARRY" and dead_from is not None and e.start >= dead_from
        if restart or resumes:
            spells.append((previous if dead_from is None else dead_from, e.start))
            dead_from = None
        if on_ball:
            previous = max(previous, e.stop)
        if dead_from is None:
            if e.kind == "BALL OUT" or (on_ball and e.tags & STOPS_PLAY):
                dead_from = e.stop
            elif e.kind == "FAULT RECEIVED" or (e.kind == "CHALLENGE" and "FAULT" in e.tags):
                dead_from = e.start
    spells.append((previous if dead_from is None else dead_from, last + 1))
    return spells


def _half_labels(events, first, last):
    """int8 labels for source frames first..last from the events of that half, in order."""
    events = sorted(events, key=lambda e: e.start)
    if events and (events[0].start < first or max(e.stop for e in events) > last):
        raise ValueError("an event falls outside its half")
    labels = np.full(last + 1 - first, D, dtype=np.int8)
    owners = [e for e in events if e.kind in ON_BALL]
    for e, following in zip(owners, owners[1:] + [None]):
        labels[e.start - first:(following.start if following else e.stop) - first] = e.team
    for start, stop in _dead_spells(events, first, last):
        labels[start - first:stop - first] = D
    return labels


def _assemble(game, period, frames, ball, players, team, events):
    """Label each half, then put the X separator between the halves."""
    labels = []
    for h in np.unique(period):
        rows = frames[period == h]
        labels.append(_half_labels([e for e in events if e.period == h], rows[0], rows[-1]))
    labels = np.concatenate(labels)
    cut = np.flatnonzero(np.diff(period)) + 1
    ball = np.column_stack([ball * PITCH, np.full(len(ball), np.nan)]).astype(np.float32)
    return Match(name="Metrica-" + str(game), fps=FPS, pitch=PITCH,
                 half=np.insert(period, cut, 0).astype(np.int8),
                 ball=np.insert(ball, cut, np.nan, axis=0),
                 players=np.insert((players * PITCH).astype(np.float32), cut, np.nan, axis=0),
                 team=team, labels=np.insert(labels, cut, X),
                 source_frame=np.insert(frames, cut, -1))


def _csv_events(path):
    code = {"Home": H, "Away": A}
    with open(path, newline="") as handle:
        return [Event(int(r["Period"]), int(r["Start Frame"]),
                      max(int(r["Start Frame"]), int(r["End Frame"])), code[r["Team"]], r["Type"],
                      set(r["Subtype"].split("-"))) for r in csv.DictReader(handle)]


def _parse_csv(game, folder):
    """Games 1 and 2. Table columns: Period, Frame, Time, x, y per player, ball x, ball y."""
    stem = _stem(game, folder)
    home = pd.read_csv(stem + CSV_FILES[1], skiprows=2).to_numpy(dtype=np.float64)
    away = pd.read_csv(stem + CSV_FILES[2], skiprows=2).to_numpy(dtype=np.float64)
    if not np.array_equal(home[:, :2], away[:, :2]):
        raise ValueError("game " + str(game) + ": home and away frames differ")
    players = np.concatenate([home[:, 3:-2], away[:, 3:-2]], axis=1).reshape(len(home), -1, 2)
    n_home = (home.shape[1] - 5) // 2
    team = np.array([HOME] * n_home + [AWAY] * (players.shape[1] - n_home), dtype=np.int8)
    return _assemble(game, home[:, 0].astype(np.int8), home[:, 1].astype(np.int64), home[:, -2:],
                     players, team, _csv_events(stem + CSV_FILES[0]))


def _json_events(path, home_id):
    with open(path) as handle:
        rows = json.load(handle)["data"]
    events = []
    for r in rows:
        subtypes = r["subtypes"] or []
        subtypes = [subtypes] if isinstance(subtypes, dict) else subtypes
        start = r["start"]["frame"]
        events.append(Event(r["period"], start, max(start, r["end"]["frame"]),
                            H if r["team"]["id"] == home_id else A, r["type"]["name"],
                            {s["name"] for s in subtypes}))
    return events


def _parse_epts(game, folder):
    """Game 3. Each column's player comes from the DataFormatSpecification of its frame range."""
    stem = _stem(game, folder)
    root = ET.parse(stem + "metadata.xml").getroot()
    values = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    n_frames, second = int(values["second_half_end"]), int(values["second_half_start"])
    period = np.where(np.arange(1, n_frames + 1) < second, 1, 2).astype(np.int8)

    person = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    side_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    specs = [(int(s.get("startFrame")), int(s.get("endFrame")),
              [person[c.get("playerChannelId")] for c in s.iter("PlayerChannelRef")][::2])
             for s in root.iter("DataFormatSpecification")]
    used = {p for _, _, columns in specs for p in columns}
    people = [p for p in side_of if p in used]
    slot = {p: k for k, p in enumerate(people)}

    home_id = next(root.iter("Team")).get("id")
    team = np.array([HOME if side_of[p] == home_id else AWAY for p in people], dtype=np.int8)
    raw, ball = _read(stem + "tracking.txt", n_frames, len(specs[0][2]))
    players = np.full((n_frames, len(people), 2), np.nan)
    for start, stop, columns in specs:
        players[start - 1:stop, [slot[p] for p in columns]] = raw[start - 1:stop]
    return _assemble(game, period, np.arange(1, n_frames + 1, dtype=np.int64), ball, players, team,
                     _json_events(stem + "events.json", home_id))


def parse(game, folder=None):
    """One game as a Match, labelled from events as the module docstring says."""
    return (_parse_epts if game == 3 else _parse_csv)(game, folder)


def cache_path(game):
    return os.path.join(DIR, "cache", "game" + str(game) + ".npz")


def load(game):
    """The cached Match, parsing it first if needed."""
    path = cache_path(game)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        parse(game).save(path)
    return Match.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("download", "cache"))
    parser.add_argument("--game", nargs="+", type=int, default=list(GAMES))
    args = parser.parse_args()
    if args.command == "download":
        download()
    else:
        for game in args.game:
            m = load(game)
            print(m.name, "frames", m.n_frames, "players", len(m.team), "pitch", m.pitch)
    sys.exit(0)
