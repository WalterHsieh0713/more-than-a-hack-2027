"""Tiny handwritten files in SkillCorner's real layout. No network, no real data."""
import json

import numpy as np
import pytest

from src.external import skillcorner
from src.match import A, AWAY, D, H, HOME, Match, X

HOME_ID, AWAY_ID = 1802, 866
ALICE, BOB = 50999, 69111
MATCH_ID = "1874553"

EMPTY_BALL = {"x": None, "y": None, "z": None, "is_detected": None}


def frame(n, period=None, ball=EMPTY_BALL, group=None, players=()):
    """One line of *_tracking_extrapolated.jsonl, with every key the provider writes."""
    corners = {k: None for k in ("x_top_left", "y_top_left", "x_bottom_left", "y_bottom_left",
                                 "x_bottom_right", "y_bottom_right", "x_top_right", "y_top_right")}
    return {"frame": n, "timestamp": None if period is None else "00:00:00.00", "period": period,
            "ball_data": ball, "possession": {"player_id": None, "group": group},
            "image_corners_projection": corners, "player_data": list(players)}


def player(pid, x, y, detected=True):
    return {"x": x, "y": y, "player_id": pid, "is_detected": detected}


def ball(x, y, z=0.2, detected=True):
    return {"x": x, "y": y, "z": z, "is_detected": detected}


SEEN = ball(0.0, 0.0)
PHASES_HEAD = "index,match_id,frame_start,frame_end,time_start,period,team_in_possession_id\n"


def phase_row(index, start, stop, team):
    return ",".join(map(str, (index, MATCH_ID, start, stop, "00:00.0", 1, team))) + "\n"


def write(folder, frames, phases=(), pitch=(105, 68)):
    """phases: [(frame_start, frame_end, team_id)]."""
    with open(folder / (MATCH_ID + "_tracking_extrapolated.jsonl"), "w") as handle:
        handle.writelines(json.dumps(f) + "\n" for f in frames)
    info = {"id": int(MATCH_ID), "pitch_length": pitch[0], "pitch_width": pitch[1],
            "home_team": {"id": HOME_ID}, "away_team": {"id": AWAY_ID},
            "players": [{"id": ALICE, "team_id": HOME_ID}, {"id": BOB, "team_id": AWAY_ID},
                        {"id": 7, "team_id": AWAY_ID}]}    # 7 stayed on the bench
    (folder / (MATCH_ID + "_match.json")).write_text(json.dumps(info))
    rows = "".join(phase_row(i, *row) for i, row in enumerate(phases))
    (folder / (MATCH_ID + "_phases_of_play.csv")).write_text(PHASES_HEAD + rows)
    return skillcorner.parse(MATCH_ID, str(folder))


def test_positions_move_from_pitch_centre_to_a_corner_in_metres(tmp_path):
    m = write(tmp_path, [frame(0, 1, ball(-52.5, 34.0, 1.5), players=[player(ALICE, 52.5, -34.0)])])
    assert (m.name, m.fps, m.pitch) == ("SkillCorner-" + MATCH_ID, 10, (105.0, 68.0))
    assert m.players[0, 0] == pytest.approx([105.0, 0.0])
    assert m.ball[0] == pytest.approx([0.0, 68.0, 1.5])


def test_each_match_uses_its_own_pitch(tmp_path):
    m = write(tmp_path, [frame(0, 1, ball(-53.0, 0.0))], pitch=(106, 68))
    assert m.pitch == (106.0, 68.0) and m.ball[0, 0] == pytest.approx(0.0)


def test_extrapolated_objects_are_nan_not_positions(tmp_path):
    m = write(tmp_path, [frame(0, 1, ball(1.0, 2.0, detected=False),
                               players=[player(ALICE, 1.0, 1.0, False), player(BOB, 2.0, 2.0)])])
    assert np.isnan(m.ball[0]).all()
    assert np.isnan(m.players[0, m.team == HOME]).all()
    assert m.players[0, m.team == AWAY][0] == pytest.approx([54.5, 36.0])


def test_a_slot_per_listed_player_with_the_right_team(tmp_path):
    m = write(tmp_path, [frame(0, 1, SEEN, players=[player(ALICE, 0.0, 0.0), player(BOB, 0.0, 0.0)])])
    assert sorted(m.team.tolist()) == [HOME, AWAY]         # the bench player gets no slot
    assert m.team.dtype == np.int8


def test_frames_without_a_period_are_separators(tmp_path):
    m = write(tmp_path, [frame(0), frame(1, 1, SEEN), frame(2), frame(3, 2, SEEN), frame(4)])
    assert m.half.tolist() == [0, 1, 0, 2, 0]
    assert m.source_frame.tolist() == [0, 1, 2, 3, 4]
    assert m.labels[[0, 2, 4]].tolist() == [X, X, X]


def test_a_phase_gives_its_team_from_start_up_to_but_not_including_end(tmp_path):
    frames = [frame(n, 1, SEEN) for n in range(6)]
    m = write(tmp_path, frames, phases=[(1, 3, HOME_ID), (3, 5, AWAY_ID)])
    assert m.labels.tolist() == [D, H, H, A, A, D]


def test_the_per_frame_group_does_not_override_the_phase_team(tmp_path):
    frames = [frame(0, 1, SEEN, group="away team"), frame(1, 1, SEEN, group=None)]
    assert write(tmp_path, frames, phases=[(0, 2, HOME_ID)]).labels.tolist() == [H, H]


def test_outside_a_phase_something_seen_is_dead_and_nothing_seen_is_unknown(tmp_path):
    frames = [frame(0, 1, SEEN),
              frame(1, 1, ball(1.0, 1.0), players=[player(ALICE, 0.0, 0.0, False)]),  # ball seen
              frame(2, 1, ball(1.0, 1.0, detected=False), players=[player(ALICE, 0.0, 0.0)]),  # player
              frame(3, 1, ball(1.0, 1.0, detected=False), players=[player(ALICE, 0.0, 0.0, False)]),
              frame(4, 1)]                                                       # replay: no data
    m = write(tmp_path, frames)
    assert m.labels.tolist() == [D, D, D, X, X]
    assert m.half.tolist() == [1] * 5                      # still match time, for the noise stats


def test_a_phase_is_believed_through_a_frame_nothing_was_seen_in(tmp_path):
    m = write(tmp_path, [frame(0, 1, SEEN), frame(1, 1), frame(2, 1, SEEN)], phases=[(0, 3, AWAY_ID)])
    assert m.labels.tolist() == [A, A, A]


def test_a_phase_cannot_label_a_frame_outside_the_halves(tmp_path):
    m = write(tmp_path, [frame(0, 1, SEEN), frame(1)], phases=[(0, 2, HOME_ID)])
    assert m.labels.tolist() == [H, X]


def test_a_skipped_frame_is_refused(tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        write(tmp_path, [frame(0, 1, SEEN), frame(2, 1, SEEN)])


def test_halves_with_no_separator_are_refused(tmp_path):
    with pytest.raises(ValueError, match="separator"):
        write(tmp_path, [frame(0, 1, SEEN), frame(1, 2, SEEN)])


def test_a_phase_of_an_unknown_team_is_refused(tmp_path):
    with pytest.raises(KeyError):
        write(tmp_path, [frame(0, 1, SEEN)], phases=[(0, 1, 999)])


def test_a_player_missing_from_the_lineup_is_refused(tmp_path):
    with pytest.raises(KeyError):
        write(tmp_path, [frame(0, 1, SEEN, players=[player(123, 0.0, 0.0)])])


def test_the_match_survives_the_cache(tmp_path):
    m = write(tmp_path, [frame(0), frame(1, 1, ball(0.0, 0.0), players=[player(ALICE, 1.0, 1.0)])],
              phases=[(1, 2, HOME_ID)])
    path = str(tmp_path / "m.npz")
    m.save(path)
    back = Match.load(path)
    assert back.labels.tolist() == m.labels.tolist() and back.fps == 10
    assert np.array_equal(back.players, m.players, equal_nan=True)
