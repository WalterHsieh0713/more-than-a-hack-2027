"""Tiny handwritten files in Metrica's real layouts. No network, no real data."""
import json

import numpy as np
import pytest

from src.external import metrica
from src.match import A, AWAY, D, H, HOME, X

HEAD = "Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y\n"


def event(team, kind, subtype, start, end=None, period=1):
    end = start if end is None else end
    return ",".join(map(str, (team, kind, subtype, period, start, start / 25, end, end / 25,
                              "Player1", "", "NaN", "NaN", "NaN", "NaN"))) + "\n"


def side(name, players, rows):
    """A RawTrackingData file. rows: [(period, frame, [x, y, ...] per player, ball_x, ball_y)]."""
    teams = ",,," + (name + ",,") * players + ",\n"
    numbers = ",,," + "".join(str(k) + ",," for k in range(1, players + 1)) + ",\n"
    names = "Period,Frame,Time [s]," + "".join("Player" + str(k) + ",," for k in range(1, players + 1)) + "Ball,\n"
    body = "".join(",".join(map(str, (p, f, f / 25, *xy, bx, by))) + "\n" for p, f, xy, bx, by in rows)
    return teams + numbers + names + body


def write_csv(folder, events, home_rows, away_rows, home_players=1, away_players=1):
    stem = str(folder / "Sample_Game_1_")
    with open(stem + "RawEventsData.csv", "w") as handle:
        handle.write(HEAD + "".join(events))
    with open(stem + "RawTrackingData_Home_Team.csv", "w") as handle:
        handle.write(side("Home", home_players, home_rows))
    with open(stem + "RawTrackingData_Away_Team.csv", "w") as handle:
        handle.write(side("Away", away_players, away_rows))
    return metrica.parse(1, str(folder))


def play(folder, events, frames=10, second=0):
    """Labels for events on a toy match of `frames` first-half and `second` second-half frames."""
    rows = [(1 if f <= frames else 2, f, [0.5, 0.5], 0.5, 0.5) for f in range(1, frames + second + 1)]
    return write_csv(folder, events, rows, rows)


def test_csv_positions_land_in_metres_with_nan_for_the_unseen(tmp_path):
    home = [(1, 1, [0.5, 0.25, "NaN", "NaN"], 0.1, 0.5), (1, 2, [1.0, 0.0, 0.2, 0.5], "NaN", "NaN")]
    away = [(1, 1, [0.5, 0.75], 0.1, 0.5), (1, 2, [0.0, 1.0], "NaN", "NaN")]
    m = write_csv(tmp_path, [], home, away, home_players=2)
    assert (m.name, m.fps, m.pitch) == ("Metrica-1", 25, (105.0, 68.0))
    assert m.team.tolist() == [HOME, HOME, AWAY]
    assert m.players[0, 0] == pytest.approx([52.5, 17.0])
    assert np.isnan(m.players[0, 1]).all()
    assert m.players[1, 2] == pytest.approx([0.0, 68.0])
    assert m.ball[0, :2] == pytest.approx([10.5, 34.0])
    assert np.isnan(m.ball[0, 2]) and np.isnan(m.ball[1]).all()


def test_an_x_frame_separates_halves_that_are_contiguous_in_the_source(tmp_path):
    m = play(tmp_path, [], frames=2, second=2)
    assert m.half.tolist() == [1, 1, 0, 2, 2]
    assert m.source_frame.tolist() == [1, 2, -1, 3, 4]
    assert m.labels[2] == X and np.isnan(m.ball[2]).all() and np.isnan(m.players[2]).all()


def test_the_owner_is_the_team_of_the_latest_on_ball_event(tmp_path):
    """The last frame is dead in every sequence that ends on frame 10: play is over at End Frame."""
    m = play(tmp_path, [event("Home", "SET PIECE", "KICK OFF", 1), event("Home", "PASS", "", 1, 3),
                        event("Home", "BALL LOST", "INTERCEPTION", 4, 5),
                        event("Away", "RECOVERY", "INTERCEPTION", 6), event("Away", "CARRY", "", 6, 10)])
    assert m.labels.tolist() == [H] * 5 + [A] * 4 + [D]


def test_a_challenge_never_changes_the_owner(tmp_path):
    m = play(tmp_path, [event("Home", "PASS", "", 1, 3), event("Away", "CHALLENGE", "GROUND-WON", 2),
                        event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H] * 5 + [A] * 4 + [D]


def test_the_ball_is_dead_from_ball_out_end_frame_until_the_set_piece(tmp_path):
    m = play(tmp_path, [event("Home", "PASS", "", 1, 2), event("Home", "BALL OUT", "", 2, 4),
                        event("Away", "SET PIECE", "THROW IN", 7), event("Away", "PASS", "", 7, 10)])
    assert m.labels.tolist() == [H, H, H, D, D, D, A, A, A, D]


@pytest.mark.parametrize("subtype", ["OFF TARGET-OUT", "ON TARGET-GOAL", "HEAD-WOODWORK-OUT"])
def test_a_shot_that_ends_play_goes_dead_at_its_end_frame(tmp_path, subtype):
    m = play(tmp_path, [event("Home", "SHOT", subtype, 1, 3), event("Away", "SET PIECE", "KICK OFF", 6),
                        event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


def test_a_saved_shot_leaves_the_ball_alive(tmp_path):
    m = play(tmp_path, [event("Home", "SHOT", "ON TARGET-SAVED", 1, 3),
                        event("Away", "RECOVERY", "SAVED", 4), event("Away", "PASS", "", 4, 10)])
    assert m.labels.tolist() == [H, H, H, A, A, A, A, A, A, D]


def test_a_goal_kick_is_a_restart_though_it_is_not_a_set_piece(tmp_path):
    m = play(tmp_path, [event("Home", "BALL OUT", "", 1, 3), event("Away", "PASS", "GOAL KICK", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


def test_a_restart_without_a_recorded_stop_is_dead_since_the_last_event_ended(tmp_path):
    m = play(tmp_path, [event("Home", "PASS", "", 1, 3), event("Away", "SET PIECE", "FREE KICK", 6),
                        event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


def test_a_recovery_restarts_play_when_there_is_no_set_piece(tmp_path):
    m = play(tmp_path, [event("Home", "BALL OUT", "", 1, 3), event("Away", "RECOVERY", "", 6),
                        event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


@pytest.mark.parametrize("kind,subtype", [("FAULT RECEIVED", ""), ("CHALLENGE", "TACKLE-FAULT-WON")])
def test_a_foul_stops_play_at_its_start_frame(tmp_path, kind, subtype):
    m = play(tmp_path, [event("Home", "PASS", "", 1, 3), event("Away", kind, subtype, 4),
                        event("Home", "SET PIECE", "FREE KICK", 7), event("Home", "PASS", "", 7, 10)])
    assert m.labels.tolist() == [H, H, H, D, D, D, H, H, H, D]


def test_advantage_lets_play_run_on(tmp_path):
    m = play(tmp_path, [event("Home", "PASS", "", 1, 3), event("Away", "CHALLENGE", "TACKLE-ADVANTAGE-LOST", 4),
                        event("Home", "PASS", "", 5, 10)])
    assert m.labels.tolist() == [H] * 9 + [D]


@pytest.mark.parametrize("subtype", ["FORCED", "OFFSIDE", "HEAD-FORCED"])
def test_the_referee_stopping_play_is_dead_from_the_end_frame(tmp_path, subtype):
    m = play(tmp_path, [event("Home", "BALL LOST", subtype, 1, 3), event("Away", "SET PIECE", "FREE KICK", 6),
                        event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


def test_a_card_during_a_stoppage_does_not_restart_play(tmp_path):
    m = play(tmp_path, [event("Home", "BALL OUT", "", 1, 3), event("Away", "CARD", "YELLOW", 4),
                        event("Away", "SET PIECE", "THROW IN", 6), event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


def test_a_carry_written_alongside_the_ball_out_does_not_cancel_the_stop(tmp_path):
    """Game 3 writes these: same Start Frame as the BALL OUT, on the next row, one frame long."""
    m = play(tmp_path, [event("Home", "BALL OUT", "", 1, 3), event("Home", "CARRY", "", 1, 2),
                        event("Away", "SET PIECE", "THROW IN", 6), event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


def test_a_carry_during_a_stoppage_is_not_a_restart(tmp_path):
    m = play(tmp_path, [event("Home", "BALL OUT", "", 1, 3), event("Home", "CARRY", "", 4, 5),
                        event("Away", "SET PIECE", "THROW IN", 6), event("Away", "PASS", "", 6, 10)])
    assert m.labels.tolist() == [H, H, D, D, D, A, A, A, A, D]


def test_an_event_that_began_before_play_stopped_cannot_restart_it(tmp_path):
    m = play(tmp_path, [event("Home", "BALL OUT", "", 1, 4), event("Away", "RECOVERY", "", 3),
                        event("Away", "SET PIECE", "THROW IN", 7), event("Away", "PASS", "", 7, 10)])
    assert m.labels.tolist() == [H, H, A, D, D, D, A, A, A, D]


def test_a_restart_with_no_stop_counts_from_the_latest_end_not_the_last_row(tmp_path):
    m = play(tmp_path, [event("Home", "PASS", "", 1, 4), event("Home", "CARRY", "", 1, 2),
                        event("Away", "SET PIECE", "FREE KICK", 7), event("Away", "PASS", "", 7, 10)])
    assert m.labels.tolist() == [H, H, H, D, D, D, A, A, A, D]


def test_a_half_is_dead_before_its_first_event_and_after_its_last(tmp_path):
    m = play(tmp_path, [event("Home", "SET PIECE", "KICK OFF", 3), event("Home", "PASS", "", 3, 5)])
    assert m.labels.tolist() == [D, D, H, H, D, D, D, D, D, D]


def test_a_set_piece_end_frame_of_zero_is_ignored(tmp_path):
    m = play(tmp_path, [event("Away", "SET PIECE", "KICK OFF", 3, 0), event("Away", "PASS", "", 3, 10)])
    assert m.labels.tolist() == [D, D] + [A] * 7 + [D]


def test_each_half_uses_only_its_own_events(tmp_path):
    m = play(tmp_path, [event("Home", "PASS", "", 1, 2), event("Away", "PASS", "", 4, 5, period=2)],
             frames=3, second=3)
    assert m.labels.tolist() == [H, D, D, X, A, D, D]


def test_an_event_outside_its_half_is_refused(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        play(tmp_path, [event("Home", "PASS", "", 12)])


def test_tracking_files_that_disagree_on_frames_are_refused(tmp_path):
    home = [(1, 1, [0.5, 0.5], 0.5, 0.5)]
    away = [(1, 2, [0.5, 0.5], 0.5, 0.5)]
    with pytest.raises(ValueError, match="frames"):
        write_csv(tmp_path, [], home, away)


# Game 3: EPTS tracking, FIFA metadata, JSON events

def channel(name):
    return ('<SplitRegister separator=","><PlayerChannelRef playerChannelId="' + name + '_x"/>'
            '<PlayerChannelRef playerChannelId="' + name + '_y"/></SplitRegister>')


def spec(start, stop, names):
    return ('<DataFormatSpecification separator=":" startFrame="' + str(start) + '" endFrame="' + str(stop) + '">'
            '<StringRegister name="frameCount"/><SplitRegister separator=";">'
            + "".join(channel(n) for n in names) + '</SplitRegister>'
            '<SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/>'
            '</SplitRegister></DataFormatSpecification>')


def provider(name, value):
    return "<ProviderParameter><Name>" + name + "</Name><Value>" + str(value) + "</Value></ProviderParameter>"


def player_channels(index, person):
    return ('<PlayerChannel id="player' + str(index) + '_x" playerId="' + person + '"/>'
            '<PlayerChannel id="player' + str(index) + '_y" playerId="' + person + '"/>')


METADATA = ('<?xml version="1.0" encoding="utf-8"?><main><Metadata><GlobalConfig><FrameRate>25</FrameRate>'
            '<ProviderGlobalParameters>' + provider("first_half_end", 2) + provider("second_half_start", 3)
            + provider("second_half_end", 4) + '</ProviderGlobalParameters></GlobalConfig>'
            '<Teams><Team id="FIFATMA"><Name>Team A</Name></Team><Team id="FIFATMB"><Name>Team B</Name></Team></Teams>'
            '<Players><Player id="P1" teamId="FIFATMA"/><Player id="P2" teamId="FIFATMA"/>'
            '<Player id="P3" teamId="FIFATMB"/></Players>'
            '<PlayerChannels>' + player_channels(1, "P1") + player_channels(2, "P2") + player_channels(3, "P3")
            + '</PlayerChannels></Metadata><DataFormatSpecifications>'
            + spec(1, 2, ["player1", "player3"]) + spec(3, 4, ["player2", "player3"])  # P2 replaces P1
            + '</DataFormatSpecifications></main>')

TRACKING = ("1:0.5,0.25;0.1,0.1:0.5,0.75\n"
            "2:0.5,0.25;0.1,0.1:NaN,NaN\n"
            "3:1.0,0.0;0.2,0.2:NaN,NaN\n"
            "4:1.0,0.0;0.2,0.2:0.5,0.5\n")


def json_event(team, kind, subtypes, start, end, period):
    return {"index": 1, "team": {"name": "x", "id": team}, "type": {"name": kind, "id": 1},
            "subtypes": subtypes, "start": {"frame": start, "time": 0.0, "x": None, "y": None},
            "end": {"frame": end, "time": 0.0, "x": None, "y": None}, "period": period,
            "from": {"name": "Player 1", "id": "P1"}, "to": None}


def write_epts(folder, events, tracking=TRACKING):
    stem = str(folder / "Sample_Game_3_")
    with open(stem + "metadata.xml", "w") as handle:
        handle.write(METADATA)
    with open(stem + "tracking.txt", "w") as handle:
        handle.write(tracking)
    with open(stem + "events.json", "w") as handle:
        json.dump({"data": events, "metadata": []}, handle)
    return metrica.parse(3, str(folder))


def test_epts_columns_follow_the_spec_of_their_frame_range_so_a_substitute_gets_their_own_slot(tmp_path):
    m = write_epts(tmp_path, [])
    assert m.team.tolist() == [HOME, HOME, AWAY]           # P1, P2, P3
    assert m.players[0, 0] == pytest.approx([52.5, 17.0])
    assert np.isnan(m.players[0, 1]).all()                 # P2 is still on the bench
    assert np.isnan(m.players[3, 0]).all()                 # P1 has gone off, after the separator
    assert m.players[3, 1] == pytest.approx([105.0, 0.0])
    assert m.players[3, 2] == pytest.approx([21.0, 13.6])  # P3 stays in slot 2 throughout


def test_epts_halves_come_from_the_provider_parameters(tmp_path):
    m = write_epts(tmp_path, [])
    assert (m.name, m.fps, m.pitch) == ("Metrica-3", 25, (105.0, 68.0))
    assert m.half.tolist() == [1, 1, 0, 2, 2]
    assert m.source_frame.tolist() == [1, 2, -1, 3, 4]
    assert m.ball[0, :2] == pytest.approx([52.5, 51.0])
    assert np.isnan(m.ball[1]).all()


def test_json_events_take_their_team_from_the_id_and_all_three_subtype_shapes(tmp_path):
    m = write_epts(tmp_path, [
        json_event("FIFATMB", "PASS", None, 1, 1, 1),
        json_event("FIFATMB", "BALL OUT", {"name": "CLEARANCE", "id": 1}, 2, 2, 1),
        json_event("FIFATMA", "SET PIECE", {"name": "THROW IN", "id": 1}, 3, 3, 2),
        json_event("FIFATMA", "SHOT", [{"name": "OFF TARGET", "id": 1}, {"name": "OUT", "id": 2}], 3, 4, 2)])
    assert m.labels.tolist() == [A, D, X, H, D]


def test_epts_refuses_a_line_with_the_wrong_number_of_slots(tmp_path):
    with pytest.raises(ValueError, match="slots"):
        write_epts(tmp_path, [], tracking="1:0.5,0.25:0.5,0.75\n2:0.5,0.25:0.5,0.75\n"
                                          "3:0.5,0.25:0.5,0.75\n4:0.5,0.25:0.5,0.75\n")
