import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src import barath
from src.match import Match
from tests.fake_atd import PITCH, random_tracking, records, write_atd, write_meta, write_raw

small = dict(seed=st.integers(0, 2**32 - 1), n_frames=st.integers(1, 30),
             n_slots=st.integers(1, 8))


def parse_lines(lines, team, n_frames):
    with tempfile.TemporaryDirectory() as folder:
        return barath.parse(write_meta(folder, team), write_raw(folder, lines), n_frames, PITCH, ())


def test_a_hand_written_record_lands_in_metres_on_the_row_before_its_frame_id():
    m = parse_lines(["1:0.5,0.25,7.0;NaN,NaN,NaN:0.5,0.75,1.0\n",
                     "2:NaN,NaN,NaN;1.0,0.0,0.0:NaN,NaN,NaN\n"], [0, 1], 2)
    assert m.players[0, 0].tolist() == [52.65, 17.0]
    assert m.players[1, 1].tolist() == [105.3, 0.0]
    assert np.isnan(m.players[0, 1]).all() and np.isnan(m.players[1, 0]).all()
    assert m.ball[0, :2].tolist() == [52.65, 51.0]
    assert np.isnan(m.ball[1]).all()


def test_groups_without_a_speed_field_also_parse():
    """Metrica's public EPTS sample writes x,y where the ATD writes x,y,speed."""
    m = parse_lines(["1:0.5,0.25:0.5,0.75\n"], [0], 1)
    assert m.players[0, 0].tolist() == [52.65, 17.0]
    assert m.ball[0, :2].tolist() == [52.65, 51.0]


@given(**small)
@settings(deadline=None, max_examples=60)
def test_positions_survive_exactly_for_any_tracking(seed, n_frames, n_slots):
    players, ball, team = random_tracking(seed, n_frames, n_slots)
    with tempfile.TemporaryDirectory() as folder:
        meta, raw = write_atd(folder, players, ball, team)
        m = barath.parse(meta, raw, n_frames, PITCH, ())
    np.testing.assert_array_equal(m.players, players * np.array(PITCH))
    np.testing.assert_array_equal(m.ball[:, :2], ball * np.array(PITCH))
    np.testing.assert_array_equal(m.team, team)


def test_ball_height_is_marked_unknown():
    m = parse_lines(["1:0.5,0.25,7.0:0.5,0.75,1.0\n"], [0], 1)
    assert np.isnan(m.ball[0, 2])


def test_match_survives_save_and_load_at_full_precision(tmp_path):
    players, ball, team = random_tracking(1, 5, 3)
    m = parse_lines(records(players, ball), team, 5)
    m.save(tmp_path / "m.npz")
    back = Match.load(tmp_path / "m.npz")
    np.testing.assert_array_equal(back.players, m.players)
    np.testing.assert_array_equal(back.ball, m.ball)


@pytest.mark.parametrize("edit", [
    lambda lines: lines[:-1],                     # a record missing from the end
    lambda lines: lines[:1] + lines[2:],          # a gap in the middle
    lambda lines: [lines[1], lines[0], lines[2]],  # out of order
    lambda lines: lines + ["4:" + lines[0].split(":", 1)[1]],  # one record too many
    lambda lines: [lines[0].replace(":", ";", 1)] + lines[1:],  # a section missing
    lambda lines: [lines[0].rstrip() + ":0.1,0.2,0.0\n"] + lines[1:],  # a section too many
], ids=["short", "gap", "order", "long", "two-sections", "four-sections"])
def test_a_file_that_is_not_one_record_per_frame_is_refused(edit):
    players, ball, team = random_tracking(2, 3, 2, missing=0)
    with pytest.raises(ValueError):
        parse_lines(edit(records(players, ball)), team, 3)


def test_a_record_with_the_wrong_number_of_slots_is_refused():
    players, ball, team = random_tracking(3, 2, 2, missing=0)
    with pytest.raises(ValueError):
        parse_lines(records(players, ball), np.append(team, 0), 2)


def test_a_trailing_blank_line_is_harmless():
    players, ball, team = random_tracking(4, 2, 2)
    m = parse_lines(records(players, ball) + ["\n"], team, 2)
    assert m.n_frames == 2


def test_half_marks_the_published_graded_blocks_to_the_frame():
    half = barath.half_array(199319, [(1, 11185, 90651), (2, 110808, 190197)])
    assert [half[f] for f in (11184, 11185, 90651, 90652)] == [0, 1, 1, 0]
    assert [half[f] for f in (110807, 110808, 190197, 190198)] == [0, 2, 2, 0]
    assert int((half > 0).sum()) == 158857


def test_blocks_from_games_are_the_published_ones():
    half = barath.half_array(199319, barath.blocks())
    assert int((half == 1).sum()) == 79467 and int((half == 2).sum()) == 79390
