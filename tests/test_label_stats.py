import os
import subprocess
import sys

import numpy as np
import pytest

from src.label_stats import label_stats, possession_rows, report
from src.match import A, D, H, ROOT, X

FPS = 25


def run(*pieces):
    return np.concatenate([np.full(n, k, dtype=np.int8) for k, n in pieces])


# 2 s H, 0.4 s A, 4 s H, 3 s stop, 1 s H (same team), 1 s stop, 10 s A (other team): 21.4 s.
SAMPLE = run((H, 50), (A, 10), (H, 100), (D, 75), (H, 25), (D, 25), (A, 250))


def test_counts_and_durations_by_hand():
    s = label_stats(SAMPLE, FPS)
    assert s["possessions"] == 5 and s["stoppages"] == 2
    assert s["mean_s"] == pytest.approx(3.48) and s["median_s"] == pytest.approx(2.0)
    assert s["poss_per_min"] == pytest.approx(5 / (21.4 / 60))


@pytest.mark.parametrize("key,want", [("under_1s", 20.0), ("under_2s", 40.0),
                                      ("under_3s", 60.0), ("under_5s", 80.0)])
def test_shares_of_short_possessions_are_strictly_under(key, want):
    assert label_stats(SAMPLE, FPS)[key] == pytest.approx(want)


def test_boundaries_are_sorted_by_how_possession_ended():
    s = label_stats(SAMPLE, FPS)
    assert s["turnovers"] == 2
    assert s["same_team_restarts"] == 1 and s["other_team_restarts"] == 1


def test_a_restart_is_counted_by_who_has_the_ball_after_it():
    same = label_stats(run((H, 25), (D, 25), (H, 25)), FPS)
    other = label_stats(run((H, 25), (D, 25), (A, 25)), FPS)
    assert (same["same_team_restarts"], same["other_team_restarts"]) == (1, 0)
    assert (other["same_team_restarts"], other["other_team_restarts"]) == (0, 1)


def test_a_possession_between_two_stoppages_is_not_a_sandwich():
    assert label_stats(run((D, 25), (H, 10), (D, 25)), FPS)["sandwiches_under_2s"] == 0


def test_a_short_spell_between_two_of_the_other_team_is_a_sandwich():
    assert label_stats(SAMPLE, FPS)["sandwiches_under_2s"] == 1
    assert label_stats(run((H, 50), (A, 50), (H, 50)), FPS)["sandwiches_under_2s"] == 0


def test_nothing_is_counted_across_ungraded_frames():
    s = label_stats(run((H, 50), (X, 10), (A, 50)), FPS)
    assert s["possessions"] == 2 and s["turnovers"] == 0
    s = label_stats(run((H, 50), (D, 25), (X, 10), (A, 50)), FPS)
    assert s["other_team_restarts"] == 0


def test_rows_give_a_person_something_to_seek_to_in_the_video():
    codes = run((X, 100), (H, 50), (D, 25), (A, 30))
    assert possession_rows(codes, FPS) == [(100, "H", 2.0), (175, "A", 1.2)]


OTHERS = {"a": run((H, 500)), "b": run((H, 100), (A, 100))}


def test_report_sets_the_public_window_beside_the_range_of_the_open_games():
    lines = report(SAMPLE, OTHERS, FPS).splitlines()
    assert lines[0].split() == ["public", "idsse_low", "idsse_high"]
    possessions = next(line for line in lines if line.startswith("possessions"))
    assert possessions.split() == ["possessions", "5.0", "1.0", "2.0"]


def test_report_fits_a_normal_terminal_panel():
    assert max(len(line) for line in report(SAMPLE, OTHERS, FPS).splitlines()) <= 72


def test_report_ends_by_listing_every_public_possession():
    assert report(SAMPLE, OTHERS, FPS).splitlines()[-1].split() == ["285", "A", "10.00"]


def test_report_still_works_before_the_open_games_are_downloaded():
    lines = report(SAMPLE, {}, FPS).splitlines()
    assert lines[0].split() == ["public"]


def test_the_reader_refuses_to_print_unless_a_person_is_at_a_terminal():
    done = subprocess.run([sys.executable, "-m", "src.public_window"], capture_output=True,
                          text=True, cwd=ROOT, env=os.environ)
    # Reduced to plain values first: a failing assert on `done` would print the captured output,
    # and that output could be match labels.
    code, leaked, told = done.returncode, "turnovers" in done.stdout, "terminal" in done.stdout
    del done
    assert (code, leaked, told) == (2, False, True)
