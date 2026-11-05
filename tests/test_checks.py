"""The comparison behind the baselines check, proven able to fail, on invented data only."""

import numpy as np
import pytest

from src import checks
from src.baseline import nearest_team, smooth
from src.match import AWAY, HOME, Match
from starter import baseline as organisers
from tests.fake_atd import PITCH, random_tracking


@pytest.fixture
def setup(tmp_path):
    players, ball, team = random_tracking(5, 400, 6)
    m = Match(name="fake", fps=25, pitch=PITCH, half=np.ones(400, dtype=np.int8),
              ball=np.column_stack([ball * PITCH, np.full(400, np.nan)]),
              players=players * np.array(PITCH), team=team)
    codes = nearest_team(m)
    for name, window in organisers.VARIANTS:
        organisers.write_submission(str(tmp_path / (name + ".csv")),
                                    organisers.smooth(codes, window))
    return m, tmp_path


def test_matching_files_pass(setup):
    m, folder = setup
    checks.baselines(m, folder)


@pytest.mark.parametrize("variant", [name for name, _ in organisers.VARIANTS])
def test_one_changed_frame_in_any_variant_fails(setup, variant):
    m, folder = setup
    path = folder / (variant + ".csv")
    lines = path.read_text().splitlines()
    frame, label = lines[100].split(",")
    lines[100] = frame + "," + ("H" if label != "H" else "A")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError):
        checks.baselines(m, folder)


def test_a_one_frame_shift_fails(setup):
    m, folder = setup
    shifted = Match(name="late", fps=25, pitch=PITCH, half=m.half,
                    ball=np.roll(m.ball, 1, axis=0), players=np.roll(m.players, 1, axis=0),
                    team=m.team)
    with pytest.raises(AssertionError):
        checks.baselines(shifted, folder)


def test_swapped_teams_fail(setup):
    m, folder = setup
    swapped = Match(name="swap", fps=25, pitch=PITCH, half=m.half, ball=m.ball,
                    players=m.players, team=AWAY - m.team + HOME)
    with pytest.raises(AssertionError):
        checks.baselines(swapped, folder)
