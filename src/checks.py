"""Automatic checks on the provided data, each one a yes or no.

Run these only through src.blind. They exist to prove the pipeline is correct, and nothing here
may ever feed a parameter: the organisers forbid training or tuning on the provided match.
"""

import os
import sys

import numpy as np

from . import barath, submission
from .baseline import nearest_team, smooth
from .match import D, ROOT, X, pm

sys.path.insert(0, ROOT)
from games import asset_paths, require  # noqa: E402
from starter import baseline as organisers  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

RESULTS = os.path.join(ROOT, "evaluation", "results")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

# The public-window board printed in README.md: (file, macroF1, spellF1, possessions).
PUBLISHED_BOARD = (
    ("baseline_smooth125", 0.5812, 0.4242, 16),
    ("baseline_smooth50", 0.6207, 0.3256, 26),
    ("baseline_smooth25", 0.6200, 0.2868, 36),
    ("baseline_raw", 0.5972, 0.0378, 131),
)


def _labels(path):
    return pm.label_codes(pm.read_label_csv(path))


def public_scores():
    """The official scorer, under our pinned packages, reproduces the README's board."""
    gt = _labels(GT_PUBLIC)
    for name, macro, spell, possessions in PUBLISHED_BOARD:
        result = pm.evaluate(gt, _labels(os.path.join(RESULTS, name + ".csv")))
        assert round(result["macro_f1"], 4) == macro, name
        assert round(result["spell_f1_mean"], 4) == spell, name
        assert result["n_pred_spells"] == possessions, name


def baselines(match=None, folder=RESULTS):
    """Our loader and baseline reproduce all four shipped files frame for frame."""
    m = match if match is not None else barath.load()
    codes = nearest_team(m)
    for name, window in organisers.VARIANTS:
        shipped = _labels(os.path.join(str(folder), name + ".csv"))
        assert np.array_equal(smooth(codes, window), shipped), name


def teams():
    """Home in the tracking metadata is Barcelona, so H means what the challenge says."""
    meta = require(["atd_meta"])[0]
    assert "barcelona" in organisers.team_names(meta)[0].lower()


def time_base():
    """Smart Tagging kickoffs contain both real kickoffs."""
    assert load_smart_tagging.check_time_base(load_smart_tagging.load())["agrees"]


def assets():
    """Every tactical asset is present."""
    assert all(os.path.isfile(path) for path in asset_paths().values())


def safety_file():
    """The fallback submission writes, validates, and has D and X where the docs require."""
    m = barath.load()
    path = os.path.join(RESULTS, "safety_smooth50.csv")
    submission.write(path, submission.safety_net(m))
    frame = pm.read_label_csv(path)
    assert pm.validate_submission(frame, m.n_frames) is None
    codes = pm.label_codes(frame)
    assert (codes[110808:submission.SECOND_HALF_KICKOFF] == D).all()
    assert (codes[submission.SECOND_HALF_KICKOFF:190198] <= D).all()
    assert (codes[m.half == 0] == X).all()
    assert (codes[m.half > 0] != X).all()


CHECKS = {check.__name__: check
          for check in (assets, teams, time_base, public_scores, baselines, safety_file)}
