"""Regenerate the submission from the saved model and the provided data.

    python -m src.submit

Reads only the tactical tracking and Smart Tagging, with weights trained on other matches. Run
through src.blind so that nothing from the match reaches an assistant.
"""

import os
import sys

import numpy as np

from . import barath, features, submission, tags
from .decode import Costs
from .match import D, ROOT
from .model import Model

sys.path.insert(0, ROOT)
from starter import load_smart_tagging  # noqa: E402

MODEL = os.path.join(ROOT, "models", "v1")
OUTPUT = os.path.join(ROOT, "evaluation", "results", "827.csv")
DECODER = Costs(turnover=2000, stop=2000, restart_other=1600, restart_same=2400)
GAIN = 1
HOME_TAG, AWAY_TAG = "Barcelona", "Atletic Club"


def finish(codes, half, kickoff=submission.SECOND_HALF_KICKOFF):
    """The published fact that the second half's graded frames before kickoff are dead ball."""
    out = codes.copy()
    out[np.flatnonzero(half == 2)[0]:kickoff] = D
    return out


def labels(m, trained):
    index = features.build(m)[1]
    signs = tags.row_signs(load_smart_tagging.load(), index, max(1, m.fps // features.HZ),
                           HOME_TAG, AWAY_TAG)
    return finish(trained.predict(m, DECODER, GAIN, signs=signs, weight=tags.WEIGHT), m.half)


def main():
    submission.write(OUTPUT, labels(barath.load(), Model.load(MODEL)))


if __name__ == "__main__":
    main()
