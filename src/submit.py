"""Regenerate the submission from the saved model and the provided data.

    python -m src.submit

Reads only the tactical tracking and Smart Tagging, with weights trained on other matches. Run
through src.blind so that nothing from the match reaches an assistant.
"""

import os
import sys

import numpy as np

from . import barath, features, stack, submission, tags
from .match import D, ROOT

sys.path.insert(0, ROOT)
from starter import load_smart_tagging  # noqa: E402

MODEL = os.path.join(ROOT, "models", "v2")
OUTPUT = os.path.join(ROOT, "evaluation", "results", "827.csv")
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
    return finish(trained.predict(m, stack.COSTS, stack.GAIN, signs=signs, weight=tags.WEIGHT),
                  m.half)


def main():
    submission.write(OUTPUT, labels(barath.load(), stack.Stack.load(MODEL)))


if __name__ == "__main__":
    main()
