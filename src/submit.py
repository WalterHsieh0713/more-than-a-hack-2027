"""Regenerate the submission from the saved model and the provided data.

    python -m src.submit

Reads only the tactical tracking, with weights trained on other matches. The model runs on the
match and on its team-swapped copy, and the two answers are averaged. Smart Tagging is not
used: on simulated tags it helped the finalist only when nearly always right. Run through
src.blind so that nothing from the match reaches an assistant.
"""

import os
from dataclasses import replace

import numpy as np

from . import barath, stack, submission
from .match import D, ROOT

MODEL = os.path.join(ROOT, "models", "v2")
OUTPUT = os.path.join(ROOT, "evaluation", "results", "827.csv")


def finish(codes, half, kickoff=submission.SECOND_HALF_KICKOFF):
    """The published fact that the second half's graded frames before kickoff are dead ball."""
    out = codes.copy()
    out[np.flatnonzero(half == 2)[0]:kickoff] = D
    return out


def labels(m, trained, kickoff=submission.SECOND_HALF_KICKOFF):
    """The model sees the second half from its kickoff, where every training match starts a half,
    and the graded frames before it are filled with D."""
    half = m.half.copy()
    half[:kickoff][half[:kickoff] == 2] = 0
    codes = trained.predict(replace(m, half=half), stack.COSTS, stack.GAIN, both_sides=True)
    return finish(codes, m.half, kickoff)


def main():
    submission.write(OUTPUT, labels(barath.load(), stack.Stack.load(MODEL)))


if __name__ == "__main__":
    main()
