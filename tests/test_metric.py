"""What we rely on from the official scorer, re-checked under our pinned numpy and pandas.

The scorer is upstream code. These guard the assumptions our design rests on, so a version bump
that changes its behaviour fails here rather than quietly in a final score.
"""

import numpy as np

from src.match import A, D, H, X, pm
from src.scoring import pooled, score

# 40 H, 10 D, 30 A, 20 H with five ungraded frames either side.
GT = np.array([X] * 5 + [H] * 40 + [D] * 10 + [A] * 30 + [H] * 20 + [X] * 5, dtype=np.int8)


def spell_f1(pred, gt=GT):
    return score(gt, pred)["spell_f1_mean"]


def test_a_perfect_submission_scores_one():
    assert spell_f1(GT) == 1.0


def test_a_constant_guess_scores_zero():
    assert spell_f1(np.full(len(GT), D, dtype=np.int8)) == 0.0


def test_swapping_home_and_away_in_both_changes_nothing():
    swap = np.array([A, H, D, X], dtype=np.int8)
    pred = GT.copy()
    pred[20:30] = A
    assert spell_f1(swap[pred], swap[GT]) == spell_f1(pred)


def test_labels_on_ungraded_frames_cannot_change_the_result():
    edited = GT.copy()
    edited[:5] = H
    edited[-5:] = A
    assert pm.evaluate(GT, edited)["spell_f1_mean"] == 1.0


def test_abstaining_inside_a_possession_is_free_on_spell_f1():
    pred = GT.copy()
    pred[10:30] = X
    assert spell_f1(pred) == 1.0
    assert score(GT, pred)["macro_f1"] < 1.0


def test_abstaining_across_a_handover_carries_the_old_label():
    pred = GT.copy()
    pred[45:85] = X      # the stoppage and the whole A possession
    assert spell_f1(pred) < 1.0


def test_pooling_never_joins_a_possession_across_two_matches():
    gt = np.array([H] * 20, dtype=np.int8)
    assert pooled([(gt, gt), (gt, gt)])["n_gt_spells"] == 2
