"""Score predictions on labelled matches with the official metric code, unchanged."""

import numpy as np

from .match import X, pm


def score(labels, pred, bootstrap_n=0):
    """pm.evaluate on one match. Frames the truth does not cover are forced to X in pred."""
    pred = np.where(labels == X, X, pred).astype(np.int8)
    return pm.evaluate(labels, pred, bootstrap_n=bootstrap_n)


def pooled(pairs):
    """Score several matches as one long match: concatenated with X separators between them.

    Pooling, rather than averaging per-match scores, is what the full-match board does with its
    ~300 possessions, so pooled spellF1 is the closest analogue to the number we are ranked on.
    """
    sep = np.array([X], dtype=np.int8)
    gt = np.concatenate([np.concatenate([g, sep]) for g, _ in pairs])
    pr = np.concatenate([np.concatenate([p, sep]) for _, p in pairs])
    return score(gt, pr)


def summary(result):
    return "spellF1 {:.4f}  macroF1 {:.4f}  poss {} (gt {})  mean {:.1f}s (gt {:.1f}s)".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
        result["n_gt_spells"], result["mean_spell_duration_s"],
        result["gt_mean_spell_duration_s"])
