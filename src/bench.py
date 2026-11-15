"""The measuring stick: honest scores on other matches, for choosing between versions.

There is no leaderboard, so every choice is made here, on external data only. The rules live in
the code: a match never trains on itself, the five Fortuna Dusseldorf home games stay on one side
of a split because they are not independent, and each stage may try only MAX_CONFIGS versions.

Intervals resample 90-second blocks of a match rather than whole possessions, because in testing
the scorer's own bootstrap came out too narrow. spell_f1 sums the official scorer's matches per
block, so it reproduces pm.evaluate exactly.
"""

import json
from dataclasses import dataclass

import numpy as np

from .external.labels import challenge_labels
from .match import pm
from .scoring import pooled

SINGLES = [["J03WMX"], ["J03WN1"], ["J03WOH"], ["J03WOY"], ["J03WPY"], ["J03WQQ"], ["J03WR9"]]
GROUPED = [["J03WMX"], ["J03WN1"], ["J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]]
BLOCK_SECONDS = 90
MAX_CONFIGS = 10


@dataclass
class Result:
    pairs: list          # [(truth, prediction)] per test match, on the challenge's labels
    score: dict          # pm.evaluate on the pooled matches


def splits(groups):
    """(test_ids, train_ids) for each group held out in turn."""
    for held_out in groups:
        yield held_out, [m for group in groups if group is not held_out for m in group]


def evaluate(fit, matches, groups, degrade=None, seed=0, level=1.0):
    """Hold out each group, train on the rest, and score what was held out, pooled.

    fit(train_matches) returns predict(match) -> codes. Each match is degraded once, with its own
    seed, so every fold sees the same tracking for it.
    """
    ready = {}
    for index, name in enumerate(sorted(matches)):
        m = matches[name]
        ready[name] = degrade(m, seed * 1000 + index, level) if degrade else m

    pairs = []
    for test, train in splits(groups):
        predict = fit([ready[name] for name in train])
        for name in test:
            m = ready[name]
            pairs.append((challenge_labels(m.labels, m.fps), predict(m)))
    return Result(pairs=pairs, score=pooled(pairs))


def _block_counts(gt, pred, fps):
    """Matches, true spells and predicted spells per block, one row of matches per IoU."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    size = BLOCK_SECONDS * fps
    n_blocks = len(gt) // size + 1
    gt_block = np.array([start // size for start, _, _ in gt_spells], dtype=np.int64)
    pred_block = np.array([start // size for start, _, _ in pred_spells], dtype=np.int64)
    hits = np.vstack([np.bincount(gt_block, pm.matched_gt_flags(overlaps, len(gt_spells), iou),
                                  n_blocks) for iou in pm.SPELL_IOUS])
    return hits, np.bincount(gt_block, minlength=n_blocks), np.bincount(pred_block,
                                                                        minlength=n_blocks)


def _counts(pairs, fps):
    parts = [_block_counts(gt, pred, fps) for gt, pred in pairs]
    return (np.hstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def _f1(hits, n_gt, n_pred):
    """Mean over IoUs of 2TP / (spells + predictions), which is the scorer's F1."""
    total = n_gt.sum() + n_pred.sum()
    return float(np.mean(2.0 * hits.sum(axis=1) / total)) if total else 0.0


def spell_f1(pairs, fps=25):
    return _f1(*_counts(pairs, fps))


def _draws(n_blocks, draws, seed):
    return np.random.default_rng(seed).integers(0, n_blocks, (draws, n_blocks))


def interval(pairs, fps=25, draws=1000, seed=0):
    """5th and 95th percentile of spellF1 over resampled blocks."""
    hits, n_gt, n_pred = _counts(pairs, fps)
    scores = [_f1(hits[:, pick], n_gt[pick], n_pred[pick])
              for pick in _draws(len(n_gt), draws, seed)]
    return tuple(float(v) for v in np.percentile(scores, (5, 95)))


def difference(better, worse, fps=25, draws=1000, seed=0):
    """Interval of spellF1(better) - spellF1(worse), resampling the same blocks for both."""
    a, b = _counts(better, fps), _counts(worse, fps)
    gaps = [_f1(a[0][:, pick], a[1][pick], a[2][pick]) - _f1(b[0][:, pick], b[1][pick], b[2][pick])
            for pick in _draws(len(a[1]), draws, seed)]
    return tuple(float(v) for v in np.percentile(gaps, (5, 95)))


def record(path, stage, config, numbers):
    """Append a result, refusing a new config once a stage has used up its tries."""
    seen = set()
    try:
        with open(path) as handle:
            entries = [json.loads(line) for line in handle]
        seen = {e["config"] for e in entries if e["stage"] == stage}
    except FileNotFoundError:
        pass
    if config not in seen and len(seen) >= MAX_CONFIGS:
        raise RuntimeError("stage {} has used its {} configs".format(stage, MAX_CONFIGS))
    with open(path, "a", newline="\n", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(stage=stage, config=config, **numbers), sort_keys=True) + "\n")
