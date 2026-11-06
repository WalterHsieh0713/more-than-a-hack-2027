"""The shared test every challenger to the frozen model must pass. External data only.

A challenger is fit(train_matches, threads) -> predict(match) -> one code per frame. train_matches
is a dict of clean matches by name, so a challenger builds its own noisy copies, for which
frozen_fit shows the way. Then:

    results = challenge.run(fit)
    challenge.report(results)

DEVELOPMENT is for iterating. CONFIRM is touched once, by a finalist, because many tries on the
same matches will produce a lucky winner. The rule to replace the frozen model: at least MIN_GAIN
better on both simulators, and not reliably worse on real tracking.
"""

import os

import numpy as np

from . import bench, dev_real, experiment, features, model, submit
from .external.labels import challenge_labels
from .match import EXTERNAL_DIR

DEVELOPMENT = ("J03WMX", "J03WN1", "J03WOH", "J03WQQ", "M1")
CONFIRM = ("J03WOY", "J03WPY", "J03WR9", "M2")
MIN_GAIN = 0.02
SIMULATED = ("noise", "noise_b")
FPS = {"noise": 25, "noise_b": 25, "real": 10}
FROZEN = os.path.join(EXTERNAL_DIR, "challenge", "frozen_development.npz")


def check_folds(folds, unlocked=False):
    if set(folds) & set(CONFIRM) and not unlocked:
        raise ValueError("the confirmation matches are for finalists only")


def held_out(fit, matches, folds, simulators, threads):
    """{simulator: [(truth, prediction)]} with each fold left out of its own training."""
    out = {name: [] for name in simulators}
    for name in folds:
        predict = fit({k: m for k, m in matches.items() if k != name}, threads)
        for sim, degrade in simulators.items():
            seen = degrade(matches[name], experiment.seed_of(name, 99), 1.0)
            out[sim].append((challenge_labels(seen.labels, seen.fps), predict(seen)))
    return out


def real_tracking(fit, matches, threads):
    """Pairs on the five SkillCorner development matches, trained on everything allowed."""
    predict = fit(matches, threads)
    pairs = []
    for match_id in dev_real.DEV:
        m = dev_real.load_dev(match_id)
        pairs.append((challenge_labels(m.labels, m.fps), predict(m)))
    return pairs


def frozen_fit(train, threads):
    """The current model, written as a challenger, which also shows how to build noisy copies."""
    parts = []
    for m in train.values():
        for name in SIMULATED:
            for copy in experiment.copies(experiment.SIMULATORS[name]):
                seen = copy(m)
                parts += [model.table(seen), model.table(features.swap_teams(seen))]
    x, y, near = (np.concatenate([p[k] for p in parts]) for k in range(3))
    trained = model.train(x, y, near, threads=threads)
    return lambda m: trained.predict(m, submit.DECODER, submit.GAIN)


def run(fit, folds=DEVELOPMENT, threads=3, unlocked=False):
    check_folds(folds, unlocked)
    matches = experiment.matches()
    simulators = {name: experiment.SIMULATORS[name] for name in SIMULATED}
    results = held_out(fit, matches, folds, simulators, threads)
    results["real"] = real_tracking(fit, matches, threads)
    return results


def compare(candidate, frozen, fps=FPS, draws=1000):
    """Paired gains on each test, and whether the candidate passes the rule."""
    rows, passes = {}, True
    for test in sorted(candidate):
        gain = bench.spell_f1(candidate[test], fps[test]) - bench.spell_f1(frozen[test], fps[test])
        low, high = bench.difference(candidate[test], frozen[test], fps=fps[test], draws=draws)
        rows[test] = dict(score=bench.spell_f1(candidate[test], fps[test]), gain=gain, low=low,
                          high=high)
        if test == "real":
            passes = passes and high >= 0
        else:
            passes = passes and gain >= MIN_GAIN and low > 0
    return dict(rows=rows, passes=passes)


def save(results, path):
    arrays = {}
    for test, pairs in results.items():
        for i, (truth, pred) in enumerate(pairs):
            arrays["{}|{}|truth".format(test, i)] = truth
            arrays["{}|{}|pred".format(test, i)] = pred
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def load(path):
    stored = np.load(path)
    results = {}
    for key in sorted(stored.files, key=lambda k: (k.split("|")[0], int(k.split("|")[1]))):
        test, index, kind = key.split("|")
        pairs = results.setdefault(test, {})
        pairs.setdefault(int(index), {})[kind] = stored[key]
    return {test: [(p[i]["truth"], p[i]["pred"]) for i in sorted(p)]
            for test, p in results.items()}


def report(results, frozen_path=FROZEN):
    outcome = compare(results, load(frozen_path))
    for test, row in outcome["rows"].items():
        print("{:<8} {:.3f}  gain {:+.3f}  interval [{:+.3f}, {:+.3f}]".format(
            test, row["score"], row["gain"], row["low"], row["high"]))
    print("PASSES THE RULE" if outcome["passes"] else "does not pass")
    return outcome


if __name__ == "__main__":
    save(run(frozen_fit, threads=9), FROZEN)
    print("frozen development results saved")
