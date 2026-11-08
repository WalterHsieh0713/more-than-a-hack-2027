"""The single final reading: the shipped model on matches never used to train, tune or choose it.

    python -m src.final_test

Fifteen SkillCorner matches (real automatic tracking at 10 fps, no simulator) and Metrica's third
game through both simulators. External data only. v1, the first submission's model without its
tags, and the 2-second baseline are scored alongside for context.

The rule, written before the run: the reading is taken once and changes nothing, unless v2 is
reliably worse than the baseline on SkillCorner, which would point to a bug to be found on other
data. Once the results file exists this only reprints it, so the reading cannot be retaken.
"""

import os

from . import bench, challenge, experiment, model, stack, submit, tune
from .baseline import predict
from .dev_real import BASELINE_SECONDS, final_matches
from .external import metrica, skillcorner
from .external.labels import challenge_labels
from .match import EXTERNAL_DIR, ROOT

RESULTS = os.path.join(EXTERNAL_DIR, "challenge", "final.npz")
V1 = os.path.join(ROOT, "models", "v1")
READERS = ("v2", "v1", "baseline")
TESTS = ("real",) + challenge.SIMULATED
GAME = 3


def readers():
    v2, v1 = stack.Stack.load(submit.MODEL), model.Model.load(V1)
    return {"v2": lambda m: v2.predict(m, stack.COSTS, stack.GAIN),
            "v1": lambda m: v1.predict(m, experiment.DECODER, 1),
            "baseline": lambda m: predict(m, BASELINE_SECONDS * m.fps)}


def seen_matches():
    """(test, match as the model sees it) for every final match."""
    for match_id in final_matches():
        yield "real", skillcorner.load(match_id)
    game = model.fill_blanks(metrica.load(GAME))
    for sim in challenge.SIMULATED:
        yield sim, experiment.SIMULATORS[sim](game, experiment.seed_of("M3", 99), 1.0)


def run(read):
    out = {}
    for test, m in seen_matches():
        truth = challenge_labels(m.labels, m.fps)
        for who, label in read.items():
            out.setdefault(who + ":" + test, []).append((truth, label(m)))
        print("done", test, m.name, flush=True)
    return out


def report(results):
    rows = {}
    for test in TESTS:
        fps = challenge.FPS[test]
        pairs = {who: results[who + ":" + test] for who in READERS}
        score = {who: bench.spell_f1(p, fps) for who, p in pairs.items()}
        rows[test] = dict(score, matches=len(pairs["v2"]))
        line = "%-8s %2d matches  v2 %.3f  v1 %.3f  baseline %.3f" % (
            test, len(pairs["v2"]), score["v2"], score["v1"], score["baseline"])
        for other in ("baseline", "v1"):
            low, high = bench.difference(pairs["v2"], pairs[other], fps=fps)
            wins = sum(bench.spell_f1([a], fps) > bench.spell_f1([b], fps)
                       for a, b in zip(pairs["v2"], pairs[other]))
            rows[test][other + "_interval"] = (low, high)
            rows[test][other + "_wins"] = int(wins)
            line += "  v2-%s %+.3f [%+.3f, %+.3f] wins %d" % (
                other, score["v2"] - score[other], low, high, wins)
        print(line, flush=True)
    worse = rows["real"]["baseline_interval"][1] < 0
    print("v2 reliably worse than the baseline on SkillCorner: look for a bug" if worse
          else "rule: nothing changes")
    return rows


def main():
    if os.path.exists(RESULTS):
        report(challenge.load(RESULTS))
        return
    challenge.save(run(readers()), RESULTS)
    bench.record(tune.LEDGER, "final", "v2", report(challenge.load(RESULTS)))


if __name__ == "__main__":
    main()
