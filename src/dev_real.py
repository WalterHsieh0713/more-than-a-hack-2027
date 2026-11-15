"""Check the model on real automatic tracking, on five SkillCorner matches set aside for it.

    python -m src.dev_real

Real matches and a real tracker, but a broadcast camera, a filled-in ball and SkillCorner's own
labels, so this warns of trouble rather than forecasting a score. The other fifteen, with Metrica's
third game, stay untouched for one final reading.
"""

import argparse
import time

import numpy as np

from . import bench, experiment, features, model
from .baseline import predict
from .external import skillcorner
from .external.labels import challenge_labels
from .match import D, X

DEV = skillcorner.MATCHES[:5]
BASELINE_SECONDS = 2


def final_matches():
    return [match for match in skillcorner.MATCHES if match not in DEV]


def load_dev(match_id):
    if match_id not in DEV:
        raise ValueError("not a development match")
    return skillcorner.load(match_id)


def call_quality(truth, pred):
    """Share right on alive or dead, and on team where both say the ball is alive."""
    graded = truth != X
    alive_truth, alive_pred = truth != D, pred != D
    both = graded & alive_truth & alive_pred & (pred != X)
    return {"alive_dead": float((alive_truth[graded] == alive_pred[graded]).mean()),
            "team": float((truth[both] == pred[both]).mean()) if both.any() else 0.0}


def train_model(rounds, threads):
    parts = []
    for m in experiment.matches().values():
        for name in ("noise", "noise_b"):
            for copy in experiment.copies(experiment.SIMULATORS[name]):
                seen = copy(m)
                parts += [model.table(seen), model.table(features.swap_teams(seen))]
    x, y, near = (np.concatenate([p[k] for p in parts]) for k in range(3))
    return model.train(x, y, near, rounds=rounds, threads=threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=model.ROUNDS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    trained = train_model(args.rounds, args.threads)
    print("trained {:.0f}s".format(time.time() - started), flush=True)

    ours, theirs = [], []
    for match_id in DEV:
        m = load_dev(match_id)
        truth = challenge_labels(m.labels, m.fps)
        mine = trained.predict(m, experiment.DECODER, gain=1)
        ours.append((truth, mine))
        theirs.append((truth, predict(m, BASELINE_SECONDS * m.fps)))
        q = call_quality(truth, mine)
        print("{}: ours {:.3f} baseline {:.3f}  alive/dead {:.3f}  team {:.3f}".format(
            match_id, bench.spell_f1(ours[-1:], m.fps), bench.spell_f1(theirs[-1:], m.fps),
            q["alive_dead"], q["team"]), flush=True)
    fps = skillcorner.FPS
    low, high = bench.difference(ours, theirs, fps=fps)
    print("pooled ours {:.3f} baseline {:.3f}  gain interval [{:.3f}, {:.3f}]".format(
        bench.spell_f1(ours, fps), bench.spell_f1(theirs, fps), low, high))
