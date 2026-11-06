"""Cross-simulator test: train through one noise simulator, test through the other.

    python -m src.experiment noise noise_b --rounds 150

Open data only. Metrica's third game and SkillCorner stay untouched for the single final score, so
this module never loads them. Held-out chances are saved so that decoder settings can be chosen
afterwards without retraining.
"""

import argparse
import os
import time
import zlib

import numpy as np

from . import bench, features, model, noise, noise_b
from .baseline import predict
from .decode import Costs
from .external import idsse, metrica
from .external.labels import challenge_labels
from .match import EXTERNAL_DIR

SIMULATORS = {"noise": noise.degrade, "noise_b": noise_b.degrade,
              "long": lambda m, seed, level: noise.degrade(m, seed, level, gap_scale=5.0),
              "none": lambda m, seed, level: noise.degrade(m, seed, 0)}
TRAIN_LEVELS = (0.7, 1.3)
TRAIN_GAMES = (1, 2)
DECODER = Costs(turnover=2000, stop=2000, restart_other=1600, restart_same=2400)
OUT = os.path.join(EXTERNAL_DIR, "held_out")


def matches():
    """Every match we may train or validate on, by name, with Metrica's blank ball bridged."""
    out = {m.name.replace("IDSSE-", ""): m for m in (idsse.load(i) for i in idsse.MATCHES)}
    for game in TRAIN_GAMES:
        out["M{}".format(game)] = model.fill_blanks(metrica.load(game))
    return out


def seed_of(name, salt):
    return (zlib.crc32(name.encode()) + salt) % (2 ** 31)


def copies(simulator, levels=TRAIN_LEVELS):
    return [lambda m, level=level: simulator(m, seed_of(m.name, int(level * 10)), level)
            for level in levels]


def groups(all_names, held_out=None):
    singles = [[name] for name in sorted(all_names)]
    return singles if held_out is None else [[name] for name in held_out]


def run(train_sim, test_sim, rounds, threads, held_out=None):
    all_matches = matches()
    started = time.time()
    tables = {}
    for name, m in all_matches.items():
        for simulator in train_sim.split("+"):
            for copy in copies(SIMULATORS[simulator]):
                seen = copy(m)
                tables.setdefault(name, []).append(model.table(seen))
                tables[name].append(model.table(features.swap_teams(seen)))
    print("rows built {:.0f}s".format(time.time() - started), flush=True)

    os.makedirs(OUT, exist_ok=True)
    ours, theirs = [], []
    for fold in groups(all_matches, held_out):
        name = fold[0]
        train_names = [n for n in all_matches if n != name]
        parts = [p for n in train_names for p in tables[n]]
        x, y, near = (np.concatenate([p[k] for p in parts]) for k in range(3))
        trained = model.train(x, y, near, rounds=rounds, threads=threads)
        del x, y, near, parts

        m = all_matches[name]
        seen = SIMULATORS[test_sim](m, seed_of(name, 99), 1.0)
        chance, boundary = trained.proba(seen)
        _, index = features.build(seen)
        np.savez(os.path.join(OUT, "{}_{}_{}.npz".format(train_sim, test_sim, name)),
                 chance=chance.astype(np.float32), boundary=boundary.astype(np.float32),
                 index=index, half=seen.half[index], truth=challenge_labels(seen.labels, seen.fps))
        ours.append(_decoded(trained, seen))
        theirs.append((ours[-1][0], predict(seen, 50)))
        print("{} done {:.0f}s  so far ours {:.3f} baseline {:.3f}".format(
            name, time.time() - started, bench.spell_f1(ours), bench.spell_f1(theirs)), flush=True)
    return ours, theirs


def _decoded(trained, seen):
    return challenge_labels(seen.labels, seen.fps), trained.predict(seen, DECODER, gain=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("train_sim")
    parser.add_argument("test_sim")
    parser.add_argument("--rounds", type=int, default=model.ROUNDS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--held-out", nargs="*")
    args = parser.parse_args()
    ours, theirs = run(args.train_sim, args.test_sim, args.rounds, args.threads, args.held_out)
    low, high = bench.difference(ours, theirs)
    print("ours {:.3f} {}  baseline {:.3f} {}  gain interval [{:.3f}, {:.3f}]".format(
        bench.spell_f1(ours), tuple(round(v, 3) for v in bench.interval(ours)),
        bench.spell_f1(theirs), tuple(round(v, 3) for v in bench.interval(theirs)), low, high))
