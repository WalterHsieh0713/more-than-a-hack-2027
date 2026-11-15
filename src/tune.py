"""Choose decoder settings from saved held-out chances, on open data only.

    python -m src.tune

Every setting tried is written to the ledger, which stops a stage after ten. A setting is scored
through both simulator directions and judged on its weaker one, so it cannot win by suiting a
single simulator.
"""

import glob
import os

import numpy as np

from . import bench, experiment
from .baseline import predict
from .decode import Costs, decode
from .match import ROOT, X

LEDGER = os.path.join(ROOT, "experiments", "ledger.jsonl")
STAGE = "stage4-decoder"
DIRECTIONS = (("noise", "noise_b"), ("noise_b", "noise"))
STEP = 5


def frames(codes, index, step, truth):
    """Row labels as one label per frame, X wherever the truth is ungraded."""
    out = np.full(len(truth), X, dtype=np.int8)
    for code, first in zip(codes, index):
        out[first:first + step] = code
    return np.where(truth == X, X, out).astype(np.int8)


def predicted(saved, costs, gain):
    half = saved["half"]
    codes = np.concatenate([decode(saved["chance"][half == h], saved["boundary"][half == h],
                                   costs, gain) for h in np.unique(half)])
    return frames(codes, saved["index"], STEP, saved["truth"])


def held_out(train_sim, test_sim):
    paths = sorted(glob.glob(os.path.join(experiment.OUT, "{}_{}_*.npz".format(train_sim,
                                                                               test_sim))))
    return [dict(np.load(path)) for path in paths]


def score(saved_matches, costs, gain):
    return bench.spell_f1([(s["truth"], predicted(s, costs, gain)) for s in saved_matches])


def settings():
    """Nine settings: three overall switching costs, three boundary weights."""
    out = {}
    for scale in (1000, 2000, 3000):
        for gain in (0, 0.5, 1):
            out["scale{}_gain{}".format(scale, gain)] = (
                Costs(turnover=scale, stop=scale, restart_other=scale * 8 // 10,
                      restart_same=scale * 12 // 10), gain)
    return out


if __name__ == "__main__":
    saved = {direction: held_out(*direction) for direction in DIRECTIONS}
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    print("{:<20} {:>16} {:>16}".format("setting", *["->".join(d) for d in DIRECTIONS]))
    for name, (costs, gain) in settings().items():
        numbers = [score(saved[d], costs, gain) for d in DIRECTIONS]
        print("{:<20} {:>16.3f} {:>16.3f}".format(name, *numbers), flush=True)
        bench.record(LEDGER, STAGE, name, {"scores": numbers})
