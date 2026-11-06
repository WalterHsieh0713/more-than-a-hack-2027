"""A dilated temporal convolution over a half's feature rows, as a challenger to the tree heads.

    python -m src.sequence NAME [--epochs N] [--dead net|trees]

Reads the same 5 Hz rows as the frozen model, a whole half at a time, so a change can be placed by
what happens on both sides of it. Two networks: one on the player columns says whether the ball is
dead, one on every column says which team has it and whether a change is near. The decoder is
unchanged. Weights live in a plain npz of arrays. Inference runs on one CPU thread; training is
seeded, and on one machine repeats exactly, though not necessarily across machines.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np

# LightGBM and torch each bring their own OpenMP runtime. On macOS the process survives only if
# torch loads first, this variable is set, and LightGBM then runs on one thread.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402

from . import bench, challenge, experiment, features, model  # noqa: E402
from .decode import decode  # noqa: E402
from .match import A, D, EXTERNAL_DIR, ROOT, X  # noqa: E402

HIDDEN = 64
DILATIONS = (1, 2, 4, 8, 16, 32, 64)     # receptive field 255 rows, about 50 s
KERNEL = 3
DROPOUT = 0.1
EPOCHS = 8
WINDOW = 512                             # rows per training sequence
BATCH = 64
LR = 1e-3
SEED = 7
ALL_COLUMNS = list(range(len(features.NAMES)))
LEDGER = os.path.join(ROOT, "experiments", "ledger_sequence.jsonl")
EXTRA = ("mean", "std", "columns", "dilations")      # npz entries that are not weights


@dataclass
class Block:
    """One half's rows in order. labels and near are None without ground truth."""
    half: int
    index: np.ndarray
    x: np.ndarray
    labels: np.ndarray
    near: np.ndarray


def sequences(m):
    x, index = features.build(m)
    half = m.half[index]
    labels = near = None
    if m.labels is not None:
        labels = model.row_labels(m, index)
        near = model.near_change(labels, model.NEAR_ROWS)
    return [Block(int(h), index[half == h], x[half == h],
                  None if labels is None else labels[half == h],
                  None if near is None else near[half == h]) for h in np.unique(half)]


def dead_targets(b):
    """[rows, 1]: 1 where dead, NaN where ungraded."""
    out = np.where(b.labels == X, np.nan, (b.labels == D).astype(np.float32))
    return out[:, None]


def live_targets(b):
    """[rows, 2]: away has the ball (NaN unless live play), and a change is near."""
    graded = b.labels != X
    away = np.where(graded & (b.labels != D), (b.labels == A).astype(np.float32), np.nan)
    near = np.where(graded, b.near.astype(np.float32), np.nan)
    return np.column_stack([away, near])


class Net(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden=HIDDEN, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.inp = torch.nn.Conv1d(n_in, hidden, 1)
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, KERNEL, padding=d, dilation=d), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Conv1d(hidden, hidden, 1))
            for d in self.dilations])
        self.out = torch.nn.Conv1d(hidden, n_out, 1)

    def forward(self, x):
        """[batch, steps, features] -> [batch, steps, outputs]."""
        h = self.inp(x.transpose(1, 2))
        for block in self.blocks:
            h = torch.relu(h + block(h))
        return self.out(h).transpose(1, 2)


@dataclass
class Scale:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def of(cls, blocks, columns):
        x = np.concatenate([b.x[:, columns] for b in blocks])
        return cls(np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-3)

    def apply(self, x):
        return np.nan_to_num((x - self.mean) / self.std, nan=0.0).astype(np.float32)


class Trained:
    """A network with the scaling and the columns it was trained on."""

    def __init__(self, net, scale, columns):
        self.net, self.scale, self.columns = net.train(False), scale, list(columns)

    def logits(self, x):
        torch.set_num_threads(1)
        with torch.no_grad():
            z = torch.from_numpy(self.scale.apply(x[:, self.columns]))[None]
            return self.net(z)[0].numpy()

    def proba(self, x):
        return 1 / (1 + np.exp(-self.logits(x).astype(np.float64)))

    def save(self, path):
        state = {k: v.numpy() for k, v in self.net.state_dict().items()}
        np.savez(path, mean=self.scale.mean, std=self.scale.std, columns=np.array(self.columns),
                 dilations=np.array(self.net.dilations), **state)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        net = Net(z["inp.weight"].shape[1], z["out.weight"].shape[0], z["inp.weight"].shape[0],
                  tuple(z["dilations"].tolist()))
        net.load_state_dict({k: torch.from_numpy(z[k]) for k in z.files if k not in EXTRA})
        return cls(net, Scale(z["mean"], z["std"]), z["columns"].tolist())


class TreeDead:
    """The frozen model's dead head, for a run that changes only the live heads."""

    def __init__(self, trees):
        self.trees = trees

    def proba(self, x):
        return self.trees.dead.predict(x[:, model.PLAYER_COLUMNS], num_threads=1)[:, None]


class TreeLive:
    """The frozen model's team and change heads, to be averaged with the network's."""

    def __init__(self, trees):
        self.trees = trees

    def proba(self, x):
        return np.column_stack([self.trees.team.predict(x, num_threads=1),
                                self.trees.change.predict(x, num_threads=1)])


def _windows(blocks, window):
    """(block number, first row) of every training sequence, half overlapping."""
    starts = []
    for i, b in enumerate(blocks):
        last = max(0, len(b.x) - window)
        starts += [(i, s) for s in range(0, last + 1, max(1, window // 2))]
        if last % max(1, window // 2):
            starts.append((i, last))
    return starts


def _gather(blocks, scaled, targets, picks, window, n_out):
    xb = np.zeros((len(picks), window, scaled[0].shape[1]), dtype=np.float32)
    yb = np.full((len(picks), window, n_out), np.nan, dtype=np.float32)
    for k, (i, s) in enumerate(picks):
        n = min(window, len(scaled[i]) - s)
        xb[k, :n], yb[k, :n] = scaled[i][s:s + n], targets[i][s:s + n]
    return torch.from_numpy(xb), torch.from_numpy(yb)


def device():
    """Apple's GPU when there is one: training there is many times faster than on the CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_net(blocks, columns, target_of, threads=1, epochs=EPOCHS, window=WINDOW, batch=BATCH,
              seed=SEED, hidden=HIDDEN, dilations=DILATIONS, lr=LR):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    where = device()
    scale = Scale.of(blocks, columns)
    scaled = [scale.apply(b.x[:, columns]) for b in blocks]
    targets = [target_of(b) for b in blocks]
    n_out = targets[0].shape[1]
    net = Net(len(columns), n_out, hidden, dilations).to(where)
    starts = _windows(blocks, window)
    steps = epochs * ((len(starts) + batch - 1) // batch)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(epochs):
        order = rng.permutation(len(starts))
        for i in range(0, len(order), batch):
            xb, yb = _gather(blocks, scaled, targets, [starts[j] for j in order[i:i + batch]],
                             window, n_out)
            xb, yb = xb.to(where), yb.to(where)
            known = torch.isfinite(yb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(xb)[known], yb[known])
            opt.zero_grad()
            loss.backward()
            opt.step()
            schedule.step()
    return Trained(net.cpu(), scale, columns)


class Mean:
    """Heads whose chances are averaged, so that several seeds count as one head."""

    def __init__(self, heads):
        self.heads = heads

    def proba(self, x):
        return np.mean([head.proba(x) for head in self.heads], axis=0)


class Challenger:
    """A dead head and one or more live heads, whose chances are averaged."""

    def __init__(self, dead, lives):
        self.dead, self.lives = dead, lives

    def predict(self, m, costs, gain=1):
        blocks = sequences(m)
        codes = []
        for b in blocks:
            dead = self.dead.proba(b.x)[:, 0]
            away, change = Mean(self.lives).proba(b.x).T
            p = np.column_stack([(1 - dead) * (1 - away), (1 - dead) * away, dead])
            codes.append(decode(p, change, costs, gain))
        return features.expand(np.concatenate(codes), np.concatenate([b.index for b in blocks]), m)

    def save(self, folder):
        """Networks as npz files of arrays, trees as LightGBM's text files, and a note of how
        they are combined."""
        os.makedirs(folder, exist_ok=True)
        nets = self.lives[0].heads
        for i, net in enumerate(nets):
            net.save(os.path.join(folder, f"live{i}.npz"))
        shape = dict(nets=len(nets), blend=len(self.lives) > 1,
                     dead="net" if isinstance(self.dead, Trained) else "trees")
        if shape["dead"] == "net":
            self.dead.save(os.path.join(folder, "dead.npz"))
        for head in [self.dead] + self.lives[1:]:
            if hasattr(head, "trees"):
                head.trees.save(folder)
                break
        with open(os.path.join(folder, "shape.json"), "w", encoding="utf-8") as handle:
            json.dump(shape, handle)

    @classmethod
    def load(cls, folder):
        with open(os.path.join(folder, "shape.json"), encoding="utf-8") as handle:
            shape = json.load(handle)
        lives = [Mean([Trained.load(os.path.join(folder, f"live{i}.npz"))
                       for i in range(shape["nets"])])]
        trees = model.Model.load(folder) if shape["blend"] or shape["dead"] == "trees" else None
        if shape["blend"]:
            lives.append(TreeLive(trees))
        dead = TreeDead(trees) if shape["dead"] == "trees" else Trained.load(
            os.path.join(folder, "dead.npz"))
        return cls(dead, lives)


def fit(matches, copies, threads=1, dead="net", blend=False, nets=1, **settings):
    blocks = []
    for m in matches.values():
        for copy in copies:
            seen = copy(m)
            blocks += sequences(seen) + sequences(features.swap_teams(seen))
    return _fit_blocks(blocks, threads, dead, blend, nets, **settings)


def _fit_blocks(blocks, threads, dead, blend, nets=1, **settings):
    seed = settings.pop("seed", SEED)
    lives = [Mean([train_net(blocks, ALL_COLUMNS, live_targets, threads, seed=seed + i, **settings)
                   for i in range(nets)])]
    if dead == "trees" or blend:
        x = np.concatenate([b.x for b in blocks])
        y = np.concatenate([b.labels for b in blocks])
        near = np.concatenate([b.near for b in blocks])
        keep = y != X
        trees = model.train(x[keep], y[keep], near[keep], threads=1)   # OpenMP: see above
    if blend:
        lives.append(TreeLive(trees))
    if dead == "trees":
        return Challenger(TreeDead(trees), lives)
    return Challenger(train_net(blocks, model.PLAYER_COLUMNS, dead_targets, threads, **settings),
                      lives)


_CACHE = {}


def challenger(dead="net", blend=False, nets=1, **settings):
    """fit(train_matches, threads) for challenge.run, with every noise copy built once."""
    def fit_(train, threads):
        blocks = []
        for name, m in train.items():
            if name not in _CACHE:
                _CACHE[name] = []
                for sim in challenge.SIMULATED:
                    for copy in experiment.copies(experiment.SIMULATORS[sim]):
                        seen = copy(m)
                        _CACHE[name] += sequences(seen) + sequences(features.swap_teams(seen))
            blocks += _CACHE[name]
        trained = _fit_blocks(blocks, threads, dead, blend, nets, **settings)
        return lambda m: trained.predict(m, experiment.DECODER, 1)
    return fit_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--dead", default="net")
    parser.add_argument("--blend", action="store_true", help="average in the tree live heads")
    parser.add_argument("--nets", type=int, default=1, help="networks from different seeds")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--folds", nargs="*", default=list(challenge.DEVELOPMENT))
    args = parser.parse_args()
    started = time.time()
    results = challenge.run(challenger(args.dead, args.blend, args.nets, epochs=args.epochs,
                                       hidden=args.hidden),
                            folds=tuple(args.folds), threads=args.threads)
    challenge.save(results, os.path.join(EXTERNAL_DIR, "challenge",
                                         "sequence_" + args.name + ".npz"))
    print(f"{time.time() - started:.0f}s")
    if tuple(args.folds) == challenge.DEVELOPMENT:
        outcome = challenge.report(results)
        bench.record(LEDGER, "sequence", args.name, dict(
            settings=dict(epochs=args.epochs, hidden=args.hidden, dead=args.dead,
                          blend=args.blend, nets=args.nets),
            passes=outcome["passes"],
            **{test: [row["score"], row["gain"], row["low"], row["high"]]
               for test, row in outcome["rows"].items()}))
    else:
        for test in results:
            print(test, f"{bench.spell_f1(results[test], challenge.FPS[test]):.3f}")
