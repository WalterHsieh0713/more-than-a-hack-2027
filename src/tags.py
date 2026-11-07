"""Smart Tagging as soft evidence for which team has the ball.

A phase names a team. Attacking phases point to that team and defending phases to the other. Tags
are automatic and only roughly timed, so they nudge the team odds rather than deciding anything.

No other match has these tags, so the weight is chosen on simulated tags on external matches:
    python -m src.tags
"""

import os

import numpy as np

from .match import A, D, H, X

WITH_BALL = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD",
             "BALL IN THE BOX", "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL"}
WITHOUT_BALL = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
NO_TEAM = "N/A"
PHASE_SECONDS = 8           # rough length of a simulated phase
TIMING_SECONDS = 2          # how far a simulated phase may sit from what it describes
RATE = 5                    # rows a second
WEIGHT = 0.5                # safe on simulated tags from barely-better-than-chance upwards


def row_signs(events, index, step, home, away):
    """+1 where tags point to home, -1 to away, 0 where they are silent or cancel out."""
    totals = np.zeros(len(index), dtype=np.int64)
    teams = {home.lower(): 1, away.lower(): -1}
    for event in events:
        code = (event["code"] or "").strip().upper()
        direction = 1 if code in WITH_BALL else -1 if code in WITHOUT_BALL else 0
        team = (event["tags"].get("Team") or NO_TEAM).strip().lower()
        if direction == 0 or team == NO_TEAM.lower():
            continue
        if team not in teams:
            raise ValueError("unknown team in tags")
        inside = (index + step > event["start_frame"]) & (index < event["end_frame"])
        totals[inside] += direction * teams[team]
    return np.sign(totals)


def apply(p, sign, weight):
    """Chances with the home-to-away odds multiplied by exp(weight * sign), dead ball unchanged."""
    out = p.copy()
    home, away = p[:, H] * np.exp(weight * sign), p[:, A]
    alive = p[:, H] + p[:, A]
    total = np.maximum(home + away, 1e-12)
    out[:, H], out[:, A] = alive * home / total, alive * away / total
    return out


def simulated(truth, reliability, coverage, rng):
    """Tags for a row-level truth: phases of about PHASE_SECONDS that cover the share asked for,
    roughly timed, and naming the team that truly has the ball with the given reliability."""
    n = len(truth)
    sign = np.zeros(n, dtype=np.int64)
    length = PHASE_SECONDS * RATE
    gap = int(length * (1 - coverage) / max(coverage, 1e-6))
    start = int(rng.integers(0, length))
    while start < n:
        stop = start + int(rng.integers(length // 2, length * 3 // 2))
        shift = int(rng.integers(-TIMING_SECONDS * RATE, TIMING_SECONDS * RATE + 1))
        first, last = max(0, start + shift), min(n, stop + shift)
        here = truth[first:last]
        here = here[(here == H) | (here == A)]
        if len(here):
            team = 1 if (here == H).mean() >= 0.5 else -1
            sign[first:last] = team if rng.random() < reliability else -team
        start = stop + int(rng.integers(0, 2 * gap + 1))
    sign[(truth == X) | (truth == D)] = 0
    return sign


def _experiment(runs=None, costs=None, stage="stage5-tags"):
    """runs: {simulator: [saved held-out chances]}, by default the frozen model's, decoded with
    costs, by default the frozen decoder."""
    from . import bench, tune
    from .decode import decode
    from .experiment import DECODER, OUT
    import glob
    ledger, costs = tune.LEDGER, DECODER if costs is None else costs
    if runs is None:
        runs = {test: [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(
            OUT, "noise+noise_b_{}_*.npz".format(test))))] for test in ("noise", "noise_b")}

    def row_truth(s):
        return s["truth"][np.minimum(s["index"] + 2, len(s["truth"]) - 1)]

    def score(weight, reliability, coverage):
        total = []
        for test, saved in runs.items():
            pairs = []
            for i, s in enumerate(saved):
                rng = np.random.default_rng(1000 * i + int(reliability * 100))
                sign = simulated(row_truth(s), reliability, coverage, rng)
                p = apply(s["chance"].astype(np.float64), sign, weight)
                half = s["half"]
                codes = np.concatenate([decode(p[half == h], s["boundary"][half == h], costs, 1)
                                        for h in np.unique(half)])
                pairs.append((s["truth"], tune.frames(codes, s["index"], tune.STEP, s["truth"])))
            total.append(bench.spell_f1(pairs))
        return float(np.mean(total))

    print("spellF1, mean of both simulators, tags covering 60% of live rows")
    print("{:<8}".format("weight") + "".join("{:>10}".format("right " + str(r))
                                             for r in (0.55, 0.65, 0.75, 0.85, 0.95)))
    for weight in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        row = [score(weight, r, 0.6) for r in (0.55, 0.65, 0.75, 0.85, 0.95)]
        print("{:<8}".format(weight) + "".join("{:>10.3f}".format(v) for v in row), flush=True)
        bench.record(ledger, stage, "weight{}".format(weight), {"scores": row})


if __name__ == "__main__":
    _experiment()
