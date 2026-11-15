"""How a label track is built: possession lengths and how possessions end.

The same numbers for any source, so the public window and the open games can be compared. Results
for BAR-ATH labels are for a person to read, and must not be used to set a parameter.
"""

import numpy as np

from .match import A, D, H, pm

SHORT = (1, 2, 3, 5)  # seconds


def _segments(codes):
    return pm.segments(codes, pm.valid_runs(codes))


def label_stats(codes, fps):
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    seconds = np.array([b - a for a, b, _ in spells]) / float(fps)
    segments = _segments(codes)
    minutes = sum(b - a for a, b, _ in segments) / float(fps) / 60

    stats = {
        "possessions": len(spells),
        "poss_per_min": len(spells) / minutes,
        "mean_s": float(seconds.mean()),
        "median_s": float(np.median(seconds)),
    }
    for limit in SHORT:
        stats["under_{}s".format(limit)] = 100 * float((seconds < limit).mean())

    turnovers = same = other = sandwiches = 0
    joined = list(zip(segments, segments[1:]))
    for (_, stop, left), (start, _, right) in joined:
        turnovers += stop == start and {left, right} == {H, A}
    for one, two, three in zip(segments, segments[1:], segments[2:]):
        if one[1] != two[0] or two[1] != three[0]:
            continue  # ungraded frames between them
        if two[2] == D and one[2] != D and three[2] != D:
            same += one[2] == three[2]
            other += one[2] != three[2]
        if two[2] in (H, A) and one[2] == three[2] and one[2] in (H, A):
            sandwiches += two[1] - two[0] < 2 * fps

    stats.update(stoppages=sum(k == D for _, _, k in segments), turnovers=int(turnovers),
                 same_team_restarts=int(same), other_team_restarts=int(other),
                 sandwiches_under_2s=int(sandwiches))
    return stats


def possession_rows(codes, fps):
    """[(first_frame, team, seconds)] for each possession, to seek to in the video."""
    return [(a, pm.LABELS[k], (b - a) / float(fps))
            for a, b, k in pm.possession_spells(codes, pm.valid_runs(codes))]


def report(public, others, fps):
    """The public window beside the lowest and highest value across the open games."""
    mine = label_stats(public, fps)
    rest = [label_stats(codes, fps) for codes in others.values()]
    heads = ["public"] + (["idsse_low", "idsse_high"] if rest else [])
    lines = ["{:<22}".format("") + "".join("{:>12}".format(head) for head in heads)]
    for key, value in mine.items():
        cells = [value] + ([min(s[key] for s in rest), max(s[key] for s in rest)] if rest else [])
        lines.append("{:<22}".format(key) + "".join("{:>12.1f}".format(cell) for cell in cells))
    lines += ["", "public window: frame, team, seconds"]
    lines += ["{:>7}  {}  {:>6.2f}".format(*row) for row in possession_rows(public, fps)]
    return "\n".join(lines)
