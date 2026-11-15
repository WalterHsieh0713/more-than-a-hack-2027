"""Bring a provider's possession labels closer to the challenge's meaning.

DFL's labels flip on every touch. CHALLENGE.md says a simple touch without control or a deflection
is not a change of possession, so a very short spell between two spells of the other team goes
back to that team. Reading the public window suggests the truth still credits short, poorly
controlled tackles, so the limit must stay small. No model or score is fitted to the match: the
limit is compared on external data only.
"""

from ..match import A, H, pm

# From the definition: a touch or deflection is not a change of possession. Set after a person read
# the public window to learn what counts as possession, and saw few spells under a second there.
FLIP_LIMIT_SECONDS = 1


def _shortest_flip(segments, limit):
    """(start, stop, team) of the shortest spell under limit between two of the other team."""
    flips = [(two[1] - two[0], two[0], two[1], one[2])
             for one, two, three in zip(segments, segments[1:], segments[2:])
             if one[1] == two[0] and two[1] == three[0] and two[2] in (H, A)
             and one[2] == three[2] and one[2] in (H, A) and two[1] - two[0] < limit]
    return min(flips)[1:] if flips else None


def merge_flips(labels, fps, max_seconds):
    """Copy of labels with flips shorter than max_seconds absorbed, shortest first, so that the
    result does not depend on the order they appear in."""
    out = labels.copy()
    runs = pm.valid_runs(labels)
    while True:
        flip = _shortest_flip(pm.segments(out, runs), max_seconds * fps)
        if flip is None:
            return out
        start, stop, team = flip
        out[start:stop] = team


def challenge_labels(labels, fps):
    """A provider's labels under the rule this project trains and validates on."""
    return merge_flips(labels, fps, FLIP_LIMIT_SECONDS)
