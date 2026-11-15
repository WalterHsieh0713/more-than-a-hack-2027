"""The submission file: written identically on every operating system, or not at all."""

import os
import sys

import numpy as np
import pandas as pd

from .baseline import predict
from .match import D, ROOT, X, label_string, pm

sys.path.insert(0, ROOT)
from games import source_frames  # noqa: E402

# Published in FAQ.md. Frames from the start of the second half up to here are graded and all D.
SECOND_HALF_KICKOFF = 113742


def write(path, codes, n_frames=None):
    """Write scorer codes as frame,label. Refuses anything the official validator would.

    newline="\\n" matters: text mode would write \\r\\n on Windows, so the same labels would hash
    differently there and the file would not count as regenerated.
    """
    labels = label_string(codes)
    table = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    reason = pm.validate_submission(table, source_frames() if n_frames is None else n_frames)
    if reason:
        raise ValueError(reason)
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def safety_net(m, kickoff=SECOND_HALF_KICKOFF):
    """The published floor plus what the docs state outright: a valid file to fall back on."""
    codes = predict(m, 50)
    codes[m.half == 0] = X
    codes[np.flatnonzero(m.half == 2)[0]:kickoff] = D
    return codes
