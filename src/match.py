"""One representation of a match, whatever it was loaded from.

Everything downstream -- features, models, decoder, scoring -- takes a Match, so the code that runs
on the open training games is the code that runs on BAR-ATH. Only the loaders differ.

Conventions:
  - positions in metres, origin at a pitch corner, x along the length in [0, L], y in [0, W]
  - NaN wherever an object is not observed, for players and ball alike
  - labels use the scorer's codes (evaluation/possession_metrics.py): H=0, A=1, D=2, X=3
  - one array over the whole match; halves are separated by at least one X frame so that the
    scorer's valid_runs() never lets a possession straddle halftime
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
HOME, AWAY = 0, 1


@dataclass
class Match:
    name: str
    fps: int
    pitch: tuple                      # (length, width) in metres
    half: np.ndarray                  # int8 [T]: 1, 2, or 0 on separator / ungraded frames
    ball: np.ndarray                  # float32 [T, 3]: x, y, z (z NaN when the source has none)
    players: np.ndarray               # float32 [T, P, 2]
    team: np.ndarray                  # int8 [P]: HOME or AWAY per slot
    labels: Optional[np.ndarray] = None   # int8 [T], or None when there is no ground truth
    source_frame: Optional[np.ndarray] = None  # int64 [T]: the provider's own frame number
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.half)

    def save(self, path):
        arrays = {k: getattr(self, k) for k in ("half", "ball", "players", "team")}
        for k in ("labels", "source_frame"):
            if getattr(self, k) is not None:
                arrays[k] = getattr(self, k)
        np.savez_compressed(path, name=self.name, fps=self.fps, pitch=np.array(self.pitch),
                            **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(name=str(z["name"]), fps=int(z["fps"]), pitch=tuple(z["pitch"].tolist()),
                   half=z["half"], ball=z["ball"], players=z["players"], team=z["team"],
                   labels=z["labels"] if "labels" in z else None,
                   source_frame=z["source_frame"] if "source_frame" in z else None)


def label_string(codes):
    """Codes to the H/A/D/X letters a submission file uses."""
    return np.array(pm.LABELS)[codes]
