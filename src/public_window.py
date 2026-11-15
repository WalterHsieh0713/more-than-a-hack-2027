"""Show a person how the public ground truth is built, beside the open games.

    uv run python -m src.public_window

Run it in your own terminal, not through an assistant: it prints labels from the match. It refuses
to run when its output is being captured. It is for understanding what counts as possession. No
number it prints may be copied into a parameter, because the organisers forbid tuning on this data.
"""

import os
import sys

from .label_stats import report
from .match import ROOT, pm
from .external import idsse

GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25


def main():
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        print("refused: run this yourself in a terminal, because it prints match labels")
        return 2
    open_games = {match_id: idsse.load(match_id).labels for match_id in idsse.MATCHES
                  if os.path.isfile(idsse.cache_path(match_id))}
    print(report(pm.label_codes(pm.read_label_csv(GT_PUBLIC)), open_games, FPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
