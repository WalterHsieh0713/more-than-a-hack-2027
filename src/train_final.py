"""Train the final model on every allowed match and save its weights.

    python -m src.train_final

Uses the German matches and Metrica Games 1-2 through both simulators, single-threaded. Never
touches the provided match, Metrica Game 3 or SkillCorner. Regenerating the submission does not
need this: it reads the saved weights, because LightGBM only promises repeatable training on one
system.
"""

import time

from . import dev_real, submit


def main():
    started = time.time()
    dev_real.train_model(rounds=dev_real.model.ROUNDS, threads=1).save(submit.MODEL)
    print("saved {} after {:.0f}s".format(submit.MODEL, time.time() - started))


if __name__ == "__main__":
    main()
