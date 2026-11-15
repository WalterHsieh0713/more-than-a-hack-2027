"""Train the final model and save its weights.

    python -m src.train_final

The averaged stack from src.stack, trained on the CC BY German matches through both simulators,
single-threaded. Never touches the provided match, Metrica or SkillCorner. Regenerating the
submission does not need this: it reads the saved weights, because LightGBM only promises
repeatable training on one system.
"""

import time

from . import experiment, stack, submit


def main():
    started = time.time()
    stack.train_averaged(experiment.german(), threads=1).save(submit.MODEL)
    print("saved {} after {:.0f}s".format(submit.MODEL, time.time() - started))


if __name__ == "__main__":
    main()
