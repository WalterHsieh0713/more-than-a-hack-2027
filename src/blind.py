"""Run named checks on the match data so that none of their output reaches the terminal.

    python -m src.blind public_scores baselines

The competition terms forbid showing match data to a hosted LLM, and an assistant reads whatever a
command prints. So each check runs in a child process whose stdout and stderr go to a local log
under data/logs, and the terminal gets one line per check, built only from its name, the exception
type and the source location:

    PASS baselines
    FAIL baselines: AssertionError at checks.py:41

A person can open the log to debug. An assistant must not. A child that dies without a verdict is
a failure, whatever its exit code.
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
import traceback

from .match import ROOT

sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

REGISTRY = "src.checks"


def _child(name, status_path, registry):
    check = importlib.import_module(registry).CHECKS[name]
    try:
        check()
        status = "PASS {}".format(name)
    except BaseException as error:
        traceback.print_exc()
        place = traceback.extract_tb(error.__traceback__)[-1]
        status = "FAIL {}: {} at {}:{}".format(name, type(error).__name__,
                                               os.path.basename(place.filename), place.lineno)
    with open(status_path, "w") as handle:
        handle.write(status)


def run(name, registry=REGISTRY):
    logs = os.path.join(DATA_DIR, "logs")
    os.makedirs(logs, exist_ok=True)
    stem = os.path.join(logs, "{}-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), os.getpid(), name))
    status_path = stem + ".status"
    with open(stem + ".log", "wb") as log:
        subprocess.call([sys.executable, "-m", "src.blind", "--registry", registry,
                         "--child", status_path, name],
                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT)
    if not os.path.isfile(status_path):
        return "FAIL {}: crashed".format(name)
    with open(status_path) as handle:
        return handle.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+")
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        _child(args.names[0], args.child, args.registry)
        return 0

    known = importlib.import_module(args.registry).CHECKS
    unknown = sorted(set(args.names) - set(known))
    if unknown:
        print("unknown check {}; have {}".format(unknown, sorted(known)))
        return 2

    results = [run(name, args.registry) for name in args.names]
    print("\n".join(results))
    return 0 if all(line.startswith("PASS") for line in results) else 1


if __name__ == "__main__":
    sys.exit(main())
