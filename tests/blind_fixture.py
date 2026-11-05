"""Checks that try every way of leaking a value through src.blind."""

import os
import sys

SECRET = "SECRET-MATCH-VALUE-0.73125"


def passes():
    print(SECRET)
    sys.stderr.write(SECRET)


def raises():
    print(SECRET)
    raise ValueError(SECRET)


def asserts():
    assert 1 == 2, SECRET


def exits_early():
    print(SECRET)
    sys.exit(0)


def dies():
    print(SECRET)
    sys.stdout.flush()
    os._exit(0)


CHECKS = {f.__name__: f for f in (passes, raises, asserts, exits_early, dies)}
