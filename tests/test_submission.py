import hashlib

import numpy as np
import pytest

from src import submission
from src.match import A, AWAY, D, H, HOME, Match, X, pm
from tests.fake_atd import PITCH


def test_the_file_is_exactly_these_bytes(tmp_path):
    path = tmp_path / "s.csv"
    submission.write(path, np.array([H, A, D, X], dtype=np.int8), n_frames=4)
    assert path.read_bytes() == b"frame,label\n0,H\n1,A\n2,D\n3,X\n"


def test_writing_twice_gives_the_same_hash(tmp_path):
    codes = np.random.default_rng(0).integers(0, 4, 1000).astype(np.int8)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    submission.write(a, codes, n_frames=1000)
    submission.write(b, codes, n_frames=1000)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def test_the_official_validator_accepts_the_file(tmp_path):
    path = tmp_path / "s.csv"
    submission.write(path, np.zeros(10, dtype=np.int8), n_frames=10)
    assert pm.validate_submission(pm.read_label_csv(path), 10) is None


def test_default_length_is_the_tactical_video(tmp_path):
    with pytest.raises(ValueError):
        submission.write(tmp_path / "s.csv", np.zeros(164488, dtype=np.int8))


def test_a_wrong_length_leaves_no_file_behind(tmp_path):
    path = tmp_path / "s.csv"
    with pytest.raises(ValueError):
        submission.write(path, np.zeros(9, dtype=np.int8), n_frames=10)
    assert not path.exists()


def test_a_missing_folder_is_created(tmp_path):
    path = tmp_path / "new" / "s.csv"
    submission.write(path, np.zeros(1, dtype=np.int8), n_frames=1)
    assert path.exists()


def test_safety_net_is_the_baseline_with_ungraded_x_and_pre_kickoff_d():
    half = np.array([0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0], dtype=np.int8)
    n = len(half)
    m = Match(name="tiny", fps=25, pitch=PITCH, half=half,
              ball=np.tile([1.0, 1.0, np.nan], (n, 1)),
              players=np.zeros((n, 1, 2)), team=np.array([HOME], dtype=np.int8))
    got = submission.safety_net(m, kickoff=9)
    assert "".join(pm.LABELS[c] for c in got) == "XXHHHHXDDHHHX"
