import hashlib

import numpy as np
import pytest

from src import checks
from src.match import A, D, H, X


def labels(minutes=20, spells_per_min=4, dead=0.3):
    """A plausible label track at 25 fps, invented."""
    n = minutes * 60 * 25
    out, teams = np.full(n, D, dtype=np.int8), (H, A)
    length = int(60 * 25 / spells_per_min)
    for i, start in enumerate(range(0, n, length)):
        out[start:start + int(length * (1 - dead))] = teams[i % 2]
    return out


def test_a_plausible_track_passes_the_sanity_bounds():
    checks.plausible(labels())


@pytest.mark.parametrize("track", [
    labels(spells_per_min=20),                       # far too many possessions
    labels(spells_per_min=1),                        # far too few
    labels(dead=0.9),                                # nearly all dead
    np.where(labels() == A, H, labels()).astype(np.int8),   # one team never has the ball
], ids=["too-many", "too-few", "all-dead", "one-team"])
def test_an_implausible_track_fails(track):
    with pytest.raises(AssertionError):
        checks.plausible(track)


def test_ungraded_frames_do_not_count_towards_the_bounds():
    track = np.concatenate([labels(), np.full(50000, X, dtype=np.int8)])
    checks.plausible(track)


def test_tags_that_mostly_agree_pass_and_reversed_teams_fail():
    n = 1000
    pred = np.array([H] * 500 + [A] * 500, dtype=np.int8)
    signs = np.array([1] * 500 + [-1] * 500)
    checks.tags_agree(pred, signs)
    with pytest.raises(AssertionError):
        checks.tags_agree(pred, -signs)


def test_the_gate_takes_three_models_and_refuses_a_fourth(tmp_path):
    path = tmp_path / "gate.jsonl"
    for name in "abc":
        checks.count_gate(path, name)
    with pytest.raises(RuntimeError):
        checks.count_gate(path, "d")


def test_rerunning_a_model_already_through_the_gate_is_not_a_new_try(tmp_path):
    path = tmp_path / "gate.jsonl"
    for name in "abc":
        checks.count_gate(path, name)
    checks.count_gate(path, "a")
    assert len(path.read_text().splitlines()) == 4


def test_the_model_hash_sees_files_in_nested_folders(tmp_path, monkeypatch):
    (tmp_path / "first").mkdir()
    (tmp_path / "first" / "dead.txt").write_text("a")
    (tmp_path / "team.txt").write_text("b")
    monkeypatch.setattr(checks.submit, "MODEL", str(tmp_path))
    before = checks._model_hash()
    (tmp_path / "first" / "dead.txt").write_text("c")
    assert checks._model_hash() != before


def test_a_flat_folder_keeps_its_old_hash_so_the_gate_still_recognises_it(tmp_path, monkeypatch):
    (tmp_path / "dead.txt").write_text("a")
    (tmp_path / "team.txt").write_text("b")
    monkeypatch.setattr(checks.submit, "MODEL", str(tmp_path))
    assert checks._model_hash() == hashlib.sha256(b"ab").hexdigest()
