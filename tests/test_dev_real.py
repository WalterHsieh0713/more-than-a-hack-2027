import numpy as np
import pytest

from src import dev_real
from src.external import skillcorner
from src.match import A, D, H, X


def test_the_development_matches_never_overlap_the_final_test():
    dev, final = set(dev_real.DEV), set(dev_real.final_matches())
    assert len(dev) == 5 and len(final) == 15
    assert not dev & final and dev | final == set(skillcorner.MATCHES)


def test_loading_a_final_test_match_for_development_is_refused():
    with pytest.raises(ValueError):
        dev_real.load_dev(dev_real.final_matches()[0])


def test_call_quality_counts_only_frames_with_a_truth():
    truth = np.array([H, H, A, A, D, D, X, X], dtype=np.int8)
    pred = np.array([H, A, A, A, D, H, H, D], dtype=np.int8)
    q = dev_real.call_quality(truth, pred)
    assert q["alive_dead"] == pytest.approx(5 / 6)
    assert q["team"] == pytest.approx(3 / 4)


def test_call_quality_counts_a_live_frame_called_dead_as_wrong_on_alive_dead_only():
    truth = np.array([H, H], dtype=np.int8)
    pred = np.array([H, D], dtype=np.int8)
    q = dev_real.call_quality(truth, pred)
    assert q["alive_dead"] == pytest.approx(0.5) and q["team"] == 1.0
