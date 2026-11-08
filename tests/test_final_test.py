import numpy as np

from src import challenge, final_test
from src.match import A, H


def test_once_results_exist_they_are_only_reprinted(tmp_path, monkeypatch, capsys):
    truth = np.array([H] * 20 + [A] * 20, dtype=np.int8)
    saved = {who + ":" + test: [(truth, truth)]
             for who in final_test.READERS for test in final_test.TESTS}
    challenge.save(saved, tmp_path / "final.npz")
    monkeypatch.setattr(final_test, "RESULTS", str(tmp_path / "final.npz"))

    def never(*args):
        raise AssertionError("the final test ran a second time")
    monkeypatch.setattr(final_test, "run", never)
    final_test.main()
    assert "v2" in capsys.readouterr().out
