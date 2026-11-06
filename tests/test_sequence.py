import numpy as np
import torch

from src import features, sequence
from src.match import A, D, H, X
from tests.test_model import COSTS, play

QUICK = dict(epochs=2, window=64, batch=8)


def test_sequences_give_one_contiguous_block_per_half_with_x_where_ungraded():
    m = play(2000)
    m.half[900:1000] = 0                      # a separator in the middle
    m.half[1000:] = 2
    blocks = sequence.sequences(m)
    assert [b.half for b in blocks] == [1, 2]
    for b in blocks:
        assert len(b.x) == len(b.labels) == len(b.near) == len(b.index)
        assert b.x.shape[1] == len(features.NAMES)
    assert (blocks[0].labels != X).all()


def test_network_returns_one_row_of_logits_per_step_for_any_length():
    net = sequence.Net(n_in=5, n_out=3, hidden=8, dilations=(1, 2))
    for n in (1, 7, 300):
        assert net(torch.zeros(2, n, 5)).shape == (2, n, 3)


def test_weights_round_trip_through_a_plain_npz(tmp_path):
    torch.manual_seed(0)
    net = sequence.Net(n_in=5, n_out=3, hidden=8, dilations=(1, 2))
    scale = sequence.Scale(np.arange(5.0), np.ones(5))
    trained = sequence.Trained(net, scale, columns=list(range(5)))
    trained.save(tmp_path / "w.npz")
    back = sequence.Trained.load(tmp_path / "w.npz")
    x = np.random.default_rng(0).normal(size=(40, 5)).astype(np.float32)
    np.testing.assert_array_equal(trained.logits(x), back.logits(x))
    assert back.columns == list(range(5))


def test_fit_is_deterministic_and_predicts_a_label_for_every_frame():
    matches = {"a": play(2000, 1), "b": play(2000, 2)}
    one = sequence.fit(matches, copies=[lambda m: m], threads=1, **QUICK)
    two = sequence.fit(matches, copies=[lambda m: m], threads=1, **QUICK)
    m = play(1500, 3)
    codes = one.predict(m, COSTS)
    assert codes.shape == m.labels.shape and set(np.unique(codes)) <= {H, A, D, X}
    np.testing.assert_array_equal(codes, two.predict(m, COSTS))


def test_the_tree_heads_run_beside_the_network_and_blending_averages_them():
    matches = {"a": play(2000, 1), "b": play(2000, 2)}
    trained = sequence.fit(matches, copies=[lambda m: m], threads=1, dead="trees", blend=True,
                           **QUICK)
    m = play(1500, 3)
    assert set(np.unique(trained.predict(m, COSTS))) <= {H, A, D, X}
    x = sequence.sequences(m)[0].x
    net, trees = (live.proba(x) for live in trained.lives)
    assert len(trained.lives) == 2 and net.shape == trees.shape == (len(x), 2)
    assert not np.allclose(net, trees)


def test_several_seeds_count_as_one_head_whose_chances_are_averaged():
    matches = {"a": play(2000, 1), "b": play(2000, 2)}
    trained = sequence.fit(matches, copies=[lambda m: m], threads=1, nets=2, **QUICK)
    x = sequence.sequences(play(1500, 3))[0].x
    head, = trained.lives
    one, two = (net.proba(x) for net in head.heads)
    assert len(head.heads) == 2 and not np.allclose(one, two)
    np.testing.assert_allclose(head.proba(x), (one + two) / 2)


def test_a_trained_network_beats_chance_on_easy_invented_play():
    matches = {str(i): play(3000, i) for i in range(4)}
    trained = sequence.fit(matches, copies=[lambda m: m], threads=1, epochs=6, window=128,
                           batch=8)
    m = play(3000, 9)
    truth = m.labels
    codes = trained.predict(m, COSTS)
    assert (codes == truth).mean() > 0.6
