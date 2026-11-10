"""The joint outcome of one instance in llmdw.scoring.first_sets: the four shares of the first sets of a language model
and of a lookahead order, from the samples of the language model (0 or 1) and the lookahead order's hit@1."""
from llmdw.scoring.first_sets import joint_shares


def test_joint_shares_of_one_instance():
    c = joint_shares([1.0, 0.0, 1.0], 0.5)
    assert c == {"both": 1 / 3, "only_lm": 1 / 3, "only_la": 1 / 6, "neither": 1 / 6}
    assert abs(sum(c.values()) - 1) < 1e-12
    assert joint_shares([0.0, 0.0, 0.0], 1.0) == {"both": 0.0, "only_lm": 0.0, "only_la": 1.0, "neither": 0.0}
    assert joint_shares([1.0], 0.0) == {"both": 0.0, "only_lm": 1.0, "only_la": 0.0, "neither": 0.0}
