from experiments.aerial.eval.metrics import compute_sr_ne_spl


def test_perfect_run():
    m = compute_sr_ne_spl(
        successes=[True, True],
        path_lengths=[10.0, 20.0],
        shortest_lengths=[10.0, 20.0],
        nes=[0.0, 0.0],
    )
    assert m["SR"] == 1.0
    assert m["NE"] == 0.0
    assert abs(m["SPL"] - 1.0) < 1e-6


def test_failed_run_zero_spl_term():
    m = compute_sr_ne_spl(
        successes=[False],
        path_lengths=[50.0],
        shortest_lengths=[10.0],
        nes=[30.0],
    )
    assert m["SR"] == 0.0
    assert m["SPL"] == 0.0
