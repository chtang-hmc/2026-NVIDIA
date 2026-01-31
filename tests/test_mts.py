import numpy as np


import sys
from pathlib import Path

# Ensure repo root is importable when running `pytest` from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import mts


def _flip(s: np.ndarray, j: int) -> np.ndarray:
    s2 = np.asarray(s, dtype=np.int8).copy()
    s2[j] *= -1
    return s2


def test_generate_bitstrings_shape_and_values():
    pop = mts.generate_bitstrings(k=7, N=13, seed=123)
    assert pop.shape == (7, 13)
    assert pop.dtype == np.int8
    assert set(np.unique(pop)).issubset({-1, 1})


def test_energy_single_vs_batch_consistency():
    rng = np.random.default_rng(0)
    pop = rng.choice(np.array([-1, 1], dtype=np.int8), size=(5, 9))
    Es = mts.energy(pop)
    assert Es.shape == (5,)
    for i in range(pop.shape[0]):
        assert int(mts.energy(pop[i])) == int(Es[i])


def test_energy_invariants_global_flip_and_reverse():
    rng = np.random.default_rng(1)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(17,))

    e = int(mts.energy(s))
    assert int(mts.energy(-s)) == e  # global spin flip
    assert int(mts.energy(s[::-1])) == e  # time reversal / sequence reversal


def test_energy_known_small_case():
    # s = [+1, +1, +1, +1]
    # C1=3, C2=2, C3=1 -> E=9+4+1=14
    s = np.ones(4, dtype=np.int8)
    assert int(mts.energy(s)) == 14


def test_combine_produces_prefix_suffix_from_parents():
    rng = np.random.default_rng(42)
    p1 = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
    p2 = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int8)
    child = mts.combine(p1, p2, rng)

    # Find the crossover point by locating first index where it differs from p1.
    # Because parents are all-ones vs all-minus-ones, this is unambiguous.
    cut = int(np.argmax(child != p1))
    # If cut==0 then it was all from p2, which should never happen (cut in [1..N-1])
    assert 1 <= cut <= len(p1) - 1
    assert np.all(child[:cut] == p1[:cut])
    assert np.all(child[cut:] == p2[cut:])


def test_mutate_probability_extremes():
    rng = np.random.default_rng(0)
    s = np.array([1, -1, 1, -1, 1], dtype=np.int8)

    # p_mut=0 -> unchanged (by content)
    out0 = mts.mutate(s, p_mut=0.0, rng=rng)
    assert np.array_equal(out0, s)

    # p_mut=1 -> all bits flipped
    out1 = mts.mutate(s, p_mut=1.0, rng=rng)
    assert np.array_equal(out1, -s)


def test_delta_energy_matches_full_recompute_for_all_flips():
    rng = np.random.default_rng(3)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(14,))
    C = mts._autocorr_vector(s)
    E = int(np.sum(C[1:] * C[1:], dtype=np.int64))
    assert E == int(mts.energy(s))

    for j in range(len(s)):
        dE = mts._delta_energy_for_flip(s, C, j)
        s2 = _flip(s, j)
        assert int(mts.energy(s2)) == E + int(dE)


def test_apply_flip_in_place_consistent_with_recompute():
    rng = np.random.default_rng(4)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(11,))
    C = mts._autocorr_vector(s)

    for j in [0, 3, 7, 10]:
        s_expected = _flip(s, j)
        C_expected = mts._autocorr_vector(s_expected)

        mts._apply_flip_in_place(s, C, j)
        assert np.array_equal(s, s_expected)
        assert np.array_equal(C, C_expected)


def test_tabu_search_returns_nonworse_best():
    rng = np.random.default_rng(5)
    s0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(18,))
    e0 = int(mts.energy(s0))

    best_s, best_E = mts.tabu_search(s0, max_steps=50, seed=0)
    assert int(best_E) <= e0
    assert int(mts.energy(best_s)) == int(best_E)


def test_mts_deterministic_with_seed():
    out1 = mts.MTS(k=12, N=16, max_iter=60, seed=123)
    out2 = mts.MTS(k=12, N=16, max_iter=60, seed=123)
    # Compare best energy and best string deterministically.
    assert int(out1[1]) == int(out2[1])
    assert np.array_equal(out1[0], out2[0])


def test_mts_population0_used_when_max_iter_zero():
    rng = np.random.default_rng(7)
    pop0 = rng.choice(np.array([-1, 1], dtype=np.int8), size=(9, 10))
    Es0 = mts.energy(pop0)
    expected_best = int(Es0.min())

    best_s, best_E, pop, Es, hist = mts.MTS(k=999, N=999, population0=pop0, max_iter=0, seed=0)
    assert int(best_E) == expected_best
    assert pop.shape == pop0.shape
    assert np.array_equal(pop, pop0.astype(np.int8))
    assert np.array_equal(Es, Es0.astype(np.int64))
    assert hist == [expected_best]

