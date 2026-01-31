## Verification strategy (MTS / LABS)

Goal: ensure our Memetic Tabu Search (MTS) implementation is **correct, deterministic when seeded, and internally consistent** before accelerating with CuPy/Numba.

### What we test (and why)

- **Energy correctness (LABS objective)**:
  - Known small-case energy matches hand calculation (e.g. all-ones length 4 has energy 14).
  - Batch energy (`(k,N)`) matches single energy (`(N,)`) per-row.

- **Known invariants / symmetries**:
  - Global spin flip: \(E(s) = E(-s)\).
  - Sequence reversal: \(E(s) = E(s[::-1])\).
  These invariants are easy to violate if indexing is wrong.

- **Tabu-search internal math**:
  - `_delta_energy_for_flip` matches full recomputation after flipping any single bit.
  - `_apply_flip_in_place` keeps the cached autocorrelation vector consistent with recomputation.

- **Search behavior / reliability**:
  - `tabu_search()` returns a best solution that is **never worse** than the starting solution.
  - `MTS()` is **deterministic given a seed**.
  - `MTS(population0=...)` uses the provided population and returns the correct initial best when `max_iter=0`.

### Where the tests live

- Unit tests: `tests/test_mts.py`
- Pytest config: `pytest.ini`

### How to run

From the repo root:

```bash
pytest
```

If you’re running in an environment without pytest installed:

```bash
python -m pip install pytest
pytest
```

### Coverage philosophy

We prioritized tests that:
- Prove the LABS objective is implemented correctly.
- Validate the incremental-update logic (most likely to break during GPU/Numba refactors).
- Ensure reproducibility (critical for benchmarking GPU speedups later).
