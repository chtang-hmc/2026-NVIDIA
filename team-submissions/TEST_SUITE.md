# Test Suite (Classical MTS / QE-MTS / BF-DCQO / GPU parity)

This document describes how we verify the correctness and reliability of our LABS solvers, based on the **PRD verification goals** and the **tests that actually exist in `tests/`**.

## Verification goals (Definition of Done)
We consider the solver “verified” when it satisfies:

- **Correctness**: LABS energy computation and incremental tabu updates are correct; solvers return valid \(\{-1,+1\}\) sequences; reported energies match independent recomputation.
- **Reproducibility**: CPU reference (`classical.mts`) is deterministic when a seed is fixed.
- **GPU/CPU parity (where applicable)**: GPU kernels preserve invariants (valid bits, energy consistency) and match CPU emulation on small deterministic cases.

## How to run the tests

### Default (fast) test run
From the repo root:

```bash
pytest
```

Notes:
- `pytest.ini` sets `addopts = -q -m "not gpu"`, so **GPU-marked tests are skipped by default**.

### Run only the classical (CPU) unit tests

```bash
pytest tests/test_mts.py
```

### Run GPU tests (requires CUDA + numba)

```bash
pytest -m gpu
```

### Run only the quantum module unit tests (requires CUDA-Q / `cudaq`)

```bash
pytest tests/test_qemts.py
```

### Run the end-to-end “validation against benchmark results” regression (slow)
This runs classical MTS, QE-MTS seeding, and BF-DCQO + MTS for \(N=1..14\) and compares energies to a known list from CPU benchmarking.

```bash
pytest tests/test_results.py
```

## Test inventory (what we test and where)

### `tests/test_mts.py` — classical correctness + determinism (CPU reference)
These tests target `classical.mts` and focus on correctness properties that are most likely to break during refactors/GPU ports:

- **Energy correctness (LABS objective)**:
  - Known hand-checkable sequence: all-ones length 4 has energy \(14\).
  - \(N=1\) edge case returns energy \(0\).
  - Scalar vs batch consistency: `energy((N,))` matches per-row of `energy((k,N))`.
  - Input validation: invalid ranks are rejected.

- **Gauge symmetries / invariants** (property-based correctness checks):
  - Global inversion: \(E(s)=E(-s)\)
  - Reversal: \(E(s)=E(s[::-1])\)
  - Alternating inversion: \(s_i \rightarrow (-1)^i s_i\) (1-based indexing)
  - Exhaustive verification of invariants for a small \(N\) over all \(2^N\) sequences.

- **Genetic operators**:
  - `combine(...)` produces a prefix from parent 1 and suffix from parent 2 with cut in \([1, N-1]\), and does not mutate parents.
  - `mutate_inplace(...)` obeys probability extremes (`p_mut=0` no-op, `p_mut=1` flips all bits) and only mutates its argument.

- **Tabu search incremental-update math**:
  - `_all_deltas(...)` matches full recomputation for every single-bit flip, and does not mutate inputs.
  - `_apply_flip_in_place(...)` keeps cached autocorrelation `C` consistent with full recomputation; round-trip (flip twice) is a no-op.

- **Search behavior / reliability**:
  - `tabu_search(...)` returns a best energy that is never worse than the starting energy and does not mutate the input.
  - `MTS(...)` is deterministic given a fixed seed.
  - `MTS(population0=..., max_iter=0)` returns the correct initial best, and matches exhaustive optimum when `population0` contains all sequences.

### `tests/test_mts_cuda.py` — GPU smoke + GPU/CPU parity (skipped by default)
These tests are marked with `@pytest.mark.gpu` and are skipped by default (see `pytest.ini`).

- **GPU/CPU parity**:
  - Emulates the kernel logic on CPU and checks exact match for a small case (`N=32`, `max_steps=15`) with 1 walker.

- **GPU smoke**:
  - Confirms GPU returns a valid \(\{-1,+1\}\) sequence and that the returned energy matches `energy(s)` recomputed on CPU.

Important practical note:
- The GPU tests currently import `mts_cuda` and `mts` as top-level modules. In this repo the CPU reference implementation is `classical.mts`, and the current CUDA prototype is in `classical/mts_cuda_old.py`. If you do not have root-level wrappers/modules named `mts.py` and `mts_cuda.py`, GPU tests will fail on import even if CUDA is available.

### `tests/test_qemts.py` — quantum-population utilities (CUDA-Q)
These tests verify the `quantum.qe_mts` building blocks:

- `get_interactions(N)` returns correctly-shaped 2-body and 4-body interaction index lists with valid ranges.
- `bitstring_convert(...)` maps CUDA-Q bitstrings to spins in \(\{-1,+1\}\).
- `quantum_population(...)` respects `popsize` and yields spin arrays of shape `(N,)`.
  - Uses mocking (`unittest.mock.patch`) to make sampling deterministic for unit testing.
- `qe_mts(population)` runs and returns an MTS-style tuple for a valid population.

### `tests/test_results.py` — end-to-end regression vs benchmark energies
These tests compare solver outputs against `VALIDATION_RESULTS` for \(N=1..14\):

- **Classical MTS regression**: `classical.mts.MTS(...)` energies match the recorded list.
- **QE-MTS regression**: seeds MTS from `quantum.qe_mts.quantum_population(...)` and checks energies match the recorded list.
- **BF-DCQO regression**: runs `quantum.bfdcqo.quantum_enhanced_mts(...)` and checks energies match the recorded list.

## Traceability to PRD verification plan
This test suite directly supports the PRD verification goals:

- **Correctness**:
  - Objective function correctness + invariants: `tests/test_mts.py`
  - Quantum population validity: `tests/test_qemts.py`
  - End-to-end agreement with known results: `tests/test_results.py`

- **Reproducibility**:
  - Deterministic seeded CPU behavior: `tests/test_mts.py::test_mts_deterministic_with_seed`

- **GPU/CPU parity**:
  - Deterministic parity + smoke: `tests/test_mts_cuda.py` (opt-in via `pytest -m gpu`)

## Expected environments / prerequisites
- **CPU-only quick verification**: Python + `pytest` + `numpy` is sufficient for `tests/test_mts.py`.
- **Quantum tests**: require CUDA-Q Python package (`cudaq`) available (e.g., qBraid CUDA-Q environment).
- **GPU tests**: require `numba` with `numba.cuda.is_available() == True` and the expected GPU entrypoint modules importable.
