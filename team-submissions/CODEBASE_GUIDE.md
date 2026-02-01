## Codebase guide (for judges)

This document is a “map” of the repository: what each folder/file is for, and which scripts are the main entrypoints for running solvers, tests, and benchmarks.

## High-level architecture (how the pieces connect)

- **Classical baseline**: CPU Memetic Tabu Search (MTS) implementation in `classical/mts.py`.
- **Quantum-enhanced variants**:
  - **QE-MTS**: `quantum/qe_mts.py` generates a quantum-sampled initial population (CUDA-Q trotterized circuit) then runs `classical.mts.MTS(...)`.
  - **BF-DCQO → MTS**: `quantum/bfdcqo.py` implements the BF-DCQO sampling loop (CUDA-Q) and then refines with `classical.mts.MTS(...)`.
- **GPU acceleration (classical)**:
  - A CUDA/C++ implementation builds a runnable binary `classical/labs_gpu` from `classical/labs_gpu.cu` + `classical/kernels.cu/.cuh` via `classical/Makefile`.
  - A prototype CUDA path in Python exists in `classical/mts_cuda_old.py` (Numba kernel).
- **Verification**: `tests/` contains unit/property tests and end-to-end regressions (see `team-submissions/TEST_SUITE-template.md`).
- **Benchmarks**: `benchmark/` contains scripts that produce CSV outputs (stored under `data/` in this repo snapshot).

## Repo layout

### `team-submissions/` (deliverables for grading)
- `team-submissions/README.md`: Deliverables checklist and grading rubric (copied from the challenge repo).
- `team-submissions/PRD.md`: Product Requirements Document (architecture choice, verification plan, success metrics).
- `team-submissions/TEST_SUITE-template.md`: Test suite documentation (how to run + what is covered).
- `team-submissions/AI_report.md`: AI usage + verification report.
- `team-submissions/CODEBASE_GUIDE.md`: This file.

### `classical/` (CPU MTS + GPU classical implementations)
- `classical/mts.py`
  - **Purpose**: CPU-optimized Memetic Tabu Search solver for LABS.
  - **Key functions**:
    - `energy(s)`: LABS objective \(E(s)=\sum_{k=1}^{N-1} C_k^2\), supports `(N,)` and `(k,N)`.
    - `tabu_search(s0, ...)`: local search using 1-bit flips + tabu list + aspiration.
    - `MTS(k, N, ...)`: memetic outer loop (population init/selection + crossover + mutation + tabu local search).
  - **Used by**: all benchmarks and all end-to-end pipelines (QE-MTS, BF-DCQO → MTS).

- `classical/labs_gpu.cu`
  - **Purpose**: CUDA/C++ implementation of a GPU-based LABS solver, intended as the accelerated classical path.
  - **Outputs**: prints best energy/sequence and timing info (parsed by `benchmark/gpu_benchmark.py`).

- `classical/kernels.cu` and `classical/kernels.cuh`
  - **Purpose**: CUDA kernel implementations and headers used by `labs_gpu.cu`.

- `classical/Makefile`
  - **Purpose**: Builds the CUDA/C++ solver binary.

- `classical/labs_gpu`
  - **Purpose**: Compiled binary (output of `make` in `classical/`).

### `quantum/` (CUDA-Q sampling + hybrid pipelines)
- `quantum/qe_mts.py`
  - **Purpose**: Implements the “QE-MTS” approach: use CUDA-Q circuit sampling to generate an initial population, then run classical MTS.
  - **Key pieces**:
    - CUDA-Q kernels for 2-qubit and 4-qubit interaction blocks (e.g., `r_zz`, `r_yz`, `r_zzzz`, ...).
    - `trotterized_circuit(...)`: the sampling circuit.
    - `get_interactions(N)`: constructs index sets for the 2-body and 4-body terms.
    - `quantum_population(...)`: returns a list of spin vectors in \(\{-1,+1\}\).
    - `qe_mts(population)`: runs `classical.mts.MTS(...)` on a provided population.

- `quantum/bfdcqo.py`
  - **Purpose**: Implements BF-DCQO sampling and a complete “BF-DCQO → MTS” hybrid pipeline.
  - **Key entrypoints**:
    - `bf_dcqo_sampler(...)`: runs the iterative bias-field sampling loop (Algorithm-style implementation).
    - `quantum_enhanced_mts(...)`: end-to-end pipeline (sample with BF-DCQO, select a population, refine with MTS, return metrics dict).
  - **Note**: This module configures CUDA-Q target (`cudaq.set_target("nvidia", option="mgpu")`) at import time.

- `quantum/main.py`
  - **Purpose**: Minimal script that calls `quantum_enhanced_mts(...)` with hard-coded parameters (quick manual run).

### `benchmark/` (benchmark runners that write CSVs)
All benchmarks were run using qBraid vCPUs and/or H100 GPUs.

- `benchmark/cpu_benchmark.py`
  - **Purpose**: Runs CPU MTS across a range of \(N\), multiple runs, and writes a CSV row per run/N.
  - **Uses**: `classical.mts.MTS(..., record_time=True)` to also record time-to-first-best.

- `benchmark/gpu_benchmark.py`
  - **Purpose**: Compiles (`make -C classical`) then runs the CUDA binary `classical/labs_gpu` for \(N\) in a loop.
  - **Behavior**: Parses stdout to extract best energy, timing, and sequence; writes `benchmark_results.csv`.

- `benchmark/qe_mts_benchmark.py`
  - **Purpose**: Benchmarks QE-MTS (quantum population generation + classical MTS refinement).
  - **CLI**: supports `--target` (CUDA-Q target), `--shots`, `--popsize`, MTS hyperparameters, and CSV output path.

- `benchmark/quantum_benchmark.py`
  - **Purpose**: Benchmarks BF-DCQO → MTS end-to-end (sampling time + MTS time + total time).
  - **CLI**: supports CUDA-Q target, BF-DCQO hyperparameters, MTS hyperparameters, and CSV output path.

### `tests/` (verification)
- `tests/test_mts.py`: unit/property tests for `classical.mts` (energy correctness, invariants, incremental updates, determinism).
- `tests/test_mts_cuda.py`: GPU smoke + parity tests (marked `gpu`, skipped by default via `pytest.ini`).
- `tests/test_qemts.py`: unit tests for `quantum.qe_mts` utilities; uses mocking for deterministic sampling.
- `tests/test_results.py`: end-to-end regression against a fixed list of “validation energies” for \(N=1..14\).
- `tests/validation_results.txt`: stored validation outputs referenced by the regression approach.

### `data/` (results + analysis)
- `data/*.csv`: benchmark output CSVs (CPU, GPU, QE-MTS, BF-DCQO).
- `data/analyze_benchmark.ipynb`: notebook used to analyze/plot benchmark CSV results.

### `tutorial_notebook/` (challenge tutorial + helper utilities)
- `tutorial_notebook/01_quantum_enhanced_optimization_LABS.ipynb`: official challenge tutorial notebook (Phase 1 baseline).
- `tutorial_notebook/auxiliary_files/labs_utils.py`: shared utility functions used by the tutorial and quantum sampling code (e.g., theta schedules).
- `tutorial_notebook/images/`: figures used in the tutorial notebook.

### Root-level docs/assets (challenge context)
- `README.md`: top-level project overview / how to run.
- `LABS-challenge-Phase1.md`, `LABS-challenge-Phase2.md`: challenge statement and environment instructions.
- `GPU_PIC_Brev.pdf`, `NVIDIA-presentation-slides-MIT-IQuHack.pdf`: provided challenge materials.
- `images/`: images referenced in challenge docs.

## Quick “judge runs” (recommended commands)

### Run the default unit tests (GPU tests skipped by default)

```bash
pytest
```

### Run GPU tests (only on CUDA machines)

```bash
pytest -m gpu
```

### Run CPU benchmark (writes CSV)

```bash
python benchmark/cpu_benchmark.py
```

### Run QE-MTS benchmark (requires CUDA-Q)

```bash
python benchmark/qe_mts_benchmark.py --n-min 1 --n-max 14 --runs 1
```

### Run BF-DCQO → MTS benchmark (requires CUDA-Q)

```bash
python benchmark/quantum_benchmark.py --n-min 1 --n-max 14 --runs 1
```

