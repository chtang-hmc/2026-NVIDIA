import argparse
import csv
import io
import signal
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Keep imports robust when running from anywhere (mirrors cpu_benchmark.py/tests).
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import classical.mts as mts  # noqa: E402


class _Timeout(Exception):
    pass


def _timeout_handler(_signum, _frame):
    raise _Timeout()


@dataclass(frozen=True)
class BFDCQOParams:
    theta_cutoff: float = 0.06
    bf_dcqo_iter: int = 3
    quantum_shots: int = 1000
    alpha: float = 0.01
    kappa: float = 5.0
    T: float = 1.0
    n_steps: int = 100
    target: Optional[str] = None  # CUDA-Q target name (optional)
    quiet: bool = True  # suppress BF-DCQO iteration prints


@dataclass(frozen=True)
class MTSParams:
    k: int = 12
    max_iter: int = 1000
    p_sample: float = 0.5
    p_mutate: float = 0.02
    tabu_steps: int = 60
    tabu_tenure: int = 10
    seed: Optional[int] = 0


def _seq_pm1_to_pm(s: np.ndarray) -> str:
    # + for +1, - for -1 (CPU/GPU benchmark formatting).
    s = np.asarray(s, dtype=np.int8)
    return "".join("+" if int(x) == 1 else "-" for x in s.tolist())


def _merit_factor(N: int, E: int) -> Optional[float]:
    # LABS merit factor: F = N^2 / (2E). Undefined at E=0.
    if E <= 0:
        return None
    return (N * N) / (2.0 * float(E))


def _maybe_set_cudaq_target(target: Optional[str]) -> None:
    if not target:
        return
    try:
        import cudaq  # type: ignore  # pylint: disable=import-error
    except Exception as e:
        raise RuntimeError(
            f"CUDA-Q is required to set target={target!r}, but import failed: {e}"
        ) from e
    cudaq.set_target(target)


def _quantum_seed(run_idx: int, N: int, base_seed: Optional[int]) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + 10_000 * int(run_idx) + int(N)


def _run_one(
    N: int, run_idx: int, qparams: BFDCQOParams, mparams: MTSParams
) -> Tuple[int, float, float, float, Optional[float], int, str]:
    """
    Returns:
      best_E, q_sample_time_s, mts_conv_time_s, total_time_s, merit_factor, generations, sequence
    """
    # Import here so the benchmark script still imports even if CUDA-Q isn't installed.
    try:
        from quantum.bfdcqo import bf_dcqo_sampler  # noqa: WPS433
    except Exception as e:
        raise RuntimeError(
            "Failed to import quantum.bfdcqo. Ensure CUDA-Q and dependencies are installed."
        ) from e

    seed = _quantum_seed(run_idx=run_idx, N=N, base_seed=mparams.seed)

    # (1) BF-DCQO sampling
    t_total0 = time.perf_counter()
    t_q0 = time.perf_counter()

    if qparams.quiet:
        with redirect_stdout(io.StringIO()):
            quantum_samples, _energy_hist = bf_dcqo_sampler(
                N=int(N),
                n_iter=int(qparams.bf_dcqo_iter),
                n_shots=int(qparams.quantum_shots),
                alpha=float(qparams.alpha),
                kappa=float(qparams.kappa),
                T=float(qparams.T),
                n_steps=int(qparams.n_steps),
                theta_cutoff=float(qparams.theta_cutoff),
            )
    else:
        quantum_samples, _energy_hist = bf_dcqo_sampler(
            N=int(N),
            n_iter=int(qparams.bf_dcqo_iter),
            n_shots=int(qparams.quantum_shots),
            alpha=float(qparams.alpha),
            kappa=float(qparams.kappa),
            T=float(qparams.T),
            n_steps=int(qparams.n_steps),
            theta_cutoff=float(qparams.theta_cutoff),
        )

    quantum_samples = np.asarray(quantum_samples, dtype=np.int8)
    if quantum_samples.ndim != 2 or quantum_samples.shape[1] != int(N):
        raise ValueError(
            f"bf_dcqo_sampler returned invalid sample shape {quantum_samples.shape}, expected (shots, {N})"
        )

    # Select best k samples by energy (vectorized).
    k = min(int(mparams.k), int(quantum_samples.shape[0]))
    Es = mts.energy(quantum_samples).astype(np.int64)
    elite_idx = np.argsort(Es)[:k]
    population0 = quantum_samples[elite_idx]

    t_q1 = time.perf_counter()

    # (2) MTS refinement seeded by BF-DCQO population
    best_s, best_E, _pop, _Es, hist, conv_time = mts.MTS(
        k=int(population0.shape[0]),
        N=int(N),
        target=0,
        max_iter=int(mparams.max_iter),
        p_sample=float(mparams.p_sample),
        p_mutate=float(mparams.p_mutate),
        tabu_steps=int(mparams.tabu_steps),
        tabu_tenure=int(mparams.tabu_tenure),
        population0=population0,
        seed=seed,
        record_time=True,
    )

    t_total1 = time.perf_counter()

    best_E = int(best_E)
    generations = max(0, int(len(hist)) - 1)
    seq = _seq_pm1_to_pm(best_s)
    merit = _merit_factor(N, best_E)

    q_sample_time_s = float(t_q1 - t_q0)
    mts_conv_time_s = float(conv_time) if conv_time is not None else float("nan")
    total_time_s = float(t_total1 - t_total0)

    return best_E, q_sample_time_s, mts_conv_time_s, total_time_s, merit, generations, seq


def run_benchmark(
    N_min: int = 1,
    N_max_inclusive: int = 39,
    runs: int = 25,
    csv_filename: str = "quantum_benchmark_results.csv",
    timeout_s: float = 180.0,
    qparams: Optional[BFDCQOParams] = None,
    mparams: Optional[MTSParams] = None,
) -> None:
    qparams = qparams or BFDCQOParams()
    mparams = mparams or MTSParams()

    # Configure CUDA-Q target (optional).
    _maybe_set_cudaq_target(qparams.target)

    print("Running Quantum benchmark (BF-DCQO → MTS)...")
    if qparams.target:
        print(f"CUDA-Q target: {qparams.target}")
    print(
        f"{'Run':<5} {'N':<5} {'Shots':<7} {'k':<4} {'Best E':<10} {'Q Sample':<12} {'MTS Conv':<12} {'Total Time':<12} {'Merit F.':<10} {'Generations':<12} {'Sequence':<20}"
    )
    print("-" * 120)

    print(f"Saving results to {csv_filename}")
    with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Run",
                "N",
                "CUDA-Q Target",
                "Shots",
                "BF-DCQO Iters",
                "Theta Cutoff",
                "Alpha",
                "Kappa",
                "T",
                "Trotter Steps",
                "Population (k)",
                "MTS Max Iter",
                "MTS p_sample",
                "MTS p_mutate",
                "MTS tabu_steps",
                "MTS tabu_tenure",
                "Best E",
                "Q Sample Time",
                "MTS Conv. Time",
                "Total Time",
                "Merit F.",
                "Generations",
                "Sequence",
            ]
        )

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            for run_idx in range(1, int(runs) + 1):
                for N in range(int(N_min), int(N_max_inclusive) + 1):
                    try:
                        if timeout_s and timeout_s > 0:
                            signal.setitimer(signal.ITIMER_REAL, float(timeout_s))

                        best_E, sample_s, conv_s, total_s, merit, generations, seq = (
                            _run_one(N, run_idx, qparams, mparams)
                        )

                        merit_str = "N/A" if merit is None else f"{merit:.6g}"
                        sample_str = f"{sample_s:.6g}"
                        conv_str = f"{conv_s:.12f}"
                        total_str = f"{total_s:.6g}"

                        print(
                            f"{run_idx:<5} {N:<5} {int(qparams.quantum_shots):<7} {int(mparams.k):<4} {best_E:<10} {sample_str:<12} {conv_str:<12} {total_str:<12} {merit_str:<10} {generations:<12} {seq:<20}"
                        )
                        writer.writerow(
                            [
                                run_idx,
                                N,
                                qparams.target or "",
                                int(qparams.quantum_shots),
                                int(qparams.bf_dcqo_iter),
                                float(qparams.theta_cutoff),
                                float(qparams.alpha),
                                float(qparams.kappa),
                                float(qparams.T),
                                int(qparams.n_steps),
                                int(mparams.k),
                                int(mparams.max_iter),
                                float(mparams.p_sample),
                                float(mparams.p_mutate),
                                int(mparams.tabu_steps),
                                int(mparams.tabu_tenure),
                                best_E,
                                sample_str,
                                conv_str,
                                total_str,
                                merit_str,
                                generations,
                                seq,
                            ]
                        )
                        csv_file.flush()

                    except _Timeout:
                        print(f"{run_idx:<5} {N:<5} TIMEOUT")
                        writer.writerow(
                            [
                                run_idx,
                                N,
                                qparams.target or "",
                                int(qparams.quantum_shots),
                                int(qparams.bf_dcqo_iter),
                                float(qparams.theta_cutoff),
                                float(qparams.alpha),
                                float(qparams.kappa),
                                float(qparams.T),
                                int(qparams.n_steps),
                                int(mparams.k),
                                int(mparams.max_iter),
                                float(mparams.p_sample),
                                float(mparams.p_mutate),
                                int(mparams.tabu_steps),
                                int(mparams.tabu_tenure),
                                "TIMEOUT",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "",
                            ]
                        )
                        csv_file.flush()
                    except Exception as e:
                        print(f"{run_idx:<5} {N:<5} ERROR: {e}")
                        writer.writerow(
                            [
                                run_idx,
                                N,
                                qparams.target or "",
                                int(qparams.quantum_shots),
                                int(qparams.bf_dcqo_iter),
                                float(qparams.theta_cutoff),
                                float(qparams.alpha),
                                float(qparams.kappa),
                                float(qparams.T),
                                int(qparams.n_steps),
                                int(mparams.k),
                                int(mparams.max_iter),
                                float(mparams.p_sample),
                                float(mparams.p_mutate),
                                int(mparams.tabu_steps),
                                int(mparams.tabu_tenure),
                                "ERROR",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "N/A",
                                "",
                            ]
                        )
                        csv_file.flush()
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, old_handler)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BF-DCQO → MTS quantum benchmark suite.")
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--n-max", type=int, default=39)
    p.add_argument("--runs", type=int, default=25)
    p.add_argument("--csv", type=str, default="quantum_benchmark_results.csv")
    p.add_argument("--timeout-s", type=float, default=180.0)

    p.add_argument("--target", type=str, default=None, help="CUDA-Q target name")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--bf-iters", type=int, default=3)
    p.add_argument("--theta-cutoff", type=float, default=0.06)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--kappa", type=float, default=5.0)
    p.add_argument("--t", type=float, default=1.0, help="Total evolution time T")
    p.add_argument("--n-steps", type=int, default=100, help="Trotter steps")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show BF-DCQO iteration logs (otherwise suppressed).",
    )

    p.add_argument("--k", type=int, default=12, help="MTS population size")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--p-sample", type=float, default=0.5)
    p.add_argument("--p-mutate", type=float, default=0.02)
    p.add_argument("--tabu-steps", type=int, default=60)
    p.add_argument("--tabu-tenure", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-seed", action="store_true", help="Disable deterministic seeding"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    qparams_ = BFDCQOParams(
        theta_cutoff=args.theta_cutoff,
        bf_dcqo_iter=args.bf_iters,
        quantum_shots=args.shots,
        alpha=args.alpha,
        kappa=args.kappa,
        T=args.t,
        n_steps=args.n_steps,
        target=args.target,
        quiet=not args.verbose,
    )
    mparams_ = MTSParams(
        k=args.k,
        max_iter=args.max_iter,
        p_sample=args.p_sample,
        p_mutate=args.p_mutate,
        tabu_steps=args.tabu_steps,
        tabu_tenure=args.tabu_tenure,
        seed=None if args.no_seed else args.seed,
    )

    run_benchmark(
        N_min=args.n_min,
        N_max_inclusive=args.n_max,
        runs=args.runs,
        csv_filename=args.csv,
        timeout_s=args.timeout_s,
        qparams=qparams_,
        mparams=mparams_,
    )
