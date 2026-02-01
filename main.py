"""
main.py

QE-MTS for LABS, where QE = BF-DCQO (bias-field digitized counterdiabatic quantum optimization).

Assumptions:
- The BF-DCQO/DCQO code from the previous cell exists in an importable module, e.g. `qe_bf_dcqo.py`,
  OR you can paste those definitions above this file's content.
- `MTS` is implemented and importable (memetic tabu search), signature:
      best_s, best_E, pop, eng, best_h = MTS(popsize, N, population0=population0)
- `cudaq` is installed and configured.

Usage:
  python main.py
"""

from __future__ import annotations

import numpy as np
import cudaq

from quantum.bfdcqo import (
    get_interactions,
    energy,
    bf_dcqo_labs,
)

from classical.mts import MTS

# ============================================================
# Helpers
# ============================================================

# LABS: generates spin {-1, 1} for bitstrings, not {0, 1}
# TODO: is it actually needed though?
def bitstring_convert(bitstring: str) -> np.ndarray:
    """
    Convert cudaq.sample output bitstring to spins in {-1, +1}.

    bitstring: the bitstring from a cudaq sample to convert
    """
    return np.array([1 if b == "1" else -1 for b in bitstring], dtype=np.int8)


def quantum_population(samples, popsize: int) -> list[np.ndarray]:
    """
    Generate the quantum-enhanced population from cudaq.sample counts.

    samples: the cudaq.sample result (counts dict-like: bitstring -> count)
    popsize: target population size
    """
    population: list[np.ndarray] = []
    for bitstring, count in samples.items():
        for _ in range(int(count)):
            population.append(bitstring_convert(bitstring))
            if len(population) >= popsize:
                return population
    return population


def random_population(popsize: int, N: int) -> list[np.ndarray]:
    """Generate random population in {-1,+1}."""
    return [np.random.choice([-1, 1], size=N).astype(np.int8) for _ in range(popsize)]


def bf_dcqo_population(
    N: int,
    popsize: int,
    *,
    n_iters: int,
    n_steps: int,
    T: float,
    nshots: int,
    alpha: float,
    theta_cutoff: float,
    final_signed: bool,
    final_kappa: float,
    bias_sign: str,
    hx: float = 1.0,
) -> tuple[list[np.ndarray], dict]:
    """
    Run BF-DCQO once and return a quantum-enhanced population for MTS.

    Strategy:
      - Run bf_dcqo_labs(...) (outer iterations + CVaR bias updates)
      - Build the MTS initial population by combining:
            (a) best solution found by BF-DCQO
            (b) additional diverse candidates sampled from a small random pool
                using bias-informed sampling (cheap and effective)
        This avoids needing bf_dcqo_labs to return all shots.

    Returns:
      population0: list of spins (len == popsize)
      info: dict with best_s, best_E, and history
    """
    best_s, best_E, hist = bf_dcqo_labs(
        N=N,
        n_iters=n_iters,
        n_steps=n_steps,
        T=T,
        nshots=nshots,
        alpha=alpha,
        hx=hx,
        theta_cutoff=theta_cutoff,
        final_signed=final_signed,
        final_kappa=final_kappa,
        bias_sign=bias_sign,
    )

    # Heavily reuse what you already do: pass a "quantum init population" into MTS.
    # Since bf_dcqo_labs returns the single best bitstring + history, we synthesize
    # the remaining pop from the learned bias in the final iteration's exp_z.
    # This matches the bias-field idea: draw candidates aligned with learned magnetizations.
    exp_z_final = hist[-1]["exp_z"]        # in [-1,1], learned from CVaR elite
    p_plus = 0.5 * (1.0 + exp_z_final)     # convert magnetization to Bernoulli for +1
    p_plus = np.clip(p_plus, 0.01, 0.99)   # avoid extremes for diversity

    population: list[np.ndarray] = [best_s.astype(np.int8)]
    while len(population) < popsize:
        # sample each spin independently as +1 with prob p_plus[j], else -1
        u = np.random.rand(N)
        s = np.where(u < p_plus, 1, -1).astype(np.int8)
        population.append(s)

    info = {"best_s": best_s, "best_E": best_E, "history": hist}
    return population[:popsize], info


# ============================================================
# Main: QE-MTS comparison (BF-DCQO init vs random init)
# ============================================================

def main():
    # ----------------------------
    # User parameters (keep in one place)
    # ----------------------------
    popsize = 100

    # Problem size
    N = 11

    # BF-DCQO / QE parameters
    T = 1.0
    n_steps = 3
    n_iters = 11
    nshots = 10_000
    alpha = 0.01
    theta_cutoff = 0.0
    hx = 1.0
    final_signed = True
    final_kappa = 5.0
    bias_sign = "bias"  # or "antibias"

    # ----------------------------
    # Build BF-DCQO (QE) population for MTS
    # ----------------------------
    quantum_init, qe_info = bf_dcqo_population(
        N=N,
        popsize=popsize,
        n_iters=n_iters,
        n_steps=n_steps,
        T=T,
        nshots=nshots,
        alpha=alpha,
        theta_cutoff=theta_cutoff,
        final_signed=final_signed,
        final_kappa=final_kappa,
        bias_sign=bias_sign,
        hx=hx,
    )

    # ----------------------------
    # Random init population
    # ----------------------------
    random_init = random_population(popsize, N)

    # ----------------------------
    # Run MTS
    # ----------------------------
    quantum_final = MTS(len(quantum_init), N, population0=quantum_init)
    random_final = MTS(len(random_init), N, population0=random_init)

    best_s_q, best_E_q, pop_q, eng_q, best_h_q = quantum_final
    best_s_r, best_E_r, pop_r, eng_r, best_h_r = random_final

    # ----------------------------
    # Report
    # ----------------------------
    print("\n===== QE-MTS (QE = BF-DCQO) vs Random-MTS =====\n")

    print("BF-DCQO best bitstring (pre-MTS):", qe_info["best_s"], "eng", qe_info["best_E"])
    print("Quantum Population best bitstring:", best_s_q, "eng", best_E_q)
    print("Randomized Population best bitstring:", best_s_r, "eng", best_E_r)

    print("\nQuantum Population best energy:", best_E_q)
    print("Randomized Population best energy:", best_E_r)

    print("\nQuantum Population mean energy:", float(np.mean(eng_q)))
    print("Randomized Population mean energy:", float(np.mean(eng_r)))

    # Optional: sanity check energies with LABS evaluator
    # (only if pop arrays are spins in {-1,+1})
    try:
        print("\nSanity check energy(best_s_q):", int(energy(np.asarray(best_s_q))))
        print("Sanity check energy(best_s_r):", int(energy(np.asarray(best_s_r))))
    except Exception:
        pass


if __name__ == "__main__":
    main()
