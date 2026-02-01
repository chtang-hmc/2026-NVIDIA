# ============================================================
# BF-DCQO (a.k.a. "Bias-field digitized counterdiabatic quantum optimization")
# for LABS, built to heavily reuse your existing DCQO/LABS notebook code.
#
# Drop this into ONE cell. It includes:
#   - LABS energy() (same as your notebook)
#   - get_interactions() (same as your notebook)
#   - your gate blocks (r_yz, r_zy, r_yzzz, r_zyzz, r_zzyz, r_zzzy, etc.)
#   - a bias-aware trotterized circuit (replaces h(reg) with per-qubit Ry init)
#   - BF-DCQO outer loop with CVaR-based magnetization update
#

import math
import numpy as np
import cudaq

# If this exists in your repo (as in the notebook), keep it:
#   theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
# Otherwise, you must supply an equivalent routine.
import tutorial_notebook.auxiliary_files.labs_utils as utils


# ----------------------------
# LABS objective (from notebook)
# ----------------------------
def energy(s: np.ndarray) -> np.ndarray:
    """LABS energy.

    E(s) = sum_{k=1..N-1} C_k(s)^2, where C_k = sum_{i=1..N-k} s_i s_{i+k}

    Supports:
      - s shape (N,)  -> returns scalar np.int64
      - s shape (k,N) -> returns (k,) energies
    """
    s = np.asarray(s)

    if s.ndim == 1:
        N = s.shape[0]
        e = np.int64(0)
        for shift in range(1, N):
            ck = int(np.dot(s[: N - shift], s[shift:]))
            e += np.int64(ck * ck)
        return e

    if s.ndim == 2:
        k_pop, N = s.shape
        e = np.zeros(k_pop, dtype=np.int64)
        for shift in range(1, N):
            ck = (s[:, : N - shift] * s[:, shift:]).sum(axis=1, dtype=np.int64)
            e += ck * ck
        return e

    raise ValueError("s must be 1D or 2D")


# ----------------------------
# Interaction sets (reused from phase 1)
# ----------------------------
def get_interactions(N: int) -> (list, list):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.
    """
    G2 = []
    G4 = []

    # --- Two-body terms ---
    for i in range(N - 2):
        max_k = (N - i) // 2
        for k in range(1, max_k + 1):
            G2.append([i, i + k])

    # --- Four-body terms ---
    for i in range(N - 3):
        max_t = (N - i - 1) // 2
        for t in range(1, max_t + 1):
            for k in range(t + 1, N - i - t):
                quad = [i, i + t, i + k, i + k + t]
                if quad[3] < N:
                    G4.append(quad)

    return G2, G4

# ----------------------------
# Bitstring conversion (reused from phase 1)
# ----------------------------
def bitstring_convert(bitstring: str) -> np.ndarray:
    """Convert '0/1' string to spins in {-1,+1}, with '1'->+1, '0'->-1."""
    return np.array([1 if b == '1' else -1 for b in bitstring])   # dtype=np.int8

# TODO: check, new
def counts_to_spin_matrix(counts) -> np.ndarray:
    """
    Expand cudaq.sample counts into a (nshots, N) spin matrix in {-1,+1}.
    This is convenient for CVaR selection / magnetization estimates.
    """
    spins = []
    for bitstring, c in counts.items():
        s = bitstring_convert(bitstring)
        for _ in range(int(c)):
            spins.append(s)
    if len(spins) == 0:
        raise RuntimeError("No shots returned by cudaq.sample.")
    return np.stack(spins, axis=0)


# ============================================================
# Gate blocks (copied from phase1)
# ============================================================
@cudaq.kernel
def r_zz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    cx(q0, q1)
    rz(theta, q1)
    cx(q0, q1)

@cudaq.kernel
def r_yz(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    # Y → Z on q0
    h(q0)
    s(q0)

    r_zz(q0, q1, 2.0 * theta)

    # Undo basis change
    s.adj(q0)
    h(q0)

@cudaq.kernel
def r_zy(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    # Y → Z on q1
    h(q1)
    s(q1)

    r_zz(q0, q1, 2.0 * theta)

    # Undo basis change
    s.adj(q1)
    h(q1)


@cudaq.kernel
def r_zzzz(q0: cudaq.qubit,
          q1: cudaq.qubit,
          q2: cudaq.qubit,
          q3: cudaq.qubit,
          theta: float):
    cx(q0, q1)
    cx(q1, q2)
    cx(q2, q3)

    rz(2.0 * theta, q3)

    cx(q2, q3)
    cx(q1, q2)
    cx(q0, q1)


@cudaq.kernel
def r_yzzz(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q0)
    h(q0)

    r_zzzz(q0, q1, q2, q3, theta)

    s.adj(q0)
    h(q0)


@cudaq.kernel
def r_zyzz(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q1)
    h(q1)

    r_zzzz(q0, q1, q2, q3, theta)

    s.adj(q1)
    h(q1)


@cudaq.kernel
def r_zzyz(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q2)
    h(q2)

    r_zzzz(q0, q1, q2, q3, theta)

    s.adj(q2)
    h(q2)


@cudaq.kernel
def r_zzzy(q0: cudaq.qubit,
           q1: cudaq.qubit,
           q2: cudaq.qubit,
           q3: cudaq.qubit,
           theta: float):
    s(q3)
    h(q3)
    r_zzzz(q0, q1, q2, q3, theta)
    s.adj(q3)
    h(q3)

# ============================================================
# BF-DCQO additions
# ============================================================
def bias_ground_state_angles(hb: np.ndarray, hx: float = 1.0) -> np.ndarray:
    """
    Prepare the product ground state of (hx * X - hb_j * Z) for each qubit
    as Ry(theta_j)|0>. A consistent choice is:
        theta_j = atan2(-hx, hb_j)
    """
    hb = np.asarray(hb, dtype=float)
    return np.arctan2(-hx * np.ones_like(hb), hb)


def cvar_subset(spins: np.ndarray, energies: np.ndarray, alpha: float) -> (np.ndarray, np.ndarray):
    """
    CVaR subset: keep lowest-energy ceil(alpha*nshots) shots.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    nshots = energies.shape[0]
    m = max(1, int(math.ceil(alpha * nshots)))
    idx = np.argsort(energies)[:m]
    return spins[idx], energies[idx]


def estimate_magnetization(spins_sub: np.ndarray) -> np.ndarray:
    """
    For Z-basis measurements with spins in {-1,+1}:
        <Z_j> = mean(spins[:, j])
    """
    return spins_sub.mean(axis=0)


def update_bias(exp_z: np.ndarray, mode: str = "unsigned", sign: str = "bias", kappa: float = 1.0) -> np.ndarray:
    """
    Bias update rule.
      mode: "unsigned" => hb_j =  ± <Z_j>
            "signed"   => hb_j =  ± sign(<Z_j>)
      sign: "bias"     => choose the '-' convention used in the paper (aligning with preferred orientation)
            "antibias" => '+'
      kappa: optional scaling (e.g., final iteration uses kappa=5)
    """
    if sign not in ("bias", "antibias"):
        raise ValueError("sign must be 'bias' or 'antibias'.")
    sgn = -1.0 if sign == "bias" else 1.0

    if mode == "unsigned":
        hb = sgn * exp_z
    elif mode == "signed":
        hb = sgn * np.sign(exp_z)
    else:
        raise ValueError("mode must be 'unsigned' or 'signed'.")

    return kappa * hb


# ============================================================
# Bias-aware DCQO circuit: same as your trotterized_circuit,
# but replace h(reg) with per-qubit Ry initialization.
# Optional theta_cutoff to skip tiny angles.
# ============================================================
@cudaq.kernel
def trotterized_circuit_bf(
    N: int,
    G2: list[list[int]],
    G4: list[list[int]],
    steps: int,
    dt: float,
    T: float,
    thetas: list[float],
    ry_init: list[float],
    theta_cutoff: float
):
    reg = cudaq.qvector(N)

    # --- BF-DCQO: biased initial state (instead of h(reg)) ---
    for i in range(N):
        ry(ry_init[i], reg[i])

    for s in range(steps):
        theta = thetas[s]

        # 2-body block
        for pair in G2:
            i = pair[0]
            j = pair[1]
            ang = -4.0 * theta
            if abs(ang) >= theta_cutoff:
                r_yz(reg[i], reg[j], ang)
                r_zy(reg[i], reg[j], ang)

        # 4-body block
        for quad in G4:
            a, b, c, d = quad[0], quad[1], quad[2], quad[3]
            ang = -8.0 * theta
            if abs(ang) >= theta_cutoff:
                r_yzzz(reg[a], reg[b], reg[c], reg[d], ang)
                r_zyzz(reg[a], reg[b], reg[c], reg[d], ang)
                r_zzyz(reg[a], reg[b], reg[c], reg[d], ang)
                r_zzzy(reg[a], reg[b], reg[c], reg[d], ang)


# ============================================================
# BF-DCQO main driver
# ============================================================
def bf_dcqo_labs(
    N: int,
    n_iters: int = 11,
    n_steps: int = 3,
    T: float = 1.0,
    nshots: int = 10_000,
    alpha: float = 0.01,              # CVaR fraction
    hx: float = 1.0,                  # transverse-field strength used in Ry init angle formula
    theta_cutoff: float = 0.0,        # gate pruning
    final_signed: bool = True,
    final_kappa: float = 5.0,         # stronger signed bias on the last iter
    bias_sign: str = "bias"           # "bias" or "antibias"
):
    """
    Returns:
      best_spin (N,) in {-1,+1}
      best_energy (int)
      history (list of dicts) with per-iter diagnostics
    """
    G2, G4 = get_interactions(N)
    dt = T / n_steps

    hb = np.zeros(N, dtype=float)  # bias fields start at 0

    best_E = None
    best_s = None
    history = []

    for it in range(n_iters):
        # --- compute DCQO thetas (reusing your notebook approach) ---
        thetas = []
        for step in range(1, n_steps + 1):
            t = step * dt
            theta_val = utils.compute_theta(t, dt, T, N, G2, G4)
            thetas.append(float(theta_val))

        # --- biased init angles ---
        ry_init = bias_ground_state_angles(hb, hx=hx).tolist()

        # --- sample circuit ---
        counts = cudaq.sample(
            trotterized_circuit_bf,
            N, G2, G4, n_steps, dt, T,
            thetas,
            ry_init,
            float(theta_cutoff),
            shots_count=int(nshots)
        )

        # --- evaluate energies ---
        spins = counts_to_spin_matrix(counts)         # (nshots, N)
        Es = energy(spins)                            # (nshots,)

        # track best-so-far
        i_best = int(np.argmin(Es))
        cur_best_E = int(Es[i_best])
        if best_E is None or cur_best_E < best_E:
            best_E = cur_best_E
            best_s = spins[i_best].copy()

        # --- CVaR subset for magnetization learning ---
        spins_sub, Es_sub = cvar_subset(spins, Es, alpha=alpha)
        exp_z = estimate_magnetization(spins_sub)

        # --- bias update ---
        is_last = (it == n_iters - 1)
        if final_signed and is_last:
            hb = update_bias(exp_z, mode="signed", sign=bias_sign, kappa=final_kappa)
        else:
            hb = update_bias(exp_z, mode="unsigned", sign=bias_sign, kappa=1.0)

        history.append({
            "iter": it + 1,
            "best_energy_seen": int(best_E),
            "cvar_mean_energy": float(np.mean(Es_sub)),
            "cvar_min_energy": int(np.min(Es_sub)),
            "hb": hb.copy(),
            "exp_z": exp_z.copy(),
        })

    return best_s, best_E, history
