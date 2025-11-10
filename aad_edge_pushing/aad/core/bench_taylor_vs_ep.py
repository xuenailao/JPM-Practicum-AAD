# aad/core/bench_taylor_vs_ep.py
"""
Benchmark: Taylor Expansion (Backprop) vs Edge-Pushing (baseline) vs FoR vs Finite Difference (FD)

- Random n-D synthetic function (no trig)
- Black–Scholes 5D: S, sigma, K, r, T
- Composite nested function (no trig)

Output style (concise):
- Speed (ms/run): EP baseline, then Taylor/FoR/FD with speed-up vs EP (x>1 => faster than EP)
- Accuracy: relative errors w.r.t. EP (Taylor vs EP, FoR vs EP, FD vs EP)
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt  # (kept for compatibility with your notebook)

from aad.core.engine import hvp_for, edge_push_hessian
from aad.core.tape import use_tape
from aad.core.var import ADVar
from aad.core.taylor_backprop import taylor_backpropagate

# Ops used to build the graph (AD-aware primitives)
from aad.ops import exp as oexp, log as olog, sqrt as osqrt, norm_cdf as onorm_cdf


# =========================
# Utilities
# =========================
def relerr(a, b):
    """Relative matrix error: ||a-b|| / max(1, ||b||)."""
    a, b = np.asarray(a), np.asarray(b)
    denom = max(1.0, np.linalg.norm(b))
    return np.linalg.norm(a - b) / denom


def hess_for(f_edgepush, inputs):
    """
    Rebuild full Hessian by n FoR HVP sweeps (Forward-over-Reverse).
    `f_edgepush`: callable that builds the AD graph from a dict of ADVars.
    `inputs`: dict of numeric values.
    """
    keys = list(inputs.keys())
    cols = []
    for k in keys:
        e = {kk: 0.0 for kk in keys}
        e[k] = 1.0
        hv = hvp_for(f_edgepush, inputs, e)
        cols.append(hv)
    return np.column_stack(cols)


def hess_taylor_backprop(f_edgepush, inputs):
    """
    Full Hessian via Taylor Backprop (single-pass, O(1) lookup after run).

    We reuse the same graph builder `f_edgepush` (ops-based), but run a tape
    and call `taylor_backpropagate` once to obtain grad & Hessian.
    """
    keys = list(inputs.keys())
    with use_tape():
        # Build AD variables in a deterministic order
        dvars = {k: ADVar(float(inputs[k]), requires_grad=True, name=k) for k in keys}
        y = f_edgepush(dvars)  # scalar output
        grad, H = taylor_backpropagate([dvars[k] for k in keys])
    return grad, H


def fd_grad_hess_scalar(func_scalar, x, eps=1e-6):
    """
    Central finite-difference gradient and Hessian for a scalar function.
    func_scalar: R^n -> R, purely numeric
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    fx = func_scalar(x)

    # gradient
    g = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        g[i] = (func_scalar(x + dx) - func_scalar(x - dx)) / (2 * eps)

    # Hessian
    H = np.zeros((n, n))
    # diagonals
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        f_pp = func_scalar(x + dx)
        f_mm = func_scalar(x - dx)
        H[i, i] = (f_pp - 2 * fx + f_mm) / (eps ** 2)
    # off-diagonals
    for i in range(n):
        for j in range(i + 1, n):
            di = np.zeros(n); di[i] = eps
            dj = np.zeros(n); dj[j] = eps
            fpp = func_scalar(x + di + dj)
            fpm = func_scalar(x + di - dj)
            fmp = func_scalar(x - di + dj)
            fmm = func_scalar(x - di - dj)
            Hij = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            H[i, j] = H[j, i] = Hij
    return g, H


def print_speed_table(label, t_ep, t_te=None, t_for=None, t_fd=None):
    """
    Show timings with speed-up vs EP only.
    Speed-up is (EP time / method time). >1 means method is faster than EP.
    """
    def ms(x): return x * 1e3
    print(f"\n== Speed (baseline = Edge-Pushing) :: {label} ==")
    print(f"EP     : {ms(t_ep):.3f} ms")
    if t_te is not None:
        print(f"Taylor : {ms(t_te):.3f} ms  (×{t_ep / t_te:.2f} vs EP)")
    if t_for is not None:
        print(f"For    : {ms(t_for):.3f} ms  (×{t_ep / t_for:.2f} vs EP)")
    if t_fd is not None:
        print(f"FD     : {ms(t_fd):.3f} ms  (×{t_ep / t_fd:.2f} vs EP)")


def print_accuracy_table(label, H_ep, H_te=None, H_for=None, H_fd=None):
    """Show relative errors w.r.t. EP."""
    print(f"\n== Accuracy (relative to EP) :: {label} ==")
    if H_te is not None:
        print(f"relerr(Taylor, EP) = {relerr(H_te, H_ep):.3e}")
    if H_for is not None:
        print(f"relerr(FoR,   EP) = {relerr(H_for, H_ep):.3e}")
    if H_fd is not None:
        print(f"relerr(FD,    EP) = {relerr(H_fd, H_ep):.3e}")


# =========================
# 1) Random n-D (no trig)
# f(x) = exp(sum x_i^2) + sqrt(sum exp(x_i + 0.3)) * log(1 + sum x_i^2)
# =========================
def make_random_funcs(n, seed=0):
    rng = np.random.default_rng(seed)
    shift = 0.3
    keys = [f"x{i}" for i in range(n)]

    def f_edge(d):
        sumsq = sum(d[k] * d[k] for k in keys)
        part1 = oexp(sumsq)
        expsum = sum(oexp(d[k] + shift) for k in keys)
        part2 = osqrt(expsum)
        part3 = olog(1.0 + sumsq)
        return part1 + part2 * part3

    def f_scalar(x):
        sumsq = np.sum(x * x)
        part1 = math.exp(sumsq)
        expsum = np.sum(np.exp(x + shift))
        part2 = math.sqrt(expsum)
        part3 = math.log(1.0 + sumsq)
        return part1 + part2 * part3

    x0 = rng.normal(loc=0.1, scale=0.3, size=n)
    inputs = {keys[i]: float(x0[i]) for i in range(n)}
    return f_edge, f_scalar, inputs


import pandas as pd

def bench_random_dims(n_list=(2, 4, 6, 8, 12), repeats=5, seed=42, fd_eps=1e-6):
    results_speed = []
    results_acc = []

    for n in n_list:
        f_edge, f_scalar, inputs = make_random_funcs(n, seed + n)
        keys = list(inputs.keys())
        x_vec = np.array([inputs[k] for k in keys])

        # === FoR ===
        t0 = time.perf_counter()
        for _ in range(repeats):
            H_for = hess_for(f_edge, inputs)
        t1 = time.perf_counter()

        # === EP baseline ===
        t2 = time.perf_counter()
        for _ in range(repeats):
            H_ep = edge_push_hessian(f_edge, inputs, sparse=False)
        t3 = time.perf_counter()

        # === Taylor Backprop ===
        t4 = time.perf_counter()
        for _ in range(repeats):
            g_t, H_te = hess_taylor_backprop(f_edge, inputs)
        t5 = time.perf_counter()

        # === FD (single evaluation) ===
        t6 = time.perf_counter()
        g_fd, H_fd = fd_grad_hess_scalar(f_scalar, x_vec, eps=fd_eps)
        t7 = time.perf_counter()

        # === Timing averages ===
        t_for = (t1 - t0) / repeats
        t_ep  = (t3 - t2) / repeats
        t_te  = (t5 - t4) / repeats
        t_fd  = (t7 - t6)

        # === Accuracy ===
        r_te  = relerr(H_te, H_ep)
        r_for = relerr(H_for, H_ep)
        r_fd  = relerr(H_fd, H_ep)

        # === Record results ===
        results_speed.append({
            "n": int(n),
            "EP (ms)": t_ep * 1e3,
            "Taylor (ms)": t_te * 1e3,
            "FoR (ms)": t_for * 1e3,
            "FD (ms)": t_fd * 1e3,
            "×(Taylor vs EP)": t_ep / t_te,
            "×(FoR vs EP)": t_ep / t_for,
            "×(FD vs EP)": t_ep / t_fd
        })

        results_acc.append({
            "n": int(n),
            "relerr(Taylor,EP)": r_te,
            "relerr(FoR,EP)": r_for,
            "relerr(FD,EP)": r_fd
        })

    # === DataFrames ===
    df_speed = pd.DataFrame(results_speed)
    df_acc = pd.DataFrame(results_acc)

    # Pretty print factors like "×1.23"
    for col in ["×(Taylor vs EP)", "×(FoR vs EP)", "×(FD vs EP)"]:
        df_speed[col] = df_speed[col].apply(lambda x: f"×{x:.2f}")

    # === Print summary tables ===
    print("\n================ SPEED COMPARISON (average per run) ================\n")
    print(df_speed.to_markdown(index=False, floatfmt=[".0f",".3f",".3f",".3f",".3f",None,None,None]))

    print("\n================ ACCURACY COMPARISON (relative to EP) ================\n")
    print(df_acc.to_markdown(index=False, floatfmt=[".0f",".3e",".3e",".3e"]))

    return df_speed, df_acc


# =========================
# 2) Black–Scholes 5D: S, sigma, K, r, T
# =========================
def f_bsm_edge_5d(d):
    """ops-based version (ADVar); do NOT use math.* on ADVar."""
    S, sigma, K, r, T = d["S"], d["sigma"], d["K"], d["r"], d["T"]
    sig2 = sigma * sigma
    denom = sigma * osqrt(T)
    d1 = (olog(S / K) + (r + 0.5 * sig2) * T) / denom
    d2 = d1 - sigma * osqrt(T)
    discK = K * oexp(-r * T)
    return S * onorm_cdf(d1) - discK * onorm_cdf(d2)


def f_bsm_scalar_5d(x):
    """Pure numeric BS call for FD: x = [S, sigma, K, r, T]."""
    S, sigma, K, r, T = x
    if sigma <= 0 or T <= 0 or K <= 0 or S <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return S * Phi(d1) - K * math.exp(-r * T) * Phi(d2)


def bench_bsm_5d(S=100.0, sigma=0.25, K=100.0, r=0.01, T=1.0,
                 repeats=1000, fd_eps=1e-6, print_matrices=True):
    """
    BSM in 5D: S, sigma, K, r, T.
    Prints full 5x5 Hessians with numeric values (Taylor & Edge-Pushing),
    plus concise speed table vs EP and accuracy vs EP.
    """
    inputs = {"S": S, "sigma": sigma, "K": K, "r": r, "T": T}
    keys = list(inputs.keys())
    x_vec = np.array([inputs[k] for k in keys])

    # FoR
    t0 = time.perf_counter()
    for _ in range(repeats):
        H_for = hess_for(f_bsm_edge_5d, inputs)
    t1 = time.perf_counter()

    # EP (baseline)
    t2 = time.perf_counter()
    for _ in range(repeats):
        H_ep = edge_push_hessian(f_bsm_edge_5d, inputs, sparse=False)
    t3 = time.perf_counter()

    # Taylor Backprop
    t4 = time.perf_counter()
    for _ in range(repeats):
        g_t, H_te = hess_taylor_backprop(f_bsm_edge_5d, inputs)
    t5 = time.perf_counter()

    # FD once
    t6 = time.perf_counter()
    g_fd, H_fd = fd_grad_hess_scalar(f_bsm_scalar_5d, x_vec, eps=fd_eps)
    t7 = time.perf_counter()

    # timings
    t_for = (t1 - t0) / repeats
    t_ep  = (t3 - t2) / repeats
    t_te  = (t5 - t4) / repeats
    t_fd  = (t7 - t6)

    print("\n== Black–Scholes 5D (S, sigma, K, r, T) ==")
    print("Order of variables:", keys)

    # Print full matrices with numeric values (Taylor & EP)
    if print_matrices:
        np.set_printoptions(precision=8, suppress=True)
        print("\nHessian (Edge-Pushing):\n", H_ep)
        print("\nHessian (Taylor):\n", H_te)
        # Optional:
        # print("\nHessian (FoR):\n", H_for)
        # print("\nHessian (FD / Bumping):\n", H_fd)

    # Accuracy vs EP
    print_accuracy_table("BSM 5D", H_ep, H_te=H_te, H_for=H_for, H_fd=H_fd)
    # Speeds vs EP
    print_speed_table("BSM 5D", t_ep, t_te=t_te, t_for=t_for, t_fd=t_fd)

    return {
        "keys": keys,
        "H_ep": H_ep, "H_taylor": H_te, "H_for": H_for, "H_fd": H_fd,
        "t_ep": t_ep, "t_taylor": t_te, "t_for": t_for, "t_fd": t_fd
    }


# =========================
# 3) Composite (no trig)
# =========================
def make_composite_funcs(n=6, seed=123):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    keys = [f"x{i}" for i in range(n)]

    def f_edge(d):
        expsum = sum(oexp(c[i] * d[keys[i]] + 0.2) for i in range(n))
        part1 = osqrt(expsum)
        sqsum = sum((c[i] * d[keys[i]]) * (c[i] * d[keys[i]]) for i in range(n))
        part2 = olog(1.0 + sqsum)
        return part1 * part2 + oexp(sum(d[k] for k in keys))

    def f_scalar(x):
        expsum = np.sum(np.exp(c * x + 0.2))
        part1 = math.sqrt(expsum)
        sqsum = np.sum((c * x) ** 2)
        part2 = math.log(1.0 + sqsum)
        return part1 * part2 + math.exp(np.sum(x))

    x0 = rng.normal(loc=0.1, scale=0.25, size=n)
    inputs = {keys[i]: float(x0[i]) for i in range(n)}
    return f_edge, f_scalar, inputs


def bench_composite(n=6, repeats=200, fd_eps=1e-6):
    f_edge, f_scalar, inputs = make_composite_funcs(n=n)
    keys = list(inputs.keys())
    x_vec = np.array([inputs[k] for k in keys])

    # One run for accuracy
    H_for = hess_for(f_edge, inputs)
    H_ep  = edge_push_hessian(f_edge, inputs, sparse=False)
    g_t, H_te = hess_taylor_backprop(f_edge, inputs)
    g_fd, H_fd = fd_grad_hess_scalar(f_scalar, x_vec, eps=fd_eps)

    print_accuracy_table(f"composite n={n}", H_ep, H_te=H_te, H_for=H_for, H_fd=H_fd)

    # Timings (averages)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = hess_for(f_edge, inputs)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(repeats):
        _ = edge_push_hessian(f_edge, inputs, sparse=False)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    for _ in range(repeats):
        _ = hess_taylor_backprop(f_edge, inputs)
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    _ = fd_grad_hess_scalar(f_scalar, x_vec, eps=fd_eps)
    t7 = time.perf_counter()

    t_for = (t1 - t0) / repeats
    t_ep  = (t3 - t2) / repeats
    t_te  = (t5 - t4) / repeats
    t_fd  = (t7 - t6)

    print_speed_table(f"composite n={n}", t_ep, t_te=t_te, t_for=t_for, t_fd=t_fd)


# =========================
# Entry (optional for CLI)
# =========================
if __name__ == "__main__":
    bench_random_dims()
    bench_bsm_5d()
    bench_composite()
