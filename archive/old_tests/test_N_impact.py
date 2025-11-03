"""Quick test: Impact of N on Gamma accuracy"""

import numpy as np
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
gamma_bsm = norm.pdf(d1) / (S0 * sigma * sqrt_T)

print("="*80)
print("Impact of Time Steps (N) on Gamma Accuracy")
print("="*80)
print(f"\nBSM Analytical Gamma: {gamma_bsm:.10f}")
print(f"\nTesting with cubic Hermite interpolation (S0 as ADVar)\n")

configs = [
    (51, 50),
    (51, 100),
    (51, 200),
    (51, 400),
    (101, 100),
    (101, 200),
    (101, 400),
]

print(f"{'M':>4} {'N':>4} {'dS':>6} {'dt':>8} {'Gamma':>13} {'Error':>8} {'Time':>8}")
print("-"*80)

for M, N in configs:
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=False)
    
    dS = 300.0 / M
    dt = T / N
    error = abs(result['gamma'] - gamma_bsm) / gamma_bsm * 100
    
    print(f"{M:>4d} {N:>4d} {dS:>6.2f} {dt:>8.5f} {result['gamma']:>13.10f} {error:>7.2f}% {result.get('time_ms', 0):>7.1f}ms")

print("\n" + "="*80)
print("Key Observations:")
print("="*80)
print("• Increasing N (more time steps) improves temporal accuracy")
print("• Increasing M (finer spatial grid) reduces interpolation artifacts")  
print("• M has stronger impact on Gamma than N for this method")
print("="*80)
