"""
Quick single test to debug performance
"""
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import CapriottiCNAAD

print("Testing AAD method with M=10, N=50...")

solver = CapriottiCNAAD(M=12, N=50)
solver.S0 = 100.0
solver.K = 100.0
solver.T = 1.0
solver.r = 0.05

t_start = time.perf_counter()
greeks = solver.compute_greeks_aad(sigma_value=0.2, eps_S=0.01)
t_end = time.perf_counter()

print(f"\nSuccess! Time: {(t_end - t_start) * 1000:.2f} ms")
print(f"Price: {greeks['price']:.6f}")
print(f"Delta: {greeks['delta']:.6f}")
print(f"Gamma: {greeks['gamma']:.6f}")
print(f"Vega: {greeks['vega']:.6f}")
print(f"Vanna: {greeks['vanna']:.6f}")
print(f"Volga: {greeks['volga']:.6f}")