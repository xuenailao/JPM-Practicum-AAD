"""
Second-Order Greeks (Vanna, Volga, Cross-Gamma) with Edge-Pushing.

Second-order Greeks measure the sensitivity of first-order Greeks to changes
in underlying parameters. These are crucial for risk management and hedging.

Key Greeks:
- Vanna: ∂²V/∂S∂σ  (how Delta changes with volatility)
- Volga: ∂²V/∂σ²   (how Vega changes with volatility)
- Cross-Gamma: ∂²V/∂σ[i,n]∂σ[j,m] (sensitivities between local vols)

This implementation uses Edge-Pushing to compute these efficiently.
"""

import numpy as np
from typing import Dict, Tuple
import time

try:
    from ..hessian.hessian_edge_pushing import HessianEdgePushing
    from ..core.local_vol_solver import LocalVolAdjoint
    from ..graph.adjacency_graph import LocalVolAdjacency
except ImportError:
    from aad_edge_pushing.pde.hessian.hessian_edge_pushing import HessianEdgePushing
    from aad_edge_pushing.pde.core.local_vol_solver import LocalVolAdjoint
    from aad_edge_pushing.pde.graph.adjacency_graph import LocalVolAdjacency


class SecondOrderGreeks:
    """
    Compute second-order Greeks for local volatility models.

    Provides efficient computation of Vanna, Volga, and cross-sensitivities
    using Edge-Pushing optimization.
    """

    def __init__(self, M: int = 200, N: int = 200):
        """
        Initialize second-order Greeks computer.

        Args:
            M: Number of spatial grid points
            N: Number of time grid points
        """
        self.M = M
        self.N = N
        self.solver = LocalVolAdjoint(M, N)
        self.adjacency = LocalVolAdjacency(M, N)
        self.hessian_comp = HessianEdgePushing(self.solver, self.adjacency)

    def compute_vanna(self, S0: float, K: float, T: float, r: float,
                     sigma_grid: np.ndarray, cp_flag: str = 'C',
                     eps_S: float = 0.01, eps_sigma: float = 1e-4
                    ) -> float:
        """
        Compute Vanna: ∂²V/∂S∂σ (mixed derivative).

        Measures how Delta changes with volatility (or how Vega changes with spot).
        Important for volatility hedging.

        Args:
            S0, K, T, r: Option parameters
            sigma_grid: Local volatility grid
            cp_flag: 'C' or 'P'
            eps_S: Finite difference epsilon for S
            eps_sigma: Finite difference epsilon for σ

        Returns:
            Vanna value
        """
        self.solver.set_local_vol_grid(sigma_grid)

        # Method: Central finite difference
        # Vanna = (∂V/∂S(S+ε, σ+δ) - ∂V/∂S(S+ε, σ-δ) - ∂V/∂S(S-ε, σ+δ) + ∂V/∂S(S-ε, σ-δ)) / (4εδ)

        # Approximate ∂V/∂S by finite difference
        def compute_delta(S, sigma_perturb=0.0):
            # Perturb sigma if needed
            if sigma_perturb != 0:
                sg = sigma_grid.copy()
                sg = sg * (1 + sigma_perturb)  # Proportional perturbation
                self.solver.set_local_vol_grid(sg)
            else:
                self.solver.set_local_vol_grid(sigma_grid)

            # Compute price at S+eps and S-eps
            p_plus, _, _ = self.solver.solve_local_vol(S + eps_S, K, T, r, cp_flag)
            p_minus, _, _ = self.solver.solve_local_vol(S - eps_S, K, T, r, cp_flag)

            delta = (p_plus - p_minus) / (2 * eps_S)
            return delta

        # Compute Vanna using mixed finite difference
        delta_sigma_plus = compute_delta(S0, sigma_perturb=eps_sigma)
        delta_sigma_minus = compute_delta(S0, sigma_perturb=-eps_sigma)

        vanna = (delta_sigma_plus - delta_sigma_minus) / (2 * eps_sigma)

        return vanna

    def compute_volga(self, S0: float, K: float, T: float, r: float,
                     sigma_grid: np.ndarray, cp_flag: str = 'C',
                     eps_sigma: float = 1e-4
                    ) -> float:
        """
        Compute Volga (Vomma): ∂²V/∂σ² (second derivative w.r.t. volatility).

        Measures convexity of option value with respect to volatility.
        Important for volatility trading strategies.

        Args:
            S0, K, T, r: Option parameters
            sigma_grid: Local volatility grid
            cp_flag: 'C' or 'P'
            eps_sigma: Finite difference epsilon

        Returns:
            Volga value
        """
        self.solver.set_local_vol_grid(sigma_grid)

        # Base price and Vega
        _, grad_base, _ = self.solver.adjoint_greeks_local(S0, K, T, r, cp_flag)
        vega_base = np.sum(grad_base)  # Total Vega (sum of all sensitivities)

        # Perturb sigma up
        sigma_plus = sigma_grid * (1 + eps_sigma)
        self.solver.set_local_vol_grid(sigma_plus)
        _, grad_plus, _ = self.solver.adjoint_greeks_local(S0, K, T, r, cp_flag)
        vega_plus = np.sum(grad_plus)

        # Perturb sigma down
        sigma_minus = sigma_grid * (1 - eps_sigma)
        self.solver.set_local_vol_grid(sigma_minus)
        _, grad_minus, _ = self.solver.adjoint_greeks_local(S0, K, T, r, cp_flag)
        vega_minus = np.sum(grad_minus)

        # Second derivative (central difference)
        volga = (vega_plus - 2*vega_base + vega_minus) / (eps_sigma**2)

        return volga

    def compute_cross_gamma(self, S0: float, K: float, T: float, r: float,
                           sigma_grid: np.ndarray, cp_flag: str = 'C',
                           focus_region: str = 'atm'
                          ) -> Tuple[Dict, Dict]:
        """
        Compute cross-gamma: ∂²V/∂σ[i,n]∂σ[j,m] for local vol parameters.

        This measures how sensitivities to one volatility point affect
        sensitivities to another. Uses Edge-Pushing for efficiency.

        Args:
            S0, K, T, r: Option parameters
            sigma_grid: Local volatility grid
            cp_flag: 'C' or 'P'
            focus_region: 'atm', 'sample', or 'all'

        Returns:
            (sparse_hessian, metadata) from Edge-Pushing computation
        """
        return self.hessian_comp.compute_hessian_smart(
            S0, K, T, r, sigma_grid, cp_flag, focus_region
        )

    def compute_all_second_order(self, S0: float, K: float, T: float, r: float,
                                sigma_grid: np.ndarray, cp_flag: str = 'C'
                               ) -> Dict:
        """
        Compute all second-order Greeks in one call.

        Args:
            S0, K, T, r: Option parameters
            sigma_grid: Local volatility grid
            cp_flag: 'C' or 'P'

        Returns:
            Dictionary with all second-order Greeks
        """
        print(f"\n{'='*80}")
        print(f"COMPUTING ALL SECOND-ORDER GREEKS")
        print(f"{'='*80}\n")

        results = {}
        t_start = time.perf_counter()

        # Vanna
        print("Computing Vanna (∂²V/∂S∂σ)...")
        t0 = time.perf_counter()
        vanna = self.compute_vanna(S0, K, T, r, sigma_grid, cp_flag)
        t_vanna = (time.perf_counter() - t0) * 1000
        results['vanna'] = vanna
        results['vanna_time_ms'] = t_vanna
        print(f"  Vanna = {vanna:.6f}  (time: {t_vanna:.2f} ms)")

        # Volga
        print("\nComputing Volga (∂²V/∂σ²)...")
        t0 = time.perf_counter()
        volga = self.compute_volga(S0, K, T, r, sigma_grid, cp_flag)
        t_volga = (time.perf_counter() - t0) * 1000
        results['volga'] = volga
        results['volga_time_ms'] = t_volga
        print(f"  Volga = {volga:.6f}  (time: {t_volga:.2f} ms)")

        # Cross-Gamma (sparse)
        print("\nComputing Cross-Gamma (∂²V/∂σ[i,n]∂σ[j,m])...")
        t0 = time.perf_counter()
        cross_gamma, meta = self.compute_cross_gamma(S0, K, T, r, sigma_grid, cp_flag, 'atm')
        t_cross = (time.perf_counter() - t0) * 1000
        results['cross_gamma'] = cross_gamma
        results['cross_gamma_meta'] = meta
        results['cross_gamma_time_ms'] = t_cross
        print(f"  Cross-Gamma entries: {meta['n_entries']}")
        print(f"  Sparsity: {meta['sparsity_percent']:.1f}%")
        print(f"  Time: {t_cross:.2f} ms")

        t_total = (time.perf_counter() - t_start) * 1000
        results['total_time_ms'] = t_total

        print(f"\n✓ All second-order Greeks computed in {t_total:.2f} ms")

        return results


def demo_second_order_greeks():
    """Demonstrate second-order Greeks computation."""
    print("="*80)
    print("SECOND-ORDER GREEKS DEMONSTRATION")
    print("="*80)

    from svi_model import create_sample_svi
    from math import log, sqrt, exp
    from scipy.stats import norm

    # Setup
    M, N = 20, 20
    S0, K, T, r = 100, 100, 1.0, 0.05

    print(f"\nSetup:")
    print(f"  Grid: {M+1}×{N+1}")
    print(f"  Option: S0={S0}, K={K}, T={T}, r={r}")

    # Create local volatility grid
    svi = create_sample_svi()
    Smax = 4 * K
    S_grid = np.linspace(0, Smax, M+1)
    T_grid = np.linspace(0, T, N+1)
    sigma_grid = svi.to_pde_grid(S_grid, T_grid, r, S_ref=S0)

    # Compute Greeks
    greeks_comp = SecondOrderGreeks(M, N)
    results = greeks_comp.compute_all_second_order(S0, K, T, r, sigma_grid, 'C')

    # Compare with BSM analytical (for reference, using average vol)
    sigma_avg = np.mean(sigma_grid)
    d1 = (log(S0/K) + (r+0.5*sigma_avg**2)*T) / (sigma_avg*sqrt(T))
    d2 = d1 - sigma_avg*sqrt(T)

    # BSM second-order Greeks
    vanna_bsm = -norm.pdf(d1) * d2 / (S0 * sigma_avg * sqrt(T))
    volga_bsm = S0 * norm.pdf(d1) * sqrt(T) * d1 * d2 / sigma_avg

    print(f"\n{'='*80}")
    print("COMPARISON WITH BSM (using average vol σ={:.4f})".format(sigma_avg))
    print(f"{'='*80}\n")

    print(f"{'Greek':<20} {'PDE Local Vol':<20} {'BSM Analytical':<20} {'Difference':<15}")
    print("-"*75)
    print(f"{'Vanna':<20} {results['vanna']:<20.6f} {vanna_bsm:<20.6f} "
          f"{abs(results['vanna']-vanna_bsm):<15.6f}")
    print(f"{'Volga':<20} {results['volga']:<20.6f} {volga_bsm:<20.6f} "
          f"{abs(results['volga']-volga_bsm):<15.6f}")

    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")

    print(f"Computation Times:")
    print(f"  Vanna: {results['vanna_time_ms']:.2f} ms")
    print(f"  Volga: {results['volga_time_ms']:.2f} ms")
    print(f"  Cross-Gamma: {results['cross_gamma_time_ms']:.2f} ms")
    print(f"  Total: {results['total_time_ms']:.2f} ms")

    print(f"\nCross-Gamma Statistics:")
    meta = results['cross_gamma_meta']
    print(f"  Non-zero entries: {meta['n_entries']:,}")
    print(f"  Sparsity: {meta['sparsity_percent']:.1f}%")
    print(f"  Theoretical speedup: {meta['speedup_theoretical']:.1f}×")

    print(f"\n✓ Second-order Greeks demonstration complete!")


if __name__ == "__main__":
    demo_second_order_greeks()
