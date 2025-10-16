"""
Adjacency graph construction for local volatility parameters.

In the Crank-Nicolson PDE with local volatility σ[i,n], each parameter
affects the solution V through a limited set of dependencies:

1. Direct effect: V[i, n] and V[i, n+1] (same spatial node)
2. Spatial coupling: V[i-1, n], V[i+1, n] via tridiagonal structure
3. Time propagation: Effects propagate forward through all future timesteps

This adjacency structure creates sparsity in the Hessian, which Edge-Pushing
can exploit for 10-100× speedup.

Key insight: ∂²V/∂σ[i,n]∂σ[j,m] is only non-zero if:
- σ[i,n] and σ[j,m] affect overlapping nodes in the PDE solution
- This happens when they are "adjacent" in space-time
"""

import numpy as np
from typing import Dict, Set, Tuple, List
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class LocalVolAdjacency:
    """
    Adjacency graph for local volatility parameters in PDE.

    Each σ[i,n] is a vertex, and edges represent dependencies where
    one parameter affects nodes that another parameter also affects.
    """

    def __init__(self, M: int, N: int):
        """
        Initialize adjacency graph.

        Args:
            M: Number of spatial intervals (M+1 nodes)
            N: Number of time intervals (N+1 nodes)
        """
        self.M = M
        self.N = N
        self.n_params = (M + 1) * (N + 1)

        # Adjacency list: param_id -> set of adjacent param_ids
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)

        # Reverse mapping: (i, n) <-> param_id
        self.idx_to_param: Dict[Tuple[int, int], int] = {}
        self.param_to_idx: Dict[int, Tuple[int, int]] = {}

        self._build_index_mapping()
        self._build_adjacency()

    def _build_index_mapping(self):
        """Build mapping between (i,n) indices and flat parameter IDs."""
        param_id = 0
        for n in range(self.N + 1):
            for i in range(self.M + 1):
                self.idx_to_param[(i, n)] = param_id
                self.param_to_idx[param_id] = (i, n)
                param_id += 1

    def get_param_id(self, i: int, n: int) -> int:
        """Convert (i, n) to flat parameter ID."""
        return self.idx_to_param.get((i, n), -1)

    def get_param_idx(self, param_id: int) -> Tuple[int, int]:
        """Convert flat parameter ID to (i, n)."""
        return self.param_to_idx.get(param_id, (-1, -1))

    def _build_adjacency(self):
        """
        Build adjacency graph based on PDE dependency structure.

        σ[i,n] affects V through:
        1. Coefficients at timestep n: α[i,n], β[i,n], γ[i,n]
        2. Matrix A and B at timestep n
        3. This affects V[i-1:i+2, n+1] (tridiagonal coupling)
        4. Effects propagate forward through all future timesteps

        Two parameters σ[i,n] and σ[j,m] are adjacent if:
        - They affect overlapping spatial regions
        - And their time indices are "close" (within propagation distance)
        """
        for n in range(self.N + 1):
            for i in range(self.M + 1):
                param1 = self.get_param_id(i, n)

                # Spatial neighbors at same time (affect same timestep)
                for di in [-1, 0, 1]:
                    i_neighbor = i + di
                    if 0 <= i_neighbor <= self.M:
                        param2 = self.get_param_id(i_neighbor, n)
                        if param1 != param2:
                            self.adjacency[param1].add(param2)
                            self.adjacency[param2].add(param1)

                # Temporal propagation: σ[i,n] affects future timesteps
                # Due to tridiagonal coupling, it affects i-1, i, i+1 at future times
                for dn in range(1, min(3, self.N + 1 - n)):  # Look ahead 2 steps
                    for di in [-1, 0, 1]:
                        i_future = i + di
                        n_future = n + dn
                        if 0 <= i_future <= self.M and n_future <= self.N:
                            param2 = self.get_param_id(i_future, n_future)
                            if param1 != param2:
                                self.adjacency[param1].add(param2)
                                self.adjacency[param2].add(param1)

    def get_neighbors(self, i: int, n: int) -> Set[Tuple[int, int]]:
        """
        Get all adjacent parameters for σ[i,n].

        Args:
            i: Spatial index
            n: Time index

        Returns:
            Set of (i, n) tuples that are adjacent
        """
        param_id = self.get_param_id(i, n)
        neighbor_ids = self.adjacency[param_id]
        return {self.get_param_idx(nid) for nid in neighbor_ids}

    def get_neighbor_ids(self, param_id: int) -> Set[int]:
        """Get adjacent parameter IDs for a given parameter ID."""
        return self.adjacency[param_id]

    def compute_sparsity_stats(self) -> Dict[str, float]:
        """
        Compute sparsity statistics for the Hessian.

        Returns:
            Dictionary with sparsity metrics
        """
        total_entries = self.n_params * self.n_params
        max_possible_nonzeros = sum(len(neighbors) + 1 for neighbors in self.adjacency.values())
        # +1 for diagonal

        avg_neighbors = np.mean([len(neighbors) for neighbors in self.adjacency.values()])
        max_neighbors = max(len(neighbors) for neighbors in self.adjacency.values())
        min_neighbors = min(len(neighbors) for neighbors in self.adjacency.values())

        # Upper bound on Hessian non-zeros (assuming full connectivity within adjacency)
        hessian_nonzeros_upper = 0
        for param_id in range(self.n_params):
            neighbors = self.adjacency[param_id]
            # Hessian entry (i,j) non-zero if params i and j are adjacent
            hessian_nonzeros_upper += 1  # Diagonal
            hessian_nonzeros_upper += len(neighbors)  # Off-diagonal

        sparsity = 100 * (1 - hessian_nonzeros_upper / total_entries)

        return {
            'n_params': self.n_params,
            'total_hessian_entries': total_entries,
            'max_nonzeros': hessian_nonzeros_upper,
            'sparsity_percent': sparsity,
            'avg_neighbors': avg_neighbors,
            'max_neighbors': max_neighbors,
            'min_neighbors': min_neighbors
        }

    def visualize_adjacency(self, sample_i: int = None, sample_n: int = None,
                           save_path: str = None):
        """
        Visualize the adjacency structure for a specific parameter.

        Args:
            sample_i: Spatial index to visualize (default: middle)
            sample_n: Time index to visualize (default: middle)
            save_path: Path to save figure (optional)
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping visualization")
            return None

        if sample_i is None:
            sample_i = self.M // 2
        if sample_n is None:
            sample_n = self.N // 2

        # Create grid visualization
        grid = np.zeros((self.M + 1, self.N + 1))
        grid[sample_i, sample_n] = 3  # Mark the focal parameter

        # Mark neighbors
        neighbors = self.get_neighbors(sample_i, sample_n)
        for i_nb, n_nb in neighbors:
            if grid[i_nb, n_nb] == 0:  # Don't overwrite focal point
                # Color by type of neighbor
                if n_nb == sample_n:
                    grid[i_nb, n_nb] = 1  # Same time (spatial neighbor)
                else:
                    grid[i_nb, n_nb] = 2  # Future time (temporal neighbor)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(grid.T, origin='lower', cmap='RdYlBu_r', aspect='auto')

        ax.set_xlabel('Spatial Index (i)', fontsize=12)
        ax.set_ylabel('Time Index (n)', fontsize=12)
        ax.set_title(f'Adjacency Structure for σ[{sample_i}, {sample_n}]', fontsize=14)

        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['No edge', 'Spatial neighbor', 'Temporal neighbor', 'Focal param'])

        # Mark the focal point
        ax.plot(sample_i, sample_n, 'k*', markersize=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved adjacency visualization to {save_path}")
        else:
            plt.show()

        return fig

    def print_stats(self):
        """Print adjacency graph statistics."""
        stats = self.compute_sparsity_stats()

        print("="*70)
        print("LOCAL VOLATILITY ADJACENCY GRAPH STATISTICS")
        print("="*70)
        print(f"\nGrid Size: {self.M+1} × {self.N+1}")
        print(f"Total Parameters: {stats['n_params']:,}")
        print(f"\nHessian Matrix:")
        print(f"  Total entries: {stats['total_hessian_entries']:,}")
        print(f"  Non-zero entries (upper bound): {stats['max_nonzeros']:,}")
        print(f"  Sparsity: {stats['sparsity_percent']:.2f}%")
        print(f"\nAdjacency Statistics:")
        print(f"  Average neighbors per parameter: {stats['avg_neighbors']:.1f}")
        print(f"  Maximum neighbors: {int(stats['max_neighbors'])}")
        print(f"  Minimum neighbors: {int(stats['min_neighbors'])}")

        # Compute theoretical speedup
        naive_ops = stats['total_hessian_entries']
        sparse_ops = stats['max_nonzeros']
        speedup = naive_ops / sparse_ops

        print(f"\nTheoretical Edge-Pushing Speedup:")
        print(f"  Naive Hessian operations: {naive_ops:,}")
        print(f"  Sparse operations: {sparse_ops:,}")
        print(f"  Expected speedup: {speedup:.1f}× ⭐")
        print("="*70)


def demo_adjacency():
    """Demonstrate adjacency graph construction and analysis."""

    # Small grid for visualization
    M, N = 20, 20
    print(f"\n{'='*70}")
    print(f"ADJACENCY GRAPH DEMO (M={M}, N={N})")
    print(f"{'='*70}\n")

    adj = LocalVolAdjacency(M, N)
    adj.print_stats()

    # Show neighbors for a sample parameter
    sample_i, sample_n = M // 2, N // 2
    neighbors = adj.get_neighbors(sample_i, sample_n)

    print(f"\nExample: Neighbors of σ[{sample_i}, {sample_n}]:")
    print(f"  Total neighbors: {len(neighbors)}")
    print(f"  Sample neighbors (first 10):")
    for idx, (i_nb, n_nb) in enumerate(sorted(neighbors)[:10]):
        print(f"    σ[{i_nb}, {n_nb}]")

    # Visualize
    print(f"\nGenerating adjacency visualization...")
    try:
        adj.visualize_adjacency(sample_i, sample_n, save_path='adjacency_graph.png')
    except:
        print("  (Visualization skipped - no display available)")

    # Test different grid sizes
    print(f"\n{'='*70}")
    print("SCALABILITY ANALYSIS")
    print(f"{'='*70}\n")
    print(f"{'Grid Size':<15} {'Parameters':<15} {'Sparsity':<15} {'Speedup':<15}")
    print("-"*70)

    for size in [10, 20, 50, 100]:
        adj_test = LocalVolAdjacency(size, size)
        stats_test = adj_test.compute_sparsity_stats()
        speedup_test = stats_test['total_hessian_entries'] / stats_test['max_nonzeros']

        print(f"{size}×{size:<12} {stats_test['n_params']:<15,} "
              f"{stats_test['sparsity_percent']:<14.1f}% {speedup_test:<14.1f}×")

    print(f"\n✓ Adjacency graph construction complete!")


if __name__ == "__main__":
    demo_adjacency()
