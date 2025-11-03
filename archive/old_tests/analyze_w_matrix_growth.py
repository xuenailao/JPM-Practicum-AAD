"""
Analyze W matrix growth during Edge-Pushing Algorithm 4
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.edge_pushing.symm_sparse_adjlist import SymmSparseAdjList

# Build graph
print("Building computational graph...")
solver = BS_PDE_AAD(S0=100, K=100, T=1.0, r=0.05, M=21, N_base=60)
result = solver.solve_pde_with_aad(100.0, 0.2, compute_hessian=False, verbose=False)

n_nodes = len(global_tape.nodes)
n_edges = sum(len(node.parents) for node in global_tape.nodes)

print(f"Graph: {n_nodes:,} nodes, {n_edges:,} edges")
print("\nSimulating Edge-Pushing to track W matrix sparsity...\n")

# Create index mapping
var_to_idx = {}
idx = 0

# Map inputs first (S0, sigma)
S0_var = ADVar(100.0, requires_grad=True, name="S0")
sigma_var = ADVar(0.2, requires_grad=True, name="sigma")
input_vars = [S0_var, sigma_var]

# We need to identify actual input vars from tape
# For now, just create mapping for all nodes
for node in global_tape.nodes:
    if id(node.out) not in var_to_idx:
        var_to_idx[id(node.out)] = idx
        idx += 1
    for parent, _ in node.parents:
        if id(parent) not in var_to_idx:
            var_to_idx[id(parent)] = idx
            idx += 1

print(f"Total variables in mapping: {len(var_to_idx):,}")

# Initialize W matrix
W = SymmSparseAdjList(len(var_to_idx))

# Track W matrix growth
checkpoints = [0.1, 0.25, 0.5, 0.75, 1.0]
checkpoint_idx = 0

nnz_history = []
sparsity_history = []

# Simulate edge-pushing
for node_idx, node in enumerate(reversed(global_tape.nodes)):
    i = var_to_idx.get(id(node.out), -1)
    if i < 0:
        continue

    # Get predecessors
    preds = []
    d1 = {}
    for parent, deriv in node.parents:
        j = var_to_idx.get(id(parent), -1)
        if j >= 0:
            preds.append(j)
            d1[j] = d1.get(j, 0.0) + float(deriv)

    preds = sorted(set(preds))

    # PUSHING STAGE (simplified - just track growth)
    neighbors = W.get_neighbors(i)

    for p, w_pi in neighbors:
        if p == i:
            # Diagonal case: W[j,k] for all j,k in preds
            for a, j in enumerate(preds):
                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue
                for b in range(a, len(preds)):
                    k = preds[b]
                    dk = d1.get(k, 0.0)
                    if dk != 0.0:
                        W.add(j, k, dj * dk * w_pi)
        else:
            # Off-diagonal case
            for j in preds:
                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue
                if j == p:
                    W.add(p, p, 2.0 * dj * w_pi)
                else:
                    W.add(p, j, dj * w_pi)

    # CREATING STAGE (if second derivative exists)
    # For simplicity, skip - mainly mul, div operations contribute
    if node.op_tag == 'mul' and len(preds) == 2:
        if preds[0] == preds[1]:
            W.add(preds[0], preds[0], 1.0)  # d²(x²)/dx² = 2
        else:
            W.add(preds[0], preds[1], 0.5)  # d²(xy)/dxdy = 1

    # Clear row/col i
    W.clear_row_col(i)

    # Track at checkpoints
    progress = (node_idx + 1) / len(global_tape.nodes)
    if checkpoint_idx < len(checkpoints) and progress >= checkpoints[checkpoint_idx]:
        nnz = W.nnz()
        sparsity = W.sparsity()
        nnz_history.append((checkpoints[checkpoint_idx], nnz))
        sparsity_history.append((checkpoints[checkpoint_idx], sparsity))

        print(f"Progress: {checkpoints[checkpoint_idx]*100:>5.0f}% | "
              f"W.nnz = {nnz:>10,} | "
              f"Sparsity = {sparsity:>6.2f}% | "
              f"Density = {100-sparsity:>6.2f}%")

        checkpoint_idx += 1

# Final statistics
final_nnz = W.nnz()
final_sparsity = W.sparsity()

print(f"\n{'='*70}")
print(f"FINAL W MATRIX STATISTICS:")
print(f"{'='*70}")
print(f"  Dimension           : {len(var_to_idx):,} × {len(var_to_idx):,}")
print(f"  Non-zeros           : {final_nnz:,}")
print(f"  Sparsity            : {final_sparsity:.2f}%")
print(f"  Density             : {100-final_sparsity:.2f}%")
print(f"  Theoretical max     : {len(var_to_idx)**2:,}")

# Analyze growth rate
if len(nnz_history) >= 2:
    print(f"\nW MATRIX GROWTH RATE:")
    print(f"  Stage              | NNZ Growth  | Relative")
    print(f"  {'-'*50}")
    for i in range(1, len(nnz_history)):
        prev_prog, prev_nnz = nnz_history[i-1]
        curr_prog, curr_nnz = nnz_history[i]
        growth = curr_nnz - prev_nnz
        rel_growth = growth / prev_nnz if prev_nnz > 0 else 0
        print(f"  {prev_prog*100:>3.0f}% → {curr_prog*100:>3.0f}%     | "
              f"{growth:>10,} | {rel_growth:>6.1f}x")

print(f"\n{'='*70}")
print(f"CONCLUSION:")
print(f"{'='*70}")

if final_nnz > 1_000_000:
    print(f"  ⚠️  W matrix has {final_nnz:,} non-zeros!")
    print(f"  ⚠️  This explains the {8080/238:.1f}x slowdown in Hessian computation")
    print(f"  ⚠️  Neighbor lookup cost: O(nnz) per node")
elif final_sparsity < 99:
    print(f"  ⚠️  W matrix is {100-final_sparsity:.1f}% dense")
    print(f"  ⚠️  Adjacency list optimization has limited benefit")
else:
    print(f"  ✓  W matrix remains sparse ({final_sparsity:.1f}%)")
    print(f"  ✓  Adjacency list optimization is effective")
