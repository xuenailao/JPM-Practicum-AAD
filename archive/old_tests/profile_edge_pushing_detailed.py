"""
Detailed profiling of Edge-Pushing Algorithm 4 bottlenecks
"""
import numpy as np
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.edge_pushing.symm_sparse_adjlist import SymmSparseAdjList
from collections import defaultdict

print("="*80)
print("DETAILED PROFILING: Edge-Pushing Algorithm 4")
print("="*80)

# Build graph
print("\n1. Building computational graph...")
solver = BS_PDE_AAD(S0=100, K=100, T=1.0, r=0.05, M=21, N_base=60)
result = solver.solve_pde_with_aad(100.0, 0.2, compute_hessian=False, verbose=False)

n_nodes = len(global_tape.nodes)
n_edges = sum(len(node.parents) for node in global_tape.nodes)

print(f"   Graph: {n_nodes:,} nodes, {n_edges:,} edges")

# Create index mapping
print("\n2. Creating variable index mapping...")
t0 = time.perf_counter()

var_to_idx = {}
idx = 0

for node in global_tape.nodes:
    if id(node.out) not in var_to_idx:
        var_to_idx[id(node.out)] = idx
        idx += 1
    for parent, _ in node.parents:
        if id(parent) not in var_to_idx:
            var_to_idx[id(parent)] = idx
            idx += 1

t_mapping = (time.perf_counter() - t0) * 1000
print(f"   Mapping time: {t_mapping:.2f} ms")
print(f"   Total variables: {len(var_to_idx):,}")

# Profile edge-pushing components
print("\n3. Profiling Edge-Pushing algorithm...")
print("-"*80)

W = SymmSparseAdjList(len(var_to_idx))

# Timing accumulators
times = {
    'get_preds': 0.0,
    'get_derivatives': 0.0,
    'get_neighbors': 0.0,
    'pushing_diagonal': 0.0,
    'pushing_offdiag': 0.0,
    'creating': 0.0,
    'clear': 0.0,
}

# Operation counters
counters = {
    'nodes_processed': 0,
    'neighbors_total': 0,
    'w_add_calls': 0,
    'w_get_calls': 0,
}

# Sample timing for detailed operations
detailed_samples = []
sample_interval = max(1, len(global_tape.nodes) // 100)  # Sample 100 nodes

for node_idx, node in enumerate(reversed(global_tape.nodes)):
    i = var_to_idx.get(id(node.out), -1)
    if i < 0:
        continue

    counters['nodes_processed'] += 1
    do_sample = (node_idx % sample_interval == 0)

    if do_sample:
        t_node_start = time.perf_counter()

    # Get predecessors
    t0 = time.perf_counter()
    preds = []
    for parent, _ in node.parents:
        j = var_to_idx.get(id(parent), -1)
        if j >= 0:
            preds.append(j)
    preds = sorted(set(preds))
    times['get_preds'] += (time.perf_counter() - t0)

    # Get derivatives
    t0 = time.perf_counter()
    d1 = {}
    for parent, deriv in node.parents:
        j = var_to_idx.get(id(parent), -1)
        if j >= 0:
            d1[j] = d1.get(j, 0.0) + float(deriv)
    times['get_derivatives'] += (time.perf_counter() - t0)

    # PUSHING STAGE
    t0 = time.perf_counter()
    neighbors = W.get_neighbors(i)
    times['get_neighbors'] += (time.perf_counter() - t0)

    counters['neighbors_total'] += len(neighbors)

    for p, w_pi in neighbors:
        if p == i:
            # Diagonal case
            t0 = time.perf_counter()
            for a, j in enumerate(preds):
                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue
                for b in range(a, len(preds)):
                    k = preds[b]
                    dk = d1.get(k, 0.0)
                    if dk != 0.0:
                        W.add(j, k, dj * dk * w_pi)
                        counters['w_add_calls'] += 1
            times['pushing_diagonal'] += (time.perf_counter() - t0)
        else:
            # Off-diagonal case
            t0 = time.perf_counter()
            for j in preds:
                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue
                if j == p:
                    W.add(p, p, 2.0 * dj * w_pi)
                    counters['w_add_calls'] += 1
                else:
                    W.add(p, j, dj * w_pi)
                    counters['w_add_calls'] += 1
            times['pushing_offdiag'] += (time.perf_counter() - t0)

    # CREATING STAGE
    t0 = time.perf_counter()
    if node.op_tag == 'mul' and len(preds) == 2:
        if preds[0] == preds[1]:
            W.add(preds[0], preds[0], 1.0)
            counters['w_add_calls'] += 1
        else:
            W.add(preds[0], preds[1], 0.5)
            counters['w_add_calls'] += 1
    elif node.op_tag == 'div' and len(preds) == 2:
        # Second derivatives for division
        pass  # Simplified for profiling
    times['creating'] += (time.perf_counter() - t0)

    # Clear row/col
    t0 = time.perf_counter()
    W.clear_row_col(i)
    times['clear'] += (time.perf_counter() - t0)

    if do_sample:
        t_node_end = time.perf_counter()
        node_time = (t_node_end - t_node_start) * 1000
        detailed_samples.append({
            'node_idx': node_idx,
            'neighbors': len(neighbors),
            'preds': len(preds),
            'time_ms': node_time,
            'op': node.op_tag
        })

# Print results
print("\n4. TIMING BREAKDOWN:")
print("-"*80)

total_time = sum(times.values()) * 1000
for component, time_s in sorted(times.items(), key=lambda x: -x[1]):
    time_ms = time_s * 1000
    pct = 100.0 * time_ms / total_time if total_time > 0 else 0
    print(f"  {component:<25} : {time_ms:>8.2f} ms ({pct:>5.1f}%)")
print(f"  {'TOTAL':<25} : {total_time:>8.2f} ms")

print("\n5. OPERATION COUNTERS:")
print("-"*80)
for name, count in counters.items():
    print(f"  {name:<25} : {count:>12,}")

avg_neighbors = counters['neighbors_total'] / counters['nodes_processed'] if counters['nodes_processed'] > 0 else 0
print(f"  {'avg_neighbors/node':<25} : {avg_neighbors:>12.2f}")

avg_adds_per_node = counters['w_add_calls'] / counters['nodes_processed'] if counters['nodes_processed'] > 0 else 0
print(f"  {'avg_W.add_calls/node':<25} : {avg_adds_per_node:>12.2f}")

print("\n6. SAMPLE NODE ANALYSIS (100 samples):")
print("-"*80)

if detailed_samples:
    # Find slowest nodes
    slowest = sorted(detailed_samples, key=lambda x: -x['time_ms'])[:10]

    print("\n  Top 10 slowest nodes:")
    print(f"  {'Rank':<6} | {'Node#':<8} | {'Op':<8} | {'Nbrs':<6} | {'Preds':<6} | {'Time(μs)':<10}")
    print(f"  {'-'*60}")
    for rank, sample in enumerate(slowest, 1):
        print(f"  {rank:<6} | {sample['node_idx']:<8} | {sample['op']:<8} | "
              f"{sample['neighbors']:<6} | {sample['preds']:<6} | {sample['time_ms']*1000:<10.2f}")

    # Analyze by operation type
    op_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0, 'max_neighbors': 0})
    for sample in detailed_samples:
        op = sample['op']
        op_stats[op]['count'] += 1
        op_stats[op]['total_time'] += sample['time_ms']
        op_stats[op]['max_neighbors'] = max(op_stats[op]['max_neighbors'], sample['neighbors'])

    print("\n  Average time per operation type:")
    print(f"  {'Operation':<10} | {'Samples':<8} | {'Avg(μs)':<10} | {'Max Nbrs':<10}")
    print(f"  {'-'*50}")
    for op, stats in sorted(op_stats.items(), key=lambda x: -x[1]['total_time']/x[1]['count']):
        avg_time = stats['total_time'] / stats['count'] * 1000
        print(f"  {op:<10} | {stats['count']:<8} | {avg_time:<10.2f} | {stats['max_neighbors']:<10}")

print("\n7. W MATRIX FINAL STATE:")
print("-"*80)
final_nnz = W.nnz()
final_sparsity = W.sparsity()
print(f"  Non-zeros               : {final_nnz:>12,}")
print(f"  Sparsity                : {final_sparsity:>11.2f}%")
print(f"  Dimension               : {len(var_to_idx):>12,} × {len(var_to_idx):,}")

print("\n8. BOTTLENECK DIAGNOSIS:")
print("="*80)

# Calculate per-operation costs
cost_per_neighbor = times['get_neighbors'] * 1000 / counters['neighbors_total'] if counters['neighbors_total'] > 0 else 0
cost_per_add = (times['pushing_diagonal'] + times['pushing_offdiag'] + times['creating']) * 1000 / counters['w_add_calls'] if counters['w_add_calls'] > 0 else 0

print(f"  Cost per neighbor lookup: {cost_per_neighbor*1000:.3f} μs")
print(f"  Cost per W.add() call   : {cost_per_add*1000:.3f} μs")
print()

# Identify bottleneck
pushing_time = times['pushing_diagonal'] + times['pushing_offdiag']
pushing_pct = 100.0 * pushing_time / sum(times.values())

if pushing_pct > 50:
    print(f"  🔴 BOTTLENECK: Pushing stage ({pushing_pct:.1f}% of time)")
    print(f"     - {counters['w_add_calls']:,} W.add() calls")
    print(f"     - Avg {avg_adds_per_node:.1f} adds per node")
    print()
    print(f"  💡 OPTIMIZATION OPPORTUNITIES:")
    print(f"     1. Reduce W.add() overhead (currently {cost_per_add*1000:.3f} μs each)")
    print(f"     2. Batch W.add() operations")
    print(f"     3. Optimize dictionary operations in SymmSparseAdjList")
else:
    print(f"  ✓ Pushing stage is well-optimized ({pushing_pct:.1f}% of time)")

neighbor_pct = 100.0 * times['get_neighbors'] / sum(times.values())
if neighbor_pct > 20:
    print(f"\n  🔴 SIGNIFICANT: Neighbor lookup ({neighbor_pct:.1f}% of time)")
    print(f"     - {counters['neighbors_total']:,} total neighbor queries")
    print(f"     - Avg {avg_neighbors:.1f} neighbors per node")
else:
    print(f"\n  ✓ Neighbor lookup is efficient ({neighbor_pct:.1f}% of time)")

print()
print("="*80)
