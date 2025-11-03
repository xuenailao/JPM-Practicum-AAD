#!/usr/bin/env python3
"""
Verify Rannacher Setup - Basic Import and Initialization Test
"""

import sys

print("="*80)
print("RANNACHER SETUP VERIFICATION")
print("="*80)

# Test 1: Import
print("\n[1/3] Testing imports...")
try:
    from aad_edge_pushing.pde.pde_aad_rannacher import BS_PDE_AAD_Rannacher
    print("  ✓ BS_PDE_AAD_Rannacher imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialization
print("\n[2/3] Testing solver initialization...")
try:
    solver = BS_PDE_AAD_Rannacher(
        S0=100.0, K=100.0, T=1.0, r=0.05,
        M=51, N_base=100,
        use_rannacher=True,
        rannacher_steps=4
    )
    print(f"  ✓ Solver initialized successfully")
    print(f"    - use_rannacher: {solver.use_rannacher}")
    print(f"    - rannacher_steps: {solver.rannacher_steps}")
    print(f"    - M: {solver.M}, N_base: {solver.N_base}")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Simple PDE solve (Jacobian only, no Hessian)
print("\n[3/3] Testing simple PDE solve (Jacobian only)...")
try:
    result = solver.solve_pde_with_aad(
        S0_val=100.0,
        sigma_val=0.2,
        compute_hessian=False,  # Skip Hessian to save time
        verbose=False
    )
    print(f"  ✓ PDE solved successfully")
    print(f"    - Price: {result['price']:.6f}")
    print(f"    - Delta: {result['delta']:.6f}")
    print(f"    - Vega: {result['vega']:.6f}")
    print(f"    - Time: {result['time_ms']:.2f} ms")
except Exception as e:
    print(f"  ✗ PDE solve failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)
print("\nRannacher implementation is ready to use.")
print("\nUsage:")
print("  from aad_edge_pushing.pde.pde_aad_rannacher import BS_PDE_AAD_Rannacher")
print("  solver = BS_PDE_AAD_Rannacher(..., use_rannacher=True, rannacher_steps=4)")
print("  result = solver.solve_pde_with_aad(...)")
print()
