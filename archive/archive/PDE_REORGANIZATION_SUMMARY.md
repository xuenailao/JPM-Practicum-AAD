# PDE Folder Reorganization Summary

## Overview
The `aad_edge_pushing/pde/` folder has been reorganized from a flat structure into a hierarchical, modular organization based on functionality and dependencies.

## New Structure

```
aad_edge_pushing/pde/
├── __init__.py                 # Main exports from all submodules
├── core/                       # Core PDE solvers
│   ├── __init__.py
│   └── local_vol_solver.py    # LocalVolSolver, LocalVolAdjoint
├── models/                     # Volatility surface models
│   ├── __init__.py
│   └── svi_model.py           # SVIModel, create_sample_svi
├── graph/                      # Adjacency graph structures
│   ├── __init__.py
│   └── adjacency_graph.py     # LocalVolAdjacency
├── hessian/                    # Hessian computation methods
│   ├── __init__.py
│   ├── hessian_computation.py         # HessianComputer
│   ├── hessian_edge_pushing.py        # HessianEdgePushing
│   ├── second_order_adjoint.py        # SecondOrderAdjoint
│   └── true_second_order_ad_optimized.py  # TrueSecondOrderADOptimized
├── greeks/                     # Greeks computation
│   ├── __init__.py
│   └── second_order_greeks.py  # SecondOrderGreeks (Vanna, Volga)
└── aad_integration/            # ADVar-based PDE solvers
    ├── __init__.py
    ├── pde_aad_solver.py       # PDEAADSolver
    ├── pde_aad_edge_pushing.py # PDEAADEdgePushing
    └── capriotti_cn_aad.py     # CapriottiCNAADFixed
```

## Module Descriptions

### 1. **core/** - Base PDE Solvers
Contains the fundamental PDE solving infrastructure:
- **LocalVolSolver**: Crank-Nicolson solver with local volatility σ(S,t)
- **LocalVolAdjoint**: Discrete adjoint method for first-order Greeks

**Key feature**: Standalone implementation with `_build_matrices` method for tridiagonal CN scheme.

### 2. **models/** - Volatility Surface Models
Parametric models for volatility surfaces:
- **SVIModel**: Stochastic Volatility Inspired model with Dupire conversion
- **create_sample_svi**: Utility to create sample SVI parameters

**Purpose**: Provides local volatility grids σ(S,t) for PDE pricing.

### 3. **graph/** - Adjacency Structures
Graph-based structures that exploit Hessian sparsity:
- **LocalVolAdjacency**: Adjacency graph for local volatility parameters

**Key insight**: In CN PDE with local volatility, each parameter σ[i,n] affects only neighboring nodes, creating sparse Hessian structure ideal for Edge-Pushing.

### 4. **hessian/** - Hessian Computation Methods
Various approaches to computing ∂²V/∂σ[i,n]∂σ[j,m]:

- **HessianComputer**: Finite difference on Jacobian (baseline)
- **HessianEdgePushing**: Edge-pushing optimized sparse Hessian (10-100× speedup)
- **SecondOrderAdjoint**: True second-order adjoint method
- **TrueSecondOrderADOptimized**: Optimized second-order AD with tangent/adjoint caching

**Dependencies**: All depend on `core.LocalVolAdjoint` and `graph.LocalVolAdjacency`.

### 5. **greeks/** - Greeks Computation
Specialized Greeks calculators for options:
- **SecondOrderGreeks**: Vanna (∂²V/∂S∂σ), Volga (∂²V/∂σ²), cross-sensitivities

**Purpose**: Risk management and hedging using Edge-Pushing optimization.

### 6. **aad_integration/** - ADVar-Based Solvers
PDE solvers that integrate with ADVar for automatic computational graph construction:

- **PDEAADSolver**: Implicit Crank-Nicolson with ADVar and sparse matrix solvers
- **PDEAADEdgePushing**: Explicit scheme with ADVar + Algorithm 4
- **CapriottiCNAADFixed**: Capriotti's corrected CN+AAD implementation

**Key innovation**: Uses ADVar for PDE operations → automatic graph construction → Algorithm 4 for Hessian extraction.

## Import Path Updates

### Internal Imports (within pde/)
All internal imports now use relative imports with proper nesting level:

- From `hessian/` to `core/`: `from ..core.local_vol_solver import ...`
- From `aad_integration/` to sibling AAD module: `from ...aad.core.var import ...`

### External Imports
Users can import from the top-level module:
```python
from aad_edge_pushing.pde import (
    LocalVolSolver,
    LocalVolAdjoint,
    SVIModel,
    HessianEdgePushing,
    SecondOrderGreeks,
    PDEAADSolver,
)
```

Or from specific submodules:
```python
from aad_edge_pushing.pde.core import LocalVolAdjoint
from aad_edge_pushing.pde.hessian import HessianEdgePushing
```

## Benefits of Reorganization

1. **Clear dependency hierarchy**: Core → Graph → Hessian → Greeks
2. **Separation of concerns**: Each folder has a single, well-defined purpose
3. **Easier navigation**: Related functionality grouped together
4. **Better maintainability**: Changes localized to appropriate submodules
5. **Explicit exports**: Each `__init__.py` documents what's available
6. **Backward compatibility**: Main `pde/__init__.py` re-exports everything

## Dependency Graph

```
models/svi_model.py (standalone)
    ↓
core/local_vol_solver.py (depends on: models)
    ↓
graph/adjacency_graph.py (standalone)
    ↓
hessian/* (depends on: core, graph, models)
    ↓
greeks/second_order_greeks.py (depends on: core, graph, hessian)
    ↓
aad_integration/* (depends on: models, and external aad/edge_pushing modules)
```

## Testing
All imports verified successfully:
```bash
python3 -c "from aad_edge_pushing.pde import *"
# ✓ Import successful! Available classes: 13
```

## Migration Notes

### For existing code using old imports:
- **Old**: `from aad_edge_pushing.pde.local_vol_solver import LocalVolAdjoint`
- **New**: `from aad_edge_pushing.pde.core import LocalVolAdjoint`
- **Or**: `from aad_edge_pushing.pde import LocalVolAdjoint` (recommended)

### For developers:
- Add new solvers to `core/`
- Add new Hessian methods to `hessian/`
- Add new volatility models to `models/`
- Update respective `__init__.py` files when adding new classes

## Files Moved

| Original Location | New Location | Category |
|-------------------|--------------|----------|
| `local_vol_solver.py` | `core/local_vol_solver.py` | Core solver |
| `svi_model.py` | `models/svi_model.py` | Volatility model |
| `adjacency_graph.py` | `graph/adjacency_graph.py` | Graph structure |
| `hessian_computation.py` | `hessian/hessian_computation.py` | Hessian method |
| `hessian_edge_pushing.py` | `hessian/hessian_edge_pushing.py` | Hessian method |
| `second_order_adjoint.py` | `hessian/second_order_adjoint.py` | Hessian method |
| `true_second_order_ad_optimized.py` | `hessian/true_second_order_ad_optimized.py` | Hessian method |
| `second_order_greeks.py` | `greeks/second_order_greeks.py` | Greeks calculator |
| `pde_aad_solver.py` | `aad_integration/pde_aad_solver.py` | AAD integration |
| `pde_aad_edge_pushing.py` | `aad_integration/pde_aad_edge_pushing.py` | AAD integration |
| `capriotti_cn_aad.py` | `aad_integration/capriotti_cn_aad.py` | AAD integration |

## Fixes Applied

1. **LocalVolSolver**: Made standalone by adding `_build_matrices` method (was previously extending missing `CrankNicolsonSolver`)
2. **Import paths**: Updated all relative imports to use correct nesting levels
3. **AAD imports**: Fixed imports from `aad` and `edge_pushing` modules (now use `...aad` to go up to `aad_edge_pushing/` level)

---
*Reorganization completed on 2025-10-23*
