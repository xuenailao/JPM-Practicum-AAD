# AAD Edge-Pushing PDE Greeks: Complete Documentation Index

## Quick Navigation

**New to the project?** Start here:
1. [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) - Command-line usage and quick start
2. [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md) - Comprehensive guide
3. [README.md](README.md) - Project overview

**Want to understand the Gamma=0 problem?** Read:
1. [EDGE_PUSHING_GAMMA_EXPLAINED.md](EDGE_PUSHING_GAMMA_EXPLAINED.md) - Complete technical explanation
2. [GAMMA_VISUAL_EXPLANATION.txt](GAMMA_VISUAL_EXPLANATION.txt) - Visual diagrams

**Want to understand theory vs practice?** Read:
1. [EDGE_PUSHING_THEORY_VS_PRACTICE.md](EDGE_PUSHING_THEORY_VS_PRACTICE.md) - Detailed reconciliation
2. [THEORY_VS_PRACTICE_DIAGRAM.txt](THEORY_VS_PRACTICE_DIAGRAM.txt) - ASCII diagrams

---

## All Documentation Files

### Essential Guides

#### [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md)
**Purpose**: Comprehensive project guide
**Contents**:
- Quick start commands
- Four methods compared
- Critical problem and solution
- Theory vs practice explanation
- Project structure
- Typical results
- Mathematical foundation
- Validation strategy
- Performance summary
- Future work

**Read this if**: You want complete understanding of the entire project

---

#### [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)
**Purpose**: Quick reference card for command-line usage
**Contents**:
- Command-line examples
- Output format
- Four methods summary table
- Key insight (Gamma=0 fix)
- Greeks definitions
- Hybrid method explanation
- Typical results
- Troubleshooting

**Read this if**: You want to quickly run the code and understand output

---

### Technical Deep Dives

#### [EDGE_PUSHING_GAMMA_EXPLAINED.md](EDGE_PUSHING_GAMMA_EXPLAINED.md)
**Purpose**: Complete technical explanation of Gamma=0 problem
**Size**: 16KB, ~700 lines
**Contents**:
- Problem statement
- Why Gamma was 0 (linear interpolation)
- Complete solution (grid-based FD)
- Mathematical justification
- Code implementation details
- Before/after comparisons
- Why it's important

**Read this if**: You want deep technical understanding of the interpolation problem

---

#### [EDGE_PUSHING_THEORY_VS_PRACTICE.md](EDGE_PUSHING_THEORY_VS_PRACTICE.md)
**Purpose**: Reconcile theoretical Edge-Pushing with PDE implementation
**Size**: 12KB, ~436 lines, Chinese language
**Contents**:
- User's theoretical explanation (correct)
- PDE implementation reality (also correct)
- Why S0 not in computation graph
- PDE numerical method constraints
- Why S0 can't be ADVar (challenges)
- Hybrid solution justification
- Final conclusion

**Read this if**: You understand Edge-Pushing theory and wonder why our PDE implementation differs

---

#### [GAMMA_VISUAL_EXPLANATION.txt](GAMMA_VISUAL_EXPLANATION.txt)
**Purpose**: Visual ASCII diagrams explaining Gamma computation
**Size**: 17KB, ASCII art
**Contents**:
- PDE grid structure diagrams
- Interpolation vs grid FD comparison
- Computation graph visualization
- Three-point stencil diagrams
- Before/after fix comparison

**Read this if**: You prefer visual explanations with diagrams

---

#### [THEORY_VS_PRACTICE_DIAGRAM.txt](THEORY_VS_PRACTICE_DIAGRAM.txt)
**Purpose**: ASCII diagrams comparing theory and practice
**Size**: 19KB, Chinese language, extensive ASCII art
**Contents**:
- Ideal computation graph (theory)
- Actual PDE graph (practice)
- Fixed grid PDE flow
- Why S0 not in graph
- Dynamic grid challenges
- Hybrid solution visualization
- Summary comparison table

**Read this if**: You want comprehensive visual comparison of theory vs implementation

---

### Benchmark Reports

#### [gamma_computation_comparison.txt](gamma_computation_comparison.txt)
**Purpose**: Text comparison of three Gamma computation methods
**Size**: 9.3KB
**Contents**:
- Linear interpolation (wrong)
- Bumping on interpolated price (wrong)
- Grid-based FD (correct)
- Side-by-side comparison

**Read this if**: You want to see why different approaches fail or succeed

---

#### [JACOBIAN_HESSIAN_BENCHMARK_REPORT.md](JACOBIAN_HESSIAN_BENCHMARK_REPORT.md)
**Purpose**: Early benchmark results
**Size**: 8.8KB
**Contents**:
- Performance comparison
- Accuracy analysis
- Method trade-offs

**Read this if**: You want performance metrics

---

### Historical Documentation

#### [README.md](README.md)
**Purpose**: Main project README
**Size**: 9.1KB
**Contents**:
- Project overview
- Installation
- Usage examples
- Repository structure

**Read this if**: You're viewing the project on GitHub

---

#### [SUMMARY.md](SUMMARY.md)
**Purpose**: Early project summary
**Size**: 6.2KB
**Contents**:
- Initial problem description
- Early solutions
- Test results

**Read this if**: You want to see project evolution

---

#### [FINAL_PROJECT_SUMMARY.md](FINAL_PROJECT_SUMMARY.md)
**Purpose**: Intermediate project summary
**Size**: 12KB
**Contents**:
- Comprehensive results
- Method comparisons
- Conclusions

**Read this if**: You want a snapshot of the project mid-development

---

#### [FINAL_TECHNICAL_REPORT.md](FINAL_TECHNICAL_REPORT.md)
**Purpose**: Technical report format
**Size**: 11KB
**Contents**:
- Formal technical documentation
- Algorithm descriptions
- Performance analysis

**Read this if**: You need formal documentation

---

## Core Implementation Files

### Four Methods

```
aad_edge_pushing/pde/
├── method_1_analytical.py          # BSM closed-form formulas
├── method_2_bumping_fixed.py       # Fixed bumping (uses grid FD)
├── method_3_double_aad_fixed.py    # Fixed double AAD
├── method_4_edge_pushing_fixed.py  # Fixed Edge-Pushing hybrid
└── benchmark_complete.py           # Run all 4 methods
```

### Core Utilities

```
aad_edge_pushing/pde/
├── simple_pde_solver.py                    # Pure numerical CN solver
├── original_pde_aad_hessian_fixed.py       # AAD engine with grid FD fix
└── ...
```

### Algorithms

```
aad_edge_pushing/algo3/
├── algo4_edge_pushing.py           # Edge-Pushing algorithm
├── algo3_block.py                  # Block-form algorithm
└── ...
```

---

## Reading Paths

### Path 1: Quick Start User
1. [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) - Get commands
2. Run benchmark: `python aad_edge_pushing/pde/benchmark_complete.py`
3. [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md) - Understand results

### Path 2: Deep Technical Understanding
1. [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md) - Overview
2. [EDGE_PUSHING_GAMMA_EXPLAINED.md](EDGE_PUSHING_GAMMA_EXPLAINED.md) - Interpolation problem
3. [EDGE_PUSHING_THEORY_VS_PRACTICE.md](EDGE_PUSHING_THEORY_VS_PRACTICE.md) - Theory reconciliation
4. [THEORY_VS_PRACTICE_DIAGRAM.txt](THEORY_VS_PRACTICE_DIAGRAM.txt) - Visual explanation

### Path 3: Code Developer
1. [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md) - Project structure
2. [simple_pde_solver.py](aad_edge_pushing/pde/simple_pde_solver.py) - Understand PDE solver
3. [original_pde_aad_hessian_fixed.py](aad_edge_pushing/pde/original_pde_aad_hessian_fixed.py) - Understand AAD engine
4. [method_4_edge_pushing_fixed.py](aad_edge_pushing/pde/method_4_edge_pushing_fixed.py) - Understand hybrid approach

### Path 4: Researcher
1. [EDGE_PUSHING_THEORY_VS_PRACTICE.md](EDGE_PUSHING_THEORY_VS_PRACTICE.md) - Theory analysis
2. [THEORY_VS_PRACTICE_DIAGRAM.txt](THEORY_VS_PRACTICE_DIAGRAM.txt) - Visual comparison
3. [EDGE_PUSHING_GAMMA_EXPLAINED.md](EDGE_PUSHING_GAMMA_EXPLAINED.md) - Technical details
4. [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md) - Future work section

---

## Key Insights Summary

### 1. Gamma = 0 Problem (SOLVED)
- **Cause**: Linear interpolation has zero second derivative
- **Solution**: Grid-based finite difference on original price grid
- **Key Files**:
  - [EDGE_PUSHING_GAMMA_EXPLAINED.md](EDGE_PUSHING_GAMMA_EXPLAINED.md)
  - [GAMMA_VISUAL_EXPLANATION.txt](GAMMA_VISUAL_EXPLANATION.txt)

### 2. Theory vs Practice (RECONCILED)
- **Theory**: Edge-Pushing computes full Hessian for black-box functions
- **Practice**: PDE uses fixed grid, S0 not in computation graph
- **Conclusion**: Both correct for their contexts
- **Key Files**:
  - [EDGE_PUSHING_THEORY_VS_PRACTICE.md](EDGE_PUSHING_THEORY_VS_PRACTICE.md)
  - [THEORY_VS_PRACTICE_DIAGRAM.txt](THEORY_VS_PRACTICE_DIAGRAM.txt)

### 3. Hybrid Solution (IMPLEMENTED)
- **Parameter derivatives** (Vega, Volga): Use AAD/Edge-Pushing
- **Spatial derivatives** (Gamma): Use grid FD
- **Mixed derivatives** (Vanna): FD on Delta w.r.t. σ
- **Key Files**:
  - [method_4_edge_pushing_fixed.py](aad_edge_pushing/pde/method_4_edge_pushing_fixed.py)
  - [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md)

### 4. Four Methods (TESTED)
- **Method 1**: Analytical (baseline)
- **Method 2**: Bumping with grid FD
- **Method 3**: Double AAD (placeholder)
- **Method 4**: Edge-Pushing hybrid
- **Key Files**:
  - [benchmark_complete.py](aad_edge_pushing/pde/benchmark_complete.py)
  - [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md)

---

## File Size Summary

```
Documentation:
  COMPLETE_PROJECT_GUIDE.md              19 KB  (Comprehensive)
  THEORY_VS_PRACTICE_DIAGRAM.txt         19 KB  (Visual)
  GAMMA_VISUAL_EXPLANATION.txt           17 KB  (Visual)
  EDGE_PUSHING_GAMMA_EXPLAINED.md        16 KB  (Technical)
  QUICK_REFERENCE.txt                    13 KB  (Quick ref)
  EDGE_PUSHING_THEORY_VS_PRACTICE.md     12 KB  (Theory)
  FINAL_PROJECT_SUMMARY.md               12 KB  (Historical)
  FINAL_TECHNICAL_REPORT.md              11 KB  (Historical)
  README.md                               9 KB  (Overview)
  gamma_computation_comparison.txt        9 KB  (Comparison)
  JACOBIAN_HESSIAN_BENCHMARK_REPORT.md    9 KB  (Benchmarks)
  SUMMARY.md                              6 KB  (Historical)

Total Documentation: ~150 KB
```

---

## Search Keywords

Use Ctrl+F to search for:

- **Gamma = 0**: EDGE_PUSHING_GAMMA_EXPLAINED.md
- **Linear interpolation**: EDGE_PUSHING_GAMMA_EXPLAINED.md
- **Grid FD**: EDGE_PUSHING_GAMMA_EXPLAINED.md, COMPLETE_PROJECT_GUIDE.md
- **Theory vs practice**: EDGE_PUSHING_THEORY_VS_PRACTICE.md
- **Computation graph**: THEORY_VS_PRACTICE_DIAGRAM.txt
- **S0 not in graph**: EDGE_PUSHING_THEORY_VS_PRACTICE.md
- **Hybrid method**: COMPLETE_PROJECT_GUIDE.md
- **Command line**: QUICK_REFERENCE.txt
- **Benchmark**: benchmark_complete.py, COMPLETE_PROJECT_GUIDE.md
- **Four methods**: COMPLETE_PROJECT_GUIDE.md, QUICK_REFERENCE.txt
- **Vanna, Volga**: COMPLETE_PROJECT_GUIDE.md
- **PDE solver**: simple_pde_solver.py
- **Edge-Pushing**: algo4_edge_pushing.py, COMPLETE_PROJECT_GUIDE.md
- **AAD**: original_pde_aad_hessian_fixed.py

---

## Language Notes

- **English**: Most documentation
- **Chinese**:
  - EDGE_PUSHING_THEORY_VS_PRACTICE.md
  - THEORY_VS_PRACTICE_DIAGRAM.txt

---

## Last Updated

**Date**: 2025-10-29
**Status**: Complete
**Next Steps**: See "Future Work" in COMPLETE_PROJECT_GUIDE.md

---

## Quick Command Reference

```bash
# Complete benchmark (all 4 methods)
python aad_edge_pushing/pde/benchmark_complete.py

# Individual method tests
python aad_edge_pushing/pde/method_1_analytical.py
python aad_edge_pushing/pde/method_2_bumping_fixed.py
python aad_edge_pushing/pde/method_3_double_aad_fixed.py
python aad_edge_pushing/pde/method_4_edge_pushing_fixed.py
```

---

**For complete command-line usage**: See [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)
**For complete technical guide**: See [COMPLETE_PROJECT_GUIDE.md](COMPLETE_PROJECT_GUIDE.md)
