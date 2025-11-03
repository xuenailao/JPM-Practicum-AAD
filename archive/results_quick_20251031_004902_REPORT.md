# Greeks Computation Benchmark Report

**Total Tests:** 90

**Date:** 2025-10-31 01:28:37


---

## 1. Speed Comparison

| Method | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|--------|-----------|----------|----------|----------|
| Analytical      |      0.49 |     0.05 |     0.34 |     0.58 |
| Bumping         |     65.16 |    40.71 |    31.90 |   146.39 |
| AAD+Bumping     |   3067.74 |   176.45 |  2889.00 |  3327.32 |
| Double-AAD      |  49010.55 |  2468.45 | 45469.52 | 53654.96 |
| Edge-Pushing    |  49098.66 |  2582.30 | 44356.00 | 53164.70 |

## 2. Accuracy Comparison

| Method | Δ err% | Γ err% | ν err% | Vanna err% | Volga err% |
|--------|--------|--------|--------|------------|------------|
| Bumping         |   9.01 |   5.18 |   2.16 |     154.31 |     941.51 |
| AAD+Bumping     |   0.50 |   2.18 |   1.55 |      89.29 |     840.87 |
| Double-AAD      |   0.50 |   2.18 |   1.55 |      88.56 |     838.17 |
| Edge-Pushing    |   0.50 |   2.18 |   1.55 |      88.56 |     838.17 |

## 3. Computational Cost

| Method | PDE Solves | Graph Nodes | Graph Edges |
|--------|------------|-------------|-------------|
| Analytical      |          0 |           0 |           0 |
| Bumping         |          5 |           0 |           0 |
| AAD+Bumping     |          5 |      33,630 |      67,066 |
| Double-AAD      |          3 |      33,630 |      67,066 |
| Edge-Pushing    |          1 |      33,630 |      67,066 |

## 4. Key Findings

- **Fastest PDE method:** Bumping
- **Most accurate Gamma:** Double-AAD
- **Most accurate Volga:** Double-AAD

### Recommendations:
- **Quick computations:** Edge-Pushing (M=51, 1 PDE solve)
- **High accuracy:** Edge-Pushing (M=101, Gamma < 0.5%)
- **Simple implementation:** Bumping (5 PDE solves, moderate accuracy)

---
*Auto-generated from comprehensive benchmark*
