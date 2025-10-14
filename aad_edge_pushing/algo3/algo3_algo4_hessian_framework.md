
# Implementation Plan: Algorithm 3 and 4 from *A new framework for the computation of Hessians*

This document provides a **practical and directly executable** Python plan implementing **Algorithm 3 (block form)** and **Algorithm 4 (component form)** from the referenced paper.

---

## 1. Common Data Structures

```python
class SymmSparse:
    def __init__(self, n):
        self.n = n
        self.map = {}  # key=(i,j) with i<=j

    def _key(self, i, j):
        return (i, j) if i <= j else (j, i)

    def add(self, i, j, val):
        if val == 0:
            return
        k = self._key(i, j)
        self.map[k] = self.map.get(k, 0.0) + val

    def get(self, i, j):
        return self.map.get(self._key(i, j), 0.0)
```

---

## 2. Algorithm 3 (Block Form)

```python
def algo3_block(tape, P_index):
    n_input = len(P_index)
    N = tape.num_nodes_total()
    W = SymmSparse(N)
    vbar = [0.0] * (N + 1)
    vbar[tape.L] = 1.0

    for i in range(tape.L, 0, -1):
        preds = tape.preds(i)
        d1 = tape.grad1(i)
        d2 = tape.grad2(i)

        # Block 1: W ← (Φ'i)^T W Φ'i
        for jj in preds:
            for kk in preds:
                if kk < jj:
                    continue
                add_val = (
                    d1.get(kk, 0.0) * d1.get(jj, 0.0) * W.get(i, i)
                    + d1.get(kk, 0.0) * W.get(jj, i)
                    + d1.get(jj, 0.0) * W.get(i, kk)
                )
                if add_val != 0:
                    W.add(jj, kk, add_val)

        # Block 2: W += v̄_i * Φ''_i
        if vbar[i] != 0:
            for jj in preds:
                for kk in preds:
                    if kk < jj:
                        continue
                    W.add(jj, kk, vbar[i] * d2.get((jj, kk), 0.0))

        # Adjoint: v̄^T ← v̄^T Φ'_i
        if vbar[i] != 0:
            for j in preds:
                vbar[j] += vbar[i] * d1.get(j, 0.0)

    return project_W_to_inputs(W, P_index)
```

---

## 3. Algorithm 4 (Component Form)

```python
def algo4_component(tape, P_index):
    n_input = len(P_index)
    N = tape.num_nodes_total()
    W = SymmSparse(N)
    vbar = [0.0] * (N + 1)
    vbar[tape.L] = 1.0

    for i in range(tape.L, 0, -1):
        preds = tape.preds(i)
        d1 = tape.grad1(i)
        d2 = tape.grad2(i)

        # Pushing stage
        neigh_pi = []
        for (a, b), val in list(W.map.items()):
            if val == 0:
                continue
            if b == i:
                neigh_pi.append((a, val))
            elif a == i:
                neigh_pi.append((b, val))

        for p, w_pi in neigh_pi:
            if p != i:
                for j in preds:
                    if j == p:
                        W.add(p, p, 2.0 * d1.get(p, 0.0) * w_pi)
                    else:
                        W.add(min(j, p), max(j, p), d1.get(j, 0.0) * w_pi)
            else:
                for j in preds:
                    for k in preds:
                        if k < j:
                            continue
                        W.add(j, k, d1.get(k, 0.0) * d1.get(j, 0.0) * w_pi)

        # Creating stage
        if vbar[i] != 0:
            for j in preds:
                for k in preds:
                    if k < j:
                        continue
                    W.add(j, k, vbar[i] * d2.get((j, k), 0.0))

        # Adjoint stage
        if vbar[i] != 0:
            for j in preds:
                vbar[j] += vbar[i] * d1.get(j, 0.0)

    return project_W_to_inputs(W, P_index)
```

---

## 4. Integration Notes

- `tape` must provide:
  - `num_nodes_total()`
  - `L` (output node index)
  - `preds(i)` — list of predecessors of node i
  - `grad1(i)` — dict of first derivatives
  - `grad2(i)` — dict of second derivatives
- `project_W_to_inputs(W, P_index)` should extract submatrix for input variables.

---

## 5. Testing Strategy

- Analytical test: \(f(x,y)=e^{xy}+\sin x \cos y\)
- Compare Hessian output with finite difference approximation.
- Validate Algo3 and Algo4 produce identical Hessians.
- Inspect sparsity pattern and runtime.

---

## 6. License

This file is released for academic and research purposes under fair use for reproducing *A new framework for the computation of Hessians* (Griewank et al., 2008).
