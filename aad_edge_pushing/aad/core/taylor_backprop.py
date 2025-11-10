# aad/core/taylor_backprop.py
from __future__ import annotations
import numpy as np
from typing import List
from .tape import global_tape
from .var import ADVar

# ---- local 2nd derivatives f_{ij} per primitive op ----
def _second_derivatives(tag: str, parents: List,):
    """
    Return local second-derivative matrix [f_{ij}] for the recorded node
    evaluated at current parent values.
    Assumes scalar primitives; fallbacks=0 for unsupported ops.
    """
    m = len(parents)
    Hloc = np.zeros((m, m), dtype=float)
    vals = [p.val for (p, _) in parents]

    if tag in ("add", "sub", "neg"):
        # all second partials are zero
        return Hloc

    if tag == "mul":
        # y = x * z => d2y/dx dz = 1 (symmetric)
        if m == 2:
            Hloc[0, 1] = 1.0
            Hloc[1, 0] = 1.0
        return Hloc

    if tag == "div":
        # y = x / z
        if m == 2:
            x, z = vals
            Hloc[0, 0] = 0.0
            Hloc[0, 1] = Hloc[1, 0] = -1.0 / (z * z)
            Hloc[1, 1] = 2.0 * x / (z ** 3)
        return Hloc

    if tag == "sqrt":
        # y = sqrt(x); f''(x) = -1 / (4 x^(3/2))
        if m == 1:
            x = float(vals[0])
            # clamp to avoid division by zero / negative input (keep AD flow stable)
            eps = 1e-12
            xx = max(x, eps)
            Hloc[0, 0] = -0.25 * (xx ** -1.5)
        return Hloc

    if tag == "exp":
        # y = exp(x): f'' = exp(x)
        if m == 1:
            Hloc[0, 0] = np.exp(vals[0])
        return Hloc

    if tag == "log":
        # y = log(x): f'' = -1/x^2
        if m == 1:
            x = float(vals[0])
            eps = 1e-12
            xx = max(x, eps)
            Hloc[0, 0] = -1.0 / (xx * xx)
        return Hloc

    if tag == "norm_cdf":
        # y = Φ(x); f'(x) = φ(x); f''(x) = -x φ(x)
        if m == 1:
            x = np.asarray(vals[0], dtype=float)
            phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
            Hloc[0, 0] = (-x) * phi
        return Hloc

    if tag == "norm_pdf":
        # y = φ(x) = 1/sqrt(2π) * exp(-x^2/2)
        # f'(x) = -x φ(x); f''(x) = (x^2 - 1) φ(x)
        if m == 1:
            x = np.asarray(vals[0], dtype=float)
            phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
            Hloc[0, 0] = (x * x - 1.0) * phi
        return Hloc

    if tag == "sin":
        if m == 1:
            x = vals[0]
            Hloc[0, 0] = -np.sin(x)
        return Hloc

    if tag == "cos":
        if m == 1:
            x = vals[0]
            Hloc[0, 0] = -np.cos(x)
        return Hloc

    if tag == "softplus":
        # f = log(1+e^x); f' = σ, f'' = σ(1-σ)
        if m == 1:
            x = vals[0]
            sig = 1.0 / (1.0 + np.exp(-x))
            Hloc[0, 0] = sig * (1.0 - sig)
        return Hloc

    if tag == "pow":
        # Parents order: (base, expo)
        base, expo = parents[0][0], parents[1][0]
        if not expo.requires_grad:
            # y = x^c
            c = float(expo.val)
            x = float(base.val)
            Hloc[0, 0] = c * (c - 1.0) * (x ** (c - 2.0))
        elif not base.requires_grad:
            # y = a^x
            a = float(base.val)
            x = float(expo.val)
            y = a ** x
            ln_a = np.log(a)
            Hloc[1, 1] = (ln_a ** 2) * y
        # if both are active: leave zeros (not yet supported)
        return Hloc

    return Hloc

def taylor_backpropagate(inputs: List[ADVar], target: ADVar=None, targets: List[ADVar]=None):
    """
    单次 Taylor 反传，返回相对于 `inputs` 的梯度与 Hessian。
    若提供 target/targets，则以指定的 ADVar 作为输出目标；否则默认 tape 的最后一个节点。
    """
    # 统一 targets
    if target is not None:
        targets = [target]
    # 注意：可能传进来的 targets 是 ADVar 里的“视图”，我们用 id() 比较更稳
    target_ids = set(id(t) for t in (targets or []))

    order = list(inputs)
    n = len(order)

    g = {}         # id(v) -> grad vec
    H = {}         # id(v) -> Hess matrix
    captured = {}  # 允许在遍历过程中捕获中间节点的 g/H（用于非最后节点目标）

    def _as_leaf(v: ADVar):
        vid = id(v)
        if vid not in g:
            vec = np.zeros((n,), dtype=float)
            try:
                idx = order.index(v)     # 只对“真正的 input 对象”置 1
                vec[idx] = 1.0
            except ValueError:
                pass                     # 常量/非输入：保持 0
            g[vid] = vec
            H[vid] = np.zeros((n, n), dtype=float)

    # forward 扫 tape，增量计算每个 node.out 的 g/H
    for node in global_tape.nodes:
        for (p, _) in node.parents:
            _as_leaf(p)

        f_i = [np.array(local, dtype=float) for (_, local) in node.parents]

        gout = np.zeros((n,), dtype=float)
        for i, (p, _) in enumerate(node.parents):
            gout += f_i[i] * g[id(p)]

        Hout = np.zeros((n, n), dtype=float)
        for i, (p, _) in enumerate(node.parents):
            Hout += f_i[i] * H[id(p)]

        Hloc = _second_derivatives(node.op_tag, node.parents)
        if Hloc.any():
            for i, (pi, _) in enumerate(node.parents):
                gi = g[id(pi)]
                if not np.any(gi):
                    continue
                for j, (pj, _) in enumerate(node.parents):
                    hij = Hloc[i, j]
                    if hij == 0.0:
                        continue
                    gj = g[id(pj)]
                    if not np.any(gj):
                        continue
                    Hout += hij * np.outer(gi, gj)

        g[id(node.out)] = gout
        H[id(node.out)] = Hout

        # 若该节点就是我们想要的目标，则把 g/H 抓取下来（以便后面直接返回）
        if id(node.out) in target_ids:
            captured[id(node.out)] = (gout.copy(), Hout.copy())

    # 选择要返回的 y
    if targets:
        # 若指定了目标，取第一个目标的 g/H
        y_id = id(targets[0])
        if y_id in captured:
            grad = captured[y_id][0]
            Hess = captured[y_id][1]
        else:
            # 目标不是由 tape 生成的新节点（例如只是某个中间变量的别名）：
            # 在这种情况下，直接从字典里取当前值（已在上面填充）
            grad = g[y_id]
            Hess = H[y_id]
    else:
        # 默认：最后一个节点
        y = global_tape.nodes[-1].out
        grad = g[id(y)]
        Hess = H[id(y)]

    return grad, Hess
