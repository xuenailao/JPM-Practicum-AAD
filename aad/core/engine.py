# aad/core/engine.py
from __future__ import annotations
import numpy as np
from typing import Sequence, Union
from .tape import global_tape
from .var import ADVar

def zero_adjoints():
    """
    Zero all adjoints found on the current tape's variables (outputs and parents).
    We scan nodes to find reachable ADVars.
    """
    seen = set()
    for node in global_tape.nodes:
        if id(node.out) not in seen:
            _zero(node.out); seen.add(id(node.out))
        for p, _ in node.parents:
            if id(p) not in seen:
                _zero(p); seen.add(id(p))

def _zero(v: ADVar):
    if isinstance(v.adj, np.ndarray):
        v.adj.fill(0.0)
    else:
        v.adj = 0.0

def reverse(outputs: Union[ADVar, Sequence[ADVar]], seed=1.0):
    """
    Run one reverse pass from given output(s).
    - outputs: ADVar or list/tuple of ADVar
    - seed: scalar or same-shaped array to seed the adjoint(s)
    """
    # Seed adjoints
    if isinstance(outputs, (list, tuple)):
        for y in outputs:
            _seed(y, 1.0)
    else:
        _seed(outputs, seed)

    # Backward sweep
    for node in reversed(global_tape.nodes):
        y = node.out
        if _is_zero(y.adj):
            continue  # no contribution to propagate
        for (p, local_partial) in node.parents:
            if not p.requires_grad:
                continue
            # Accumulate: p.adj += y.adj * (∂y/∂p)
            p.adj = p.adj + y.adj * local_partial  # numpy will broadcast if arrays

def _seed(v: ADVar, seed):
    if isinstance(v.adj, np.ndarray):
        v.adj += np.ones_like(v.val, dtype=float) * seed
    else:
        v.adj += float(seed)

def _is_zero(x):
    try:
        return (x == 0).all()
    except Exception:
        return x == 0