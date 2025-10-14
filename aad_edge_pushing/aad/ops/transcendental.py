# aad/ops/transcendental.py
import numpy as np
from ..core.var import ADVar
from ..core.tape import global_tape
from .arithmetic import _as_ad

def exp(x):
    x = _as_ad(x, requires_grad=False)
    ex = np.exp(x.val)
    out = ADVar(ex)
    out.dot = ex * x.dot
    global_tape.push_node(op_tag="exp", out=out, parents=[(x, ex)])
    return out

def log(x):
    x = _as_ad(x, requires_grad=False)
    out = ADVar(np.log(x.val))
    out.dot = (1.0 / x.val) * x.dot
    global_tape.push_node(op_tag="log", out=out, parents=[(x, 1.0/x.val)])
    return out

def sqrt(x):
    x = _as_ad(x, requires_grad=False)
    s = np.sqrt(x.val)
    out = ADVar(s)
    out.dot = (0.5 / s) * x.dot
    global_tape.push_node(op_tag="sqrt", out=out, parents=[(x, 0.5/s)])
    return out