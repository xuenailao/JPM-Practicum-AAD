# aad/ops/arithmetic.py
import numpy as np
from aad.core.var import ADVar
from aad.core.tape import global_tape

def _as_ad(x, requires_grad=False):
    return x if isinstance(x, ADVar) else ADVar(x, requires_grad=requires_grad)

def _binary(x, y, f, dfdx, dfdy, tag):
    x = _as_ad(x, requires_grad=False)
    y = _as_ad(y, requires_grad=False)
    out = ADVar(f(x.val, y.val))
    global_tape.push_node(
        op_tag=tag, out=out,
        parents=[(x, dfdx(x.val, y.val)), (y, dfdy(x.val, y.val))]
    )
    return out

def add(x, y): return _binary(x, y, lambda a,b:a+b, lambda a,b:1.0, lambda a,b:1.0, "add")
def sub(x, y): return _binary(x, y, lambda a,b:a-b, lambda a,b:1.0, lambda a,b:-1.0, "sub")
def mul(x, y): return _binary(x, y, lambda a,b:a*b, lambda a,b:b,   lambda a,b:a,    "mul")
def div(x, y): return _binary(x, y, lambda a,b:a/b, lambda a,b:1.0/b, lambda a,b:-a/(b*b), "div")

def neg(x):
    x = _as_ad(x, requires_grad=False)
    out = ADVar(-x.val)
    global_tape.push_node(op_tag="neg", out=out, parents=[(x, -1.0)])
    return out

def pow(x, y):
    # Only safe for x>0 when y not integer; minimal demo
    x = _as_ad(x, requires_grad=False)
    y = _as_ad(y, requires_grad=False)
    out = ADVar(x.val ** y.val)
    dfdx = y.val * (x.val ** (y.val - 1.0))
    dfdy = (x.val ** y.val) * (np.log(x.val) if np.all(x.val > 0) else 0.0)
    global_tape.push_node(op_tag="pow", out=out, parents=[(x, dfdx), (y, dfdy)])
    return out

# Bind Python operators to ADVar
ADVar.__add__ = lambda self, other: add(self, other)
ADVar.__radd__ = lambda self, other: add(other, self)
ADVar.__sub__ = lambda self, other: sub(self, other)
ADVar.__rsub__ = lambda self, other: sub(other, self)
ADVar.__mul__ = lambda self, other: mul(self, other)
ADVar.__rmul__ = lambda self, other: mul(other, self)
ADVar.__truediv__ = lambda self, other: div(self, other)
ADVar.__rtruediv__ = lambda self, other: div(other, self)
ADVar.__neg__ = lambda self: neg(self)
ADVar.__pow__ = lambda self, other: pow(self, other)