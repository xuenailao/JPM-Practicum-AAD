# aad/core/seeds.py


#-----------------------------------------------------------------------------
# we "plant" a seed (dy/dy = 1) at the output, then let gradients grow 
# backwards through the tape.
#-----------------------------------------------------------------------------
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Union
import numpy as np

from .var import ADVar
from .tape import use_tape
from .engine import reverse, zero_adjoints


def value(x: Any) -> Any:
    """Return the numeric value of an ADVar or pass-through for plain numbers."""
    return x.val if isinstance(x, ADVar) else x


def _ensure_ad(v: Any, *, name: str, requires_grad: bool = True) -> ADVar:
    """Wrap a plain value as ADVar if needed."""
    return v if isinstance(v, ADVar) else ADVar(v, requires_grad=requires_grad, name=name)


# ----------------------------- single-input grad ----------------------------- #
def grad(f: Callable[[ADVar], ADVar],
         x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Gradient of a scalar-output function y=f(x) at x0 (single input).
    One reverse pass inside a fresh tape.
    """
    with use_tape():
        x = _ensure_ad(x0, name="x", requires_grad=True)
        y = f(x)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")
        # expect scalar output
        if hasattr(y.val, "shape") and getattr(y.val, "shape", ()) != ():
            raise ValueError("grad(f, x0) expects scalar output.")
        zero_adjoints()
        reverse(y, seed=1.0)
        return x.adj


# ----------------------------- multi-input grads ----------------------------- #
def grads(f: Callable[[Dict[str, ADVar]], ADVar],
          inputs: Dict[str, Union[float, np.ndarray]]) -> Dict[str, Union[float, np.ndarray]]:
    """
    Gradient of a scalar-output function y=f(vars) w.r.t. ALL inputs (dict form).
    Runs ONE reverse pass to get every ∂y/∂var simultaneously.

    Parameters
    ----------
    f       : function taking a dict {name: ADVar} and returning scalar ADVar
    inputs  : dict {name: numeric}

    Returns
    -------
    dict {name: numeric}  # gradients in the same keys/order as inputs
    """
    with use_tape():
        vars_ad: Dict[str, ADVar] = {
            k: _ensure_ad(v, name=k, requires_grad=True) for k, v in inputs.items()
        }
        y = f(vars_ad)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")
        # expect scalar output
        if hasattr(y.val, "shape") and getattr(y.val, "shape", ()) != ():
            raise ValueError("grads(f, inputs) expects scalar output.")
        zero_adjoints()
        reverse(y, seed=1.0)
        return {k: vars_ad[k].adj for k in inputs.keys()}


def grads_list(f: Callable[[List[ADVar]], ADVar],
               x0_list: Iterable[Union[float, np.ndarray]]) -> List[Union[float, np.ndarray]]:
    """
    Same as grads(), but inputs are provided as a list; returns a list of partials
    in the same order.

    Example
    -------
    f = lambda xs: xs[0]*xs[0] + 3*xs[1]
    grads_list(f, [2.0, 4.0]) -> [4.0, 3.0]
    """
    with use_tape():
        xs: List[ADVar] = [
            _ensure_ad(v, name=f"x{i}", requires_grad=True) for i, v in enumerate(x0_list)
        ]
        y = f(xs)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")
        # expect scalar output
        if hasattr(y.val, "shape") and getattr(y.val, "shape", ()) != ():
            raise ValueError("grads_list(f, x0_list) expects scalar output.")
        zero_adjoints()
        reverse(y, seed=1.0)
        return [x.adj for x in xs]


