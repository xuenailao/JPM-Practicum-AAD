# aad/core/var.py
from __future__ import annotations
import numpy as np
from typing import Any, Optional

class ADVar:
    """
    Active variable for reverse-mode AD.

    Fields
    ------
    val : float | np.ndarray
        Forward value.
    adj : float | np.ndarray
        Adjoint (gradient accumulator); same shape as val.
    requires_grad : bool
        If False, this variable is treated as constant (no parents).
    name : Optional[str]
        Debug/pretty name.
    """
    __array_priority__ = 1000  # so numpy ufuncs prefer ADVar hooks

    def __init__(self, val: Any, *, requires_grad: bool = True, name: Optional[str] = None):
        # Type check: only allow numeric scalars, sequences, or numpy arrays
        if not isinstance(val, (int, float, list, tuple, np.ndarray)):
            raise TypeError(
                f"ADVar only accepts numeric types (int, float, list, tuple, ndarray), "
                f"but got {type(val)}"
            )

        # Convert input to numeric format:
        # - list/tuple/ndarray → numpy array
        # - int/float → cast to float
        self.val = np.asarray(val) if isinstance(val, (list, tuple, np.ndarray)) else float(val)

        # Initialize adjoint (gradient accumulator) with same shape as val, filled with zeros
        self.adj = np.zeros_like(self.val, dtype=float)

        # Flag: whether this variable should be tracked for gradients
        self.requires_grad = requires_grad

        # Optional user-provided name for debugging / pretty-printing
        self.name = name

    def __repr__(self):
        # Short label: "req" if requires_grad=True, else "const"
        rg = "req" if self.requires_grad else "const"

        # Return a developer-friendly string with value, grad flag, and name
        return f"ADVar({self.val!r}, {rg}, name={self.name!r})"

