# aad/__init__.py
from .core import ADVar, grad, reverse
from . import ops  # ensure operator overloading is registered

__all__ = ["ADVar", "grad", "reverse", "ops"]

