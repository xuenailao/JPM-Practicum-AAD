# aad_edge_pushing/__init__.py
from .aad.core import ADVar, grad, reverse
from .aad import ops  # ensure operator overloading is registered

__all__ = ["ADVar", "grad", "reverse", "ops"]
