# aad/core/__init__.py
from .var import ADVar
from .tape import global_tape, use_tape
from .engine import reverse, zero_adjoints
from .seeds import grad, value

__all__ = [
    "ADVar",
    "global_tape", "use_tape",
    "reverse", "zero_adjoints",
    "grad", "value",
]

