"""
Mutable runtime flags shared across the whole package.

IMPORTANT: always access these as `state.VERBOSE` / `state.ALLOW_REJECT_RULES`
(i.e. `from . import state`) rather than `from .state import VERBOSE`. The
CLI (cli.main) sets these after argument parsing; other modules must see the
live value, and a `from .state import X` import would bind a stale copy at
import time instead of reading the module's current value.
"""
VERBOSE = False
ALLOW_REJECT_RULES = False
