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

# RNG seed for this run, set by cli.main() from --seed/-S (or left None for
# the old default of "different every run"). Read this rather than caching
# the value, for the same reason as the flags above. See common.set_global_seed().
RNG_SEED = None
