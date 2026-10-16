"""
Shared globals for the Stage-0 token-strip multiprocessing workers.

These are mutated from two different modules: token_strip.py (which
bootstraps/tears down the worker pool) and extractor.py (which clears them
for memory hygiene once Stage 0 is done). Plain `global` statements only
rebind a name within the module that declares them, so to have both modules
mutate the *same* state (matching the original single-file behaviour) we
keep it here and always mutate via attribute access, e.g.
`worker_state.p0_worker_base_set = ...`, never `global p0_worker_base_set`.
"""
p0_worker_base_set = set()
p0_worker_base_by_len = {}
