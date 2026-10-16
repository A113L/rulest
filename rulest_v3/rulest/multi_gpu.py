"""
Fan-out layer that runs the extraction pipeline across several GPUs at
once: one GPUEngine (in-process) or ProcessEngineProxy (isolated process)
per device, work split across them, results merged.
"""
import time
import threading
import concurrent.futures
from collections import Counter
from typing import List

from .common import log_info, log_warn, log_error, bold, green
from .gpu_engine import GPUEngine
from .gpu_worker import ProcessEngineProxy

class MultiGPUEngine:
    """Wraps N GPUEngine instances and splits work across them.

    Stage 1 splits base_words across GPUs (word-dominated).
    Stage S and Stage 2 split chains across GPUs (chain-dominated) — each GPU
    tests its chain slice against all base_words. Stage 3 (genetic) also splits
    chains (its per-generation population) across GPUs the same way — see
    GeneticRuleEvolver.evaluate_population, which accepts a MultiGPUEngine
    directly rather than being handed a single engine.
    """

    def __init__(self, params):
        self._params = params
        self.engines: List['GPUEngine'] = []
        # Adaptive load balancing: measured throughput per (engine, work-kind),
        # smoothed with an EMA. Splits then default to equal shares until we
        # have at least one real measurement, then shift toward proportional-
        # to-speed shares. Kept on the MultiGPUEngine (not per-stage) so a slow
        # card (e.g. an Intel iGPU alongside a discrete GPU) gets calibrated
        # once and every later stage — including Stage 3's GA, which reuses
        # this same instance — starts from a sane split instead of 50/50.
        self._speed_ema = {}      # (id(eng), kind) -> items/sec
        self._speed_alpha = 0.5   # EMA weight for the newest sample
        self._warned_slow = set() # kinds we've already logged a slow-card warning for

    def record_speed(self, eng, kind, n_items, elapsed):
        if n_items <= 0 or elapsed <= 0.01:
            return
        rate = n_items / elapsed
        key = (id(eng), kind)
        prev = self._speed_ema.get(key)
        self._speed_ema[key] = rate if prev is None else (self._speed_alpha*rate + (1-self._speed_alpha)*prev)

    def vendor_ratio_split(self, items, ratios, engines=None):
        """Split `items` across `engines` using a fixed per-vendor ratio instead
        of measured throughput. `ratios` maps an uppercase substring to match
        against the device name (e.g. 'NVIDIA', 'INTEL') -> fraction. Only used
        for Stage S (seed pass); Stage 2 keeps its normal adaptive
        weighted_split(kind='chains') behaviour untouched.

        Falls back to weighted_split (adaptive/50-50) if we can't confidently
        match every engine to exactly one vendor key, so an unexpected device
        mix never silently drops work."""
        engines = self.engines if engines is None else engines
        n = len(engines)
        if n <= 1 or not items:
            return [items]
        names = []
        for eng in engines:
            dev = getattr(eng, 'device', None)
            nm = getattr(dev, 'name', None)
            names.append(nm.upper() if nm else '')
        weights = []
        for nm in names:
            matched = [w for key, w in ratios.items() if key in nm]
            if len(matched) != 1:
                log_warn("[SEED]    Vendor split ratio couldn't uniquely match all devices "
                         "— falling back to adaptive split for this pass.")
                return self.weighted_split(items, kind='chains', engines=engines)
            weights.append(matched[0])
        wsum = sum(weights) or 1.0
        weights = [w/wsum for w in weights]
        sizes = [max(1, int(round(w*len(items)))) for w in weights]
        diff = len(items) - sum(sizes)
        sizes[-1] += diff
        sizes[-1] = max(1, sizes[-1]) if sizes[-1] > 0 else sizes[-1]
        slices = []; idx = 0
        for sz in sizes:
            sz = max(0, sz)
            slices.append(items[idx:idx+sz])
            idx += sz
        return slices

    def weighted_split(self, items, kind, min_frac=0.03, engines=None):
        """Split `items` across `engines` (default: self.engines) proportionally
        to each engine's measured throughput for this work `kind` (words/chains/
        ga), instead of always slicing evenly. Falls back to an equal split for
        any engine we have no measurement for yet, so the very first batch of a
        run is always plain 50/50 (there's nothing to weight by until something
        has actually been timed) and subsequent batches self-correct. Every
        engine is guaranteed at least `min_frac` of the items (default 3%) so
        a slow card still gets *some* work each round and its speed estimate
        keeps getting refreshed, rather than starving to zero and never being
        re-measured (e.g. if it happened to be busy compiling on round one).

        Always pass the exact engine list the caller is about to zip() the
        returned slices against (e.g. after filtering out disabled engines) —
        this must default to self.engines rather than assuming it, since a
        stale/mismatched engine list here would misalign slices to engines."""
        engines = self.engines if engines is None else engines
        n = len(engines)
        if n <= 1 or not items:
            return [items]
        speeds = [self._speed_ema.get((id(eng), kind)) for eng in engines]
        if all(s is None for s in speeds):
            chunk = max(1, (len(items) + n - 1) // n)
            return [items[i*chunk:(i+1)*chunk] for i in range(n)]
        # Engines with no measurement yet: assume the median known speed so
        # they get a fair-ish initial slice rather than being starved to the
        # floor before we've ever timed them.
        known = [s for s in speeds if s is not None]
        fallback = sorted(known)[len(known)//2]
        speeds = [s if s is not None else fallback for s in speeds]
        total = sum(speeds) or 1.0
        weights = [max(min_frac, s/total) for s in speeds]
        wsum = sum(weights)
        weights = [w/wsum for w in weights]
        # Warn once per work-kind if one engine is dramatically slower — the
        # user asked specifically about this (Intel iGPU next to a discrete
        # GPU): worth flagging since even a perfectly weighted split still
        # pays some coordination overhead for a card contributing almost
        # nothing.
        if kind not in self._warned_slow and len(known) == n:
            fastest = max(speeds); slowest = min(speeds)
            if fastest > 0 and slowest/fastest < 0.15:
                slow_idx = speeds.index(slowest)
                dev = getattr(engines[slow_idx], 'device', None)
                dev_name = getattr(dev, 'name', 'device').strip() if dev else f"GPU {slow_idx}"
                log_warn(f"[GPU]  {dev_name} is measured at <15% the speed of your fastest GPU for "
                         f"'{kind}' work — consider excluding it via --device to avoid it dragging "
                         f"down the combined throughput.")
                self._warned_slow.add(kind)
        sizes = [max(1, int(round(w*len(items)))) for w in weights]
        # Rounding can over/under-allocate by a few items — fix up the last
        # slice so slices always sum to exactly len(items).
        diff = len(items) - sum(sizes)
        sizes[-1] += diff
        sizes[-1] = max(1, sizes[-1]) if sizes[-1] > 0 else sizes[-1]
        slices = []
        idx = 0
        for sz in sizes:
            sz = max(0, sz)
            slices.append(items[idx:idx+sz])
            idx += sz
        return slices

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value
        for eng in self.engines:
            eng.params = value

    @property
    def device(self):
        return self.engines[0].device if self.engines else None

    def initialize_gpu(self, device_specs):
        # Each GPU gets its own OS process (see ProcessEngineProxy above) so a
        # single wedged driver can be hard-killed instead of hanging the run.
        for spec in device_specs:
            eng = ProcessEngineProxy(self._params, spec)
            if eng.initialize_gpu(spec):
                self.engines.append(eng)
            else:
                log_warn(f"[GPU]  Skipping device {spec} — init failed")
        if not self.engines:
            return False
        if len(self.engines) == 1:
            log_info("[GPU]  Only one GPU initialised — falling back to single-GPU mode")
        else:
            log_info(f"[GPU]  {bold(str(len(self.engines)))} GPUs active for parallel processing "
                      f"(process-isolated — a hung driver on one GPU is killed without affecting the others)")
        return True

    def _prune_disabled(self):
        """Drop any engine whose kernel build timed out (see GPUEngine.disabled).
        That device's driver may still be stuck holding an internal lock, so we
        stop issuing it further work and continue with whatever GPU(s) remain
        instead of retrying (and potentially hanging on) it again next stage."""
        alive = []
        for eng in self.engines:
            if eng.disabled:
                name = eng.device.name.strip() if eng.device else "unknown device"
                log_warn(f"[GPU]  Dropping unresponsive device ({name}) — continuing with remaining GPU(s)")
            else:
                alive.append(eng)
        removed = len(self.engines) - len(alive)
        self.engines = alive
        return removed

    def compile_kernel(self, force=False):
        # Recompiling right on the heels of concurrent multi-GPU execution
        # (Stage 1 / Seed Pass just ran both devices at once) has been seen to
        # trigger a build-time hang on mixed-vendor setups (e.g. NVIDIA +
        # Intel) even though each device compiles fine completely on its own
        # — some driver stacks share internal locking across vendors within
        # the same process that a single-GPU run never exercises. Draining
        # every engine's queue first, and giving the driver stack a brief
        # moment to settle, avoids issuing a build while another vendor's
        # driver may still be mid-transition.
        if len(self.engines) > 1:
            for eng in self.engines:
                if eng.queue is not None:
                    eng._safe_queue_finish()
            time.sleep(0.5)
        ok = True
        for eng in self.engines:
            ok = eng.compile_kernel(force=force) and ok
        self._prune_disabled()
        if not self.engines:
            log_error("[GPU]  All GPUs became unresponsive during kernel compile")
            return False
        return ok

    def generate_bloom_filter_gpu(self, target_words):
        primary = self.engines[0]
        # ProcessEngineProxy.generate_bloom_filter_gpu already resolves the
        # "already on this GPU's VRAM" sentinel server-side (inside the
        # worker process, where the real cl.Buffer/queue actually live) and
        # always hands back a plain numpy array here — never a live GPU
        # handle, since that can't cross a process boundary.
        bf = primary.generate_bloom_filter_gpu(target_words)
        if len(self.engines) > 1 and bf is not None:
            for eng in self.engines[1:]:
                eng.upload_bloom_filter(bf)
        return bf

    def upload_bloom_filter(self, bf):
        for eng in self.engines:
            eng.upload_bloom_filter(bf)

    def build_numeric_seed_families(self, max_depth):
        return self.engines[0].build_numeric_seed_families(max_depth)

    def prepare_rule_pool(self, chains):
        """Broadcast a rule_index/gpu_rules rebuild (scoped to `chains`) to
        EVERY active engine — not just the primary. evaluate_population()
        (see genetic.py) may shard a batch's chains across all engines, so
        every engine needs the same complete token universe for that batch,
        regardless of which slice it personally ends up evaluating. See
        GPUEngine.prepare_rule_pool() for why this needs to be an explicit,
        unconditional rebuild rather than an on-demand/cached one.
        """
        ok = True
        for eng in self.engines:
            if getattr(eng, 'disabled', False):
                continue
            try:
                if not eng.prepare_rule_pool(chains):
                    ok = False
            except Exception as exc:
                name = self._engine_label(self.engines.index(eng), eng)
                log_warn(f"[GPU]  prepare_rule_pool failed on {name}: {exc}")
                ok = False
        return ok

    def verify_pairs_exact(self, base_words, selected_chains, pairs, batch_size=200000):
        """Verify ``(chain_idx, base_word_idx)`` pairs across all active GPUs.

        The exact verifier is used while building target-word coverage, so
        keeping this single-GPU would serialize the most expensive part of the
        otherwise multi-GPU coverage pipeline.  Pairs are sharded (not
        ``base_words``) because each pair is an independent GPU workload and
        the caller requires results in the original pair order.

        The same ``selected_chains`` and ``base_words`` are sent to every
        worker, while each worker receives only its pair slice.  This avoids
        duplicating the pair list in the GPU kernel and lets the existing
        process-isolated ``ProcessEngineProxy`` instances execute concurrently.

        ``weighted_split(kind='exact')`` uses measured exact-verification
        throughput after the first round.  The first round is equal-split, then
        the split self-corrects for heterogeneous GPUs.

        Returns a list aligned 1:1 with ``pairs``.  A worker failure leaves
        ``None`` for that worker's slice and is logged; successful workers'
        results are never reordered.
        """
        if not pairs:
            return []
        if not self.engines:
            return [None] * len(pairs)

        if len(self.engines) == 1:
            t0 = time.time()
            result = self.engines[0].verify_pairs_exact(
                base_words, selected_chains, pairs, batch_size=batch_size
            )
            self.record_speed(self.engines[0], 'exact', len(pairs), time.time() - t0)
            self._prune_disabled()
            return result

        # Shard the actual verification work.  Splitting base_words would be
        # wrong here because ``pairs`` already identifies the exact work that
        # must be performed and may be very sparse.
        slices = self.weighted_split(pairs, kind='exact')
        sizes_str = "/".join(f"{len(s):,}" for s in slices)
        log_info(
            f"[GPU]  EXACT VERIFY split across {len(self.engines)} GPUs "
            f"({sizes_str} pairs)"
        )

        results = [None] * len(pairs)
        jobs = []
        offset = 0
        for eng, sl in zip(self.engines, slices):
            if sl:
                jobs.append((eng, sl, offset))
            offset += len(sl)

        def _run(eng, sl):
            t0 = time.time()
            result = eng.verify_pairs_exact(
                base_words, selected_chains, sl, batch_size=batch_size
            )
            self.record_speed(eng, 'exact', len(sl), time.time() - t0)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            future_map = {
                pool.submit(_run, eng, sl): (eng, sl, start)
                for eng, sl, start in jobs
            }
            for future, (eng, sl, start) in (
                (f, future_map[f]) for f in future_map
            ):
                try:
                    worker_result = future.result()
                    if worker_result is None:
                        worker_result = [None] * len(sl)
                    if len(worker_result) != len(sl):
                        raise RuntimeError(
                            f"exact verifier returned {len(worker_result)} results "
                            f"for {len(sl)} pairs"
                        )
                    results[start:start + len(sl)] = worker_result
                except Exception as exc:
                    name = self._engine_label(self.engines.index(eng), eng)
                    log_warn(
                        f"[GPU]  EXACT VERIFY worker {name} failed for "
                        f"{len(sl):,} pairs: {exc}"
                    )
                    # Keep the slice as None.  Do not shift/reorder results
                    # from other GPUs.  The caller treats None as an
                    # unverified pair, preventing false coverage claims.

        self._prune_disabled()
        return results

    def shutdown(self):
        """Cleanly terminate every worker process. Safe to call multiple
        times / on partially-initialised state."""
        for eng in self.engines:
            try: eng.terminate()
            except Exception: pass

    def _parallel_run(self, fn_per_engine, slices, kind=None):
        """Runs fn_per_engine(eng, slice) for every non-empty (engine, slice) pair
        concurrently. When `kind` is given, times each engine's call and feeds
        the result into record_speed() so the *next* weighted_split() call for
        that work kind can account for it — this is how the split self-corrects
        from an initial 50/50 toward each card's actual measured throughput."""
        n = len(self.engines)
        def _run(eng, sl):
            t0 = time.time()
            r = fn_per_engine(eng, sl)
            if kind is not None:
                self.record_speed(eng, kind, len(sl), time.time()-t0)
            return r
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(_run, eng, sl)
                    for eng, sl in zip(self.engines, slices) if sl]
            merged = Counter()
            for f in futs:
                try:
                    merged.update(f.result())
                except Exception as exc:
                    log_warn(f"[GPU]  Worker thread error: {exc}")
        return merged

    def _engine_label(self, idx, eng):
        dev = getattr(eng, 'device', None)
        name = getattr(dev, 'name', None)
        name = name.strip() if name else None
        return f"GPU{idx}:{name}" if name else f"GPU{idx}"

    def process_all_words_single_rule(self, base_words, rules, bloom_filter):
        if len(self.engines) == 1:
            return self.engines[0].process_all_words_single_rule(base_words, rules, bloom_filter)
        slices = self.weighted_split(base_words, kind='words')
        sizes_str = "/".join(f"{len(s):,}" for s in slices)
        log_info(f"[GPU]  STAGE 1 split across {len(self.engines)} GPUs ({sizes_str} words; "
                 f"one progress bar per device)")
        shared_combos = [0]; shared_lock = threading.Lock()
        return self._parallel_run(
            lambda eng, sl: eng.process_all_words_single_rule(
                sl, rules, bloom_filter, _silent=False,
                _shared_combos=shared_combos, _shared_lock=shared_lock,
                _position=self.engines.index(eng), _label=self._engine_label(self.engines.index(eng), eng)),
            slices, kind='words')

    def _ensure_engine_ready(self, eng, bloom_filter, phase1_rules):
        if eng.bloom_buf is None: eng.upload_bloom_filter(bloom_filter)
        if not eng.program: eng.compile_kernel()
        if not eng.rule_index: eng.ensure_rules_ready(phase1_rules)

    def run_seed_extraction_pass(self, base_words, sbd, bloom_filter, phase1_rules):
        if len(self.engines) == 1:
            return self.engines[0].run_seed_extraction_pass(base_words, sbd, bloom_filter, phase1_rules)
        # Pre-extract chains from sbd so we can split them across GPUs
        multi = []
        for depth, chains in sorted(sbd.items()):
            if depth >= 2:
                multi.extend(chains)
        if not multi:
            log_info("[SEED]    No multi-depth seeds")
            return Counter()
        total = len(multi)
        d_levels = sorted(d for d in sbd if d >= 2)
        depth_range_str = (f"d{d_levels[0]}–d{d_levels[-1]}" if len(d_levels) > 1
                           else (f"d{d_levels[0]}" if d_levels else "none"))
        log_info(f"[SEED]    Numeric seed pass: {total:,} chains across {len(d_levels)} depth level(s) ({depth_range_str})")
        if len(self.engines) > 1:
            for eng in self.engines:
                if eng.queue is not None:
                    eng._safe_queue_finish()
            time.sleep(0.5)
        for eng in self.engines:
            self._ensure_engine_ready(eng, bloom_filter, phase1_rules)
        self._prune_disabled()
        if not self.engines:
            log_error("[SEED]    All GPUs became unresponsive — aborting seed pass")
            return Counter()
        if len(self.engines) == 1:
            result = self.engines[0]._run_chains_against_words(base_words, multi, "  SEED PASS ")
            log_info(f"[SEED]    {bold(green(str(len(result))))} unique seed chains passed bloom filter")
            return result
        # Stage S (seed pass) uses a fixed 75/25 NVIDIA/Intel split rather than
        # the adaptive throughput-based split, per user request. Stage 2 below
        # is untouched and keeps using weighted_split(kind='chains').
        chain_slices = self.vendor_ratio_split(multi, {'NVIDIA': 0.75, 'INTEL': 0.25})
        sizes_str = "/".join(f"{len(s):,}" for s in chain_slices)
        log_info(f"[GPU]  SEED split across {len(self.engines)} GPUs ({sizes_str} chains; "
                 f"one progress bar per device)")
        shared_combos = [0]; shared_lock = threading.Lock()
        merged = self._parallel_run(
            lambda eng, sl: eng._run_chains_against_words(
                base_words, sl, "  SEED PASS ", False,
                shared_combos, shared_lock,
                _position=self.engines.index(eng), _label=self._engine_label(self.engines.index(eng), eng)),
            chain_slices, kind='chains')
        log_info(f"[SEED]    {bold(green(str(len(merged))))} unique seed chains passed bloom filter")
        return merged

    def process_all_words_chain_rules(self, base_words, rules, max_depth, bloom_filter,
                                       single_counter, seed_chains=None, prebuilt_sbd=None):
        if len(self.engines) == 1:
            return self.engines[0].process_all_words_chain_rules(
                base_words, rules, max_depth, bloom_filter, single_counter, seed_chains, prebuilt_sbd)
        # Generate chains once on the primary engine, then split across GPUs
        primary = self.engines[0]
        if primary.bloom_buf is None: primary.upload_bloom_filter(bloom_filter)
        if not primary.rule_index: primary.ensure_rules_ready(rules)
        chains = primary.generate_informed_chains(rules, single_counter, max_depth, seed_chains, prebuilt_sbd)
        if not chains: return Counter()
        if len(self.engines) > 1:
            for eng in self.engines:
                if eng.queue is not None:
                    eng._safe_queue_finish()
            time.sleep(0.5)
        for eng in self.engines[1:]:
            self._ensure_engine_ready(eng, bloom_filter, rules)
        self._prune_disabled()
        if len(self.engines) == 1:
            result = primary._run_chains_against_words(base_words, chains, "  STAGE 2 ")
            log_info(f"[S2]    {bold(green(str(len(result))))} unique chain rules passed bloom filter")
            return result
        chain_slices = self.weighted_split(chains, kind='chains')
        sizes_str = "/".join(f"{len(s):,}" for s in chain_slices)
        log_info(f"[GPU]  STAGE 2 split across {len(self.engines)} GPUs ({sizes_str} chains; "
                 f"one progress bar per device)")
        shared_combos = [0]; shared_lock = threading.Lock()
        merged = self._parallel_run(
            lambda eng, sl: eng._run_chains_against_words(
                base_words, sl, "  STAGE 2 ", False,
                shared_combos, shared_lock,
                _position=self.engines.index(eng), _label=self._engine_label(self.engines.index(eng), eng)),
            chain_slices, kind='chains')
        log_info(f"[S2]    {bold(green(str(len(merged))))} unique chain rules passed bloom filter")
        return merged
