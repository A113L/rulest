"""
GPUExtractor: the top-level pipeline orchestrator. Wires together device
selection, Stage 0 (token-strip), Stage 1 (single-rule GPU sweep), Stage S
(built-in numeric/pattern seed families), Stage 2 (rule-chain GPU search),
and Stage 3 (--genetic) into one `extract_rules()` call, and owns the
single-vs-multi-GPU decision.
"""
import time
from collections import Counter, defaultdict

from . import worker_state
from .common import (
    log_info, log_warn, log_error, log_debug, log_section,
    bold, green, cyan, yellow, _kb, HashcatRuleValidator,
    MAX_HASHCAT_CHAIN, TIME_SAFETY_FACTOR,
)
from .devices import get_all_devices, get_all_gpu_only_devices, calculate_dynamic_parameters
from .kernel_source import GPUCompatibleRulesGenerator
from .gpu_engine import GPUEngine
from .multi_gpu import MultiGPUEngine
from .genetic import GeneticRuleEvolver
from .token_strip import (
    extract_token_strip_rules, _log_token_strip_stats, _generate_toggle_chain_seeds,
)
from .profiling import stage_profiler

class GPUExtractor:
    def __init__(self, base_count, target_count, max_depth, device_spec=None,
                 target_hours=0.5, max_chains=None, seed_rules_file=None, bloom_mb=None,
                 builtin_seeds=True, bloom_no_shard=False,
                 genetic=False, genetic_generations=50, genetic_pop=200, genetic_elite=0.15,
                 token_strip=False, token_strip_min_stem=4, token_strip_max_prefix=4,
                 token_strip_max_suffix=4, token_strip_min_leet_amb=3,
                 token_strip_workers=0, token_strip_chunk_size=0):
        self.base_count = base_count
        self.target_count = target_count
        self.user_max_depth = max_depth
        self.device_spec = device_spec
        self.max_chains = max_chains
        self.seed_rules_file = seed_rules_file
        self.bloom_mb = bloom_mb
        self.builtin_seeds = builtin_seeds
        self.bloom_no_shard = bloom_no_shard
        self.genetic = genetic
        self.genetic_generations = genetic_generations
        self.genetic_pop = genetic_pop
        self.genetic_elite = genetic_elite
        self.token_strip = token_strip
        self.token_strip_min_stem = token_strip_min_stem
        self.token_strip_max_prefix = token_strip_max_prefix
        self.token_strip_max_suffix = token_strip_max_suffix
        self.token_strip_min_leet_amb = token_strip_min_leet_amb
        self.token_strip_workers = token_strip_workers
        self.token_strip_chunk_size = token_strip_chunk_size
        self.params = calculate_dynamic_parameters(base_count, target_count, None, target_hours,
                                                   bloom_mb_override=bloom_mb, bloom_no_shard=bloom_no_shard)
        self.params['MAX_CHAIN_DEPTH'] = self.user_max_depth
        self.rules_gen = GPUCompatibleRulesGenerator()
        self.gpu_engine = None
        self.validator = HashcatRuleValidator()

    def load_seed_rules(self):
        if not self.seed_rules_file: return []
        seeds = []
        try:
            with open(self.seed_rules_file,'r',encoding='latin-1') as f:
                for line in f:
                    line=line.strip('\n\r')
                    if line and not line.startswith('#') and self.validator.validate_rule_for_gpu(line):
                        seeds.append(line)
            log_info(f"[SEED] Loaded {bold(str(len(seeds)))} seed rules")
        except Exception as e: log_warn(f"Seed load failed: {e}")
        return seeds

    def _note_gpu_phase(self, stage_name, source=None):
        """Fold the just-finished GPU call's detailed host/event breakdown
        (GpuPhaseProfile, same shape CELF's profiler uses) into `stage_name`'s
        row in the unified profiling table. Call this right after a stage's
        GPU call returns, before the next GPU-heavy call overwrites the
        engine's `_last_gpu_phase_profile`. `source` defaults to
        self.gpu_engine; STAGE 3 passes the GeneticRuleEvolver instead, since
        its GPU time is accumulated across a whole run of generations rather
        than attached to a single engine call. Silently does nothing when
        the source has no such data (not profiling, or a MultiGPUEngine/
        ProcessEngineProxy run that never populated it) -- the coarse
        per-stage wall time keeps working regardless."""
        getter = getattr(source if source is not None else self.gpu_engine,
                          "last_gpu_phase_summary", None)
        if getter is None:
            return
        try:
            summary = getter()
        except Exception:
            return
        if summary:
            stage_profiler.note(stage_name, extra=summary)

    def extract_rules(self, base_words, target_words, **depth_overrides):
        all_counts = Counter()
        rules = self.rules_gen.generate_gpu_compatible_rules()
        ts_singles = []
        ts_chains = []
        ts_sbd = defaultdict(set)
        builtin_set = set(rules)
        all_seeds = self.load_seed_rules()
        n_s0_chains_to_stage2 = 0
        ts_extra_singles = []  # defined early to avoid UnboundLocalError

        if self.token_strip:
            log_section("STAGE 0 — Token-Strip Rule Extraction (Core + Insert)")
            with stage_profiler.track("STAGE 0 — Token-Strip", unit="word") as _s0t:
                base_set_ts = set(base_words)
                log_info(f"[S0]    {len(target_words):,} target words  base {len(base_set_ts):,}  min-stem={self.token_strip_min_stem}  prefix={self.token_strip_max_prefix}  suffix={self.token_strip_max_suffix}  leet-amb={self.token_strip_min_leet_amb}")
                s0_max_depth = min(self.user_max_depth, MAX_HASHCAT_CHAIN)
                ts_all = extract_token_strip_rules(target_words, base_set_ts, max_depth=s0_max_depth,
                                                   min_stem_len=self.token_strip_min_stem,
                                                   max_prefix_len=self.token_strip_max_prefix,
                                                   max_suffix_len=self.token_strip_max_suffix,
                                                   max_leet_ambiguity=self.token_strip_min_leet_amb,
                                                   workers=self.token_strip_workers,
                                                   chunk_size=self.token_strip_chunk_size)
                for r in ts_all:
                    d = len(r.split())
                    if d==1: ts_singles.append(r)
                    else:
                        if d <= self.user_max_depth:
                            ts_chains.append(r)
                            ts_sbd[d].add(r)
                        else:
                            log_debug(f"[S0] Dropping chain depth {d} > user max depth {self.user_max_depth}: {r}")
                _log_token_strip_stats(len(target_words), ts_all, inject_sbd=self.builtin_seeds)
                if self.user_max_depth >= 2:
                    toggle_max_depth = min(12, self.user_max_depth)
                    toggle = _generate_toggle_chain_seeds(toggle_max_depth)
                    for tc in toggle:
                        d = len(tc.split())
                        if d>=2 and d <= self.user_max_depth:
                            ts_chains.append(tc)
                            ts_sbd.setdefault(d,set()).add(tc)
                            n_s0_chains_to_stage2 += 1
                ts_extra_singles = [r for r in ts_singles if r not in builtin_set]
                if ts_chains:
                    all_seeds = list(all_seeds) + ts_chains
                n_s0_chains_to_stage2 += len(ts_chains)
                _s0t.set_count(len(target_words))

        is_explicit_multi = (
            self.device_spec is not None
            and self.device_spec != 'all'
            and ',' in self.device_spec
        )
        if self.device_spec == 'all' or is_explicit_multi:
            if is_explicit_multi:
                # User (or the RCR GUI) passed an explicit comma-separated
                # device list, e.g. "--device 0,1". Use exactly those specs
                # instead of auto-discovering every GPU on the system.
                specs = [s.strip() for s in self.device_spec.split(',') if s.strip()]
                if not specs:
                    log_error(f"[GPU]  No valid device IDs parsed from '{self.device_spec}'"); return all_counts
            else:
                gpu_devs = get_all_gpu_only_devices()
                if not gpu_devs:
                    log_error("[GPU]  No GPU devices found for --device all"); return all_counts
                all_devs = get_all_devices()
                specs = [str(i) for i, (_p, d) in enumerate(all_devs) if d in gpu_devs]
            if len(specs) > 1:
                self.gpu_engine = MultiGPUEngine(self.params)
                if not self.gpu_engine.initialize_gpu(specs):
                    return all_counts
            else:
                self.gpu_engine = GPUEngine(self.params)
                if not self.gpu_engine.initialize_gpu(specs[0] if specs else None):
                    return all_counts
        else:
            self.gpu_engine = GPUEngine(self.params)
            if not self.gpu_engine.initialize_gpu(self.device_spec):
                return all_counts
        self.params = calculate_dynamic_parameters(self.base_count, self.target_count, self.gpu_engine.device,
                                                   self.params['TARGET_SECONDS']/3600,
                                                   bloom_mb_override=self.bloom_mb, bloom_no_shard=self.bloom_no_shard)
        seed_pass_depth = max(9, self.user_max_depth)  # seed pass explores numeric/year patterns that need
                                                        # depth >=9 regardless of the user's general --max-depth
        self.params['MAX_CHAIN_DEPTH'] = seed_pass_depth
        self.gpu_engine.params = self.params

        extra_seeds = [s for s in all_seeds if ' ' not in s.strip()]
        extra_valid = [s for s in extra_seeds if s not in builtin_set]
        rules_phase1 = list(dict.fromkeys(rules + ts_extra_singles + extra_valid))
        seed_chains = [s for s in all_seeds if ' ' in s.strip() and len(s.split()) <= self.user_max_depth]

        if extra_valid: log_info(f"[SEED] {len(extra_valid)} seed single-rule(s) added to STAGE 1")
        if seed_chains and self.user_max_depth<2: log_warn(f"[SEED] {len(seed_chains)} chain seed(s) ignored — requires --max-depth >= 2")

        worker_state.p0_worker_base_set = set()
        worker_state.p0_worker_base_by_len = {}

        if not base_words or not target_words:
            log_warn("Empty wordlist(s), aborting extraction")
            return Counter()
        log_info("[GPU]  Building bloom filter on GPU …")
        with stage_profiler.track("Bloom filter build", unit="word") as _bft:
            if not self.gpu_engine.compile_kernel(): return all_counts
            bloom_filter = self.gpu_engine.generate_bloom_filter_gpu(target_words)
            self.gpu_engine.upload_bloom_filter(bloom_filter)
            _bft.set_count(len(target_words))
            self._note_gpu_phase("Bloom filter build")

        log_section("STAGE 1 — Single Rule Search")
        seed_note = f"  ({len(extra_valid)} from seeds)" if extra_valid else ""
        log_info(f"[S1]    {len(base_words):,} base words × {len(rules_phase1):,} atomic rules{seed_note}")
        t0 = time.time()
        with stage_profiler.track("STAGE 1 — Single Rule Search", unit="combo") as _s1t:
            single = self.gpu_engine.process_all_words_single_rule(base_words, rules_phase1, bloom_filter)
            _s1t.set_count(len(base_words) * len(rules_phase1))
            self._note_gpu_phase("STAGE 1 — Single Rule Search")
        t1 = time.time()
        all_counts.update(single)

        seed_hits = Counter()
        if self.builtin_seeds and not _kb.poll():
            log_section("STAGE S — Seed Extraction (families A-M)")
            with stage_profiler.track("STAGE S — Seed Extraction") as _sst:
                sbd = self.gpu_engine.build_numeric_seed_families(max_depth=seed_pass_depth)
                if ts_sbd:
                    n_injected = 0
                    n_ts_total = 0
                    for depth, chains in ts_sbd.items():
                        n_ts_total += len(chains)
                        before = len(sbd.setdefault(depth, set()))
                        sbd[depth].update(chains)
                        n_injected += len(sbd[depth]) - before
                    n_dupes = n_ts_total - n_injected
                    dupe_str = f"  ({n_dupes:,} already in builtin families)" if n_dupes else ""
                    if n_injected:
                        log_info(f"[SEED]    STAGE 0 injected {bold(cyan(str(n_injected)))} chain(s) into STAGE S sbd{dupe_str}")
                seed_hits = self.gpu_engine.run_seed_extraction_pass(base_words, sbd, bloom_filter, rules_phase1)
                all_counts.update(seed_hits)
                _sst.set_count(sum(len(v) for v in sbd.values()), unit="chain")
                self._note_gpu_phase("STAGE S — Seed Extraction")
            ts = time.time()
        else:
            if _kb.quit_requested:
                log_warn("[SEED]    Skipped — early exit")
                stage_profiler.note("STAGE S — Seed Extraction", extra="skipped: early exit")
            else:
                log_info(f"[SEED]    {yellow('Skipped')} (--no-builtin-seeds)")
                stage_profiler.note("STAGE S — Seed Extraction", extra="skipped: --no-builtin-seeds")
            sbd = {}
            ts = t1

        self.params['MAX_CHAIN_DEPTH'] = self.user_max_depth
        self.gpu_engine.params = self.params
        stage2_gpu_ready = self.gpu_engine.compile_kernel(force=True)
        if not stage2_gpu_ready:
            log_warn("Failed to recompile kernel for Stage 2 depth")
        # If this is a MultiGPUEngine and every device ended up disabled (e.g. a
        # driver hung during compile and was dropped), or a single GPUEngine got
        # poisoned the same way, there's nothing left to run Stage 2 on — skip it
        # cleanly instead of calling into a dead engine.
        no_gpu_left = (hasattr(self.gpu_engine, 'engines') and not self.gpu_engine.engines) or \
                      (not hasattr(self.gpu_engine, 'engines') and self.gpu_engine.context is None)
        if no_gpu_left:
            log_error("[S2]    No usable GPU remaining — skipping STAGE 2 and STAGE 3")

        if self.user_max_depth > 1 and not _kb.poll() and not no_gpu_left:
            log_section("STAGE 2 — Rule Chain Search")
            log_info(f"[S2]    {bold(cyan(str(n_s0_chains_to_stage2)))} rules from STAGE 0 injected into chain search")

            if self.genetic and self.user_max_depth >= 2:
                _min_ga = 120.0
                _ga_frac = 0.20
                reserved_ga = max(_min_ga, self.params['TARGET_SECONDS'] * _ga_frac)
            else:
                reserved_ga = 0.0
            remaining = max(0, self.params['TARGET_SECONDS'] - (t1-t0) - reserved_ga)
            budget = remaining * self.params['EST_COMBOS_PER_SEC'] * TIME_SAFETY_FACTOR
            depths = list(range(2, self.user_max_depth+1))
            if budget>0 and base_words and depths:
                depth_budgets = {d: max(0, int(budget/len(depths)/(len(base_words)*d))) for d in depths}
            else:
                depth_budgets = {d:0 for d in depths}
            MIN_CHAINS = 5_000
            depth_budgets = {d: max(v, MIN_CHAINS) for d,v in depth_budgets.items()}
            for d in depths:
                key = f'depth{d}_override'
                if key in depth_overrides and depth_overrides[key] is not None:
                    depth_budgets[d] = depth_overrides[key]
                depth_budgets[d] = max(0, depth_budgets[d])
            if self.max_chains:
                total = sum(depth_budgets.values())
                if total > self.max_chains:
                    scale = self.max_chains / total
                    depth_budgets = {d: int(v*scale) for d,v in depth_budgets.items()}
            for d, bgt in depth_budgets.items():
                self.params[f'CHAIN_GEN_LIMIT_{d}'] = bgt
            log_info(f"[S2]    depth 2-{self.user_max_depth} | " + " | ".join(f"d{d}:{v:,}" for d,v in depth_budgets.items()))
            # IMPORTANT: with a single in-process GPUEngine, self.params and
            # gpu_engine.params were always the *same* dict object, so mutating
            # self.params in place (just above) was automatically visible to the
            # engine — no re-push needed. With process-isolated workers
            # (MultiGPUEngine/ProcessEngineProxy) that's no longer true: each
            # "params =" assignment ships a *copy* of the dict into the worker
            # process over IPC, so any mutation made afterwards (like the
            # CHAIN_GEN_LIMIT_* budgets we just computed) never reaches the
            # workers unless we push it again here. Without this, every worker
            # reads CHAIN_GEN_LIMIT_* as 0, generates ~no chains, and STAGE 2
            # finishes suspiciously fast with an almost-empty result.
            self.gpu_engine.params = self.params
            with stage_profiler.track("STAGE 2 — Rule Chain Search", unit="chain") as _s2t:
                chains = self.gpu_engine.process_all_words_chain_rules(base_words, rules_phase1, self.user_max_depth,
                                                                       bloom_filter, single, seed_chains=seed_chains,
                                                                       prebuilt_sbd=sbd)
                _s2t.set_count(sum(depth_budgets.values()))
                self._note_gpu_phase("STAGE 2 — Rule Chain Search")
            all_counts.update(chains)
        else:
            reason = "early exit" if _kb.poll() else ("no usable GPU" if no_gpu_left else "--max-depth 1")
            stage_profiler.note("STAGE 2 — Rule Chain Search", extra=f"skipped: {reason}")

        if self.genetic and self.user_max_depth >= 2 and not _kb.poll() and not no_gpu_left:
            log_section("STAGE 3 — Genetic Algorithm Rule Evolution")
            rule_pool = HashcatRuleValidator.validate_rules_for_gpu(rules_phase1)
            hot_rules = [r for r,_ in sorted(single.items(), key=lambda kv:-kv[1])]
            t_now = time.time()
            remaining = max(0.0, self.params['TARGET_SECONDS'] - (t_now - t0))
            if self.genetic and self.user_max_depth >= 2:
                _min_ga = 120.0
                _ga_frac = 0.20
                reserved_ga = max(_min_ga, self.params['TARGET_SECONDS'] * _ga_frac)
            else:
                reserved_ga = 0.0
            ga_budget = remaining + reserved_ga
            if ga_budget < 5.0:
                log_warn(f"[S3]    Only {ga_budget:.1f}s available — consider raising --target-hours")
            else:
                log_info(f"[S3]    Budget: {bold(f'{ga_budget:.0f}s')}")
            known_set = set(all_counts.keys())
            # Stage 3 now fans out across every active GPU the same way Stage 1/2/Seed
            # do (see GeneticRuleEvolver.evaluate_population) — pass the MultiGPUEngine
            # itself instead of unwrapping to engines[0]. Make sure every engine has
            # its bloom filter / rules / kernel ready first (Stage 2 only does this
            # for engines[1:] when it itself splits across GPUs; if e.g. Stage 2 was
            # skipped or ran single-GPU for some reason, secondary engines could
            # otherwise be unprepared here).
            if isinstance(self.gpu_engine, MultiGPUEngine) and len(self.gpu_engine.engines) > 1:
                for eng in self.gpu_engine.engines:
                    self.gpu_engine._ensure_engine_ready(eng, bloom_filter, rules_phase1)
                self.gpu_engine._prune_disabled()
            evolver = GeneticRuleEvolver(self.gpu_engine, base_words, rule_pool, self.user_max_depth,
                                         pop_size=self.genetic_pop, elite_frac=self.genetic_elite,
                                         seed_hits=seed_hits, known_rules=known_set)
            with stage_profiler.track("STAGE 3 — Genetic Algorithm", unit="gen") as _s3t:
                ga_hits = evolver.evolve(hot_rules, self.genetic_generations, ga_budget)
                _s3t.set_count(self.genetic_generations)
                self._note_gpu_phase("STAGE 3 — Genetic Algorithm", source=evolver)
            before = len(all_counts)
            all_counts.update(ga_hits)
            new_from_ga = len(all_counts)-before
            truly_novel = sum(1 for r in ga_hits if r not in known_set)
            log_info(f"[S3]    {bold(cyan(str(new_from_ga)))} net new rules from STAGE 3  ({bold(green(str(len(ga_hits))))} total GA hits, {bold(cyan(str(truly_novel)))} genuinely novel)")

        validated = Counter({r:c for r,c in all_counts.items() if HashcatRuleValidator.validate_rule_for_gpu(r) and r.isascii()})
        return validated
