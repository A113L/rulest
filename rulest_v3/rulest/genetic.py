"""
Stage 3 (--genetic): a genetic algorithm that evolves rule *chains* against
the GPU-scored fitness of how many target words they explain, using
diversity-aware selection (Jaccard distance over covered-word signatures)
to avoid collapsing onto near-duplicate high-scoring chains.
"""
import time
import shutil
import random
import threading
import queue
import concurrent.futures
from collections import Counter

import numpy as np
from tqdm import tqdm

from .common import (
    log_info, log_warn, bold, green, cyan, yellow, _fmt_speed, _kb,
    HashcatRuleValidator, BUILTIN_PROBES,
)
from .minimize import compute_rule_signature_hash
from .multi_gpu import MultiGPUEngine

class GeneticRuleEvolver:
    def __init__(self, gpu_engine, base_words, rule_pool, max_depth,
                 pop_size=200, elite_frac=0.15, tournament_k=4, crossover_p=0.80,
                 mut_replace_p=0.60, mut_insert_p=0.20, mut_delete_p=0.20,
                 seed_hits=None, known_rules=None,
                 parsimony_alpha=0.05, sharing_sigma=0.5):
        self.gpu_engine = gpu_engine
        self.base_words = base_words
        self.rule_pool = rule_pool
        self.max_depth = max(2, max_depth)
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.tournament_k = tournament_k
        self.crossover_p = crossover_p
        total = mut_replace_p+mut_insert_p+mut_delete_p
        if total<=0: total=1.0
        self._mut_weights = [mut_replace_p/total, mut_insert_p/total, mut_delete_p/total]
        self.seed_hits = seed_hits or Counter()
        self.seed_chains_sorted = [r for r,_ in sorted(self.seed_hits.items(), key=lambda kv:-kv[1])] if self.seed_hits else []
        self.known_rules = known_rules or set()
        self._sig_cache = {}
        self._sig_cache_max = 50000
        self._sig_to_best = {}
        # -- Feature: CPU/GPU pipelining (see _start_prefetch) --
        self._prefetch_q = None
        self._prefetch_stop = None
        self._prefetch_thread = None
        # -- Feature: word subsampling (see _pick_generation_words) --
        self.active_words = base_words
        self._sample_offset = 0
        self._ga_cached_engines = {}
        # -- Feature: parsimony pressure (Occam's razor) --
        # Divides fitness by (1 + parsimony_alpha*(depth-2)) so that among two
        # chains with equal marginal coverage, the shorter one wins ties. Keeps
        # depth-2 chains (the cheapest possible) unpenalized, scales up slowly.
        self.parsimony_alpha = parsimony_alpha
        # -- Feature: fitness sharing (structural diversity pressure) --
        # sigma is the Jaccard-distance niche radius: individuals whose token
        # sets are closer than this to each other compete for shared fitness,
        # discouraging the population from converging on near-identical chains.
        self.sharing_sigma = sharing_sigma
        # -- Feature: cross-generation hit cache --
        # Elites (and any chain that happens to reappear verbatim) are re-sent
        # to the GPU every generation even though their hit set against the
        # *same* active_words hasn't changed. Cache by (chain_str) -> (active_words
        # identity, hit-set) and skip GPU evaluation on a hit; invalidated
        # automatically whenever active_words changes (subsample rotation or
        # full-sweep boundary), since the id() won't match.
        self._hit_cache = {}
        self._hit_cache_max = 20000

    def _get_sig(self, chain_str):
        if chain_str not in self._sig_cache:
            if len(self._sig_cache) > self._sig_cache_max:
                for k in list(self._sig_cache.keys())[:self._sig_cache_max//10]:
                    del self._sig_cache[k]
            self._sig_cache[chain_str] = compute_rule_signature_hash(chain_str, BUILTIN_PROBES)
        return self._sig_cache[chain_str]

    def _update_sig_registry(self, raw_map):
        new=0
        for cs, hits in raw_map.items():
            if hits<=0: continue
            sig = self._get_sig(cs)
            if sig.startswith('__UNSUPPORTED__'): continue
            if sig not in self._sig_to_best or hits>self._sig_to_best[sig][1]:
                self._sig_to_best[sig] = (cs, hits)
                new+=1
        for sig,(best,_) in self._sig_to_best.items():
            self.known_rules.add(best)
        return new

    def _sig_is_covered(self, cs):
        if not self._sig_to_best: return False
        sig = self._get_sig(cs)
        return sig in self._sig_to_best

    def _mutate_adaptive(self, tokens):
        tokens = self._mutate(tokens)
        for _ in range(2):
            if not self._sig_is_covered(' '.join(tokens)):
                break
            tokens = self._mutate(tokens)
        return tokens

    def initial_population(self, hot_rules):
        hot = hot_rules[:50]
        pop_set = set()
        n_hot = int(self.pop_size*0.3)
        n_seeded = int(self.pop_size*0.3)
        tries=0
        while len(pop_set)<n_hot and tries<n_hot*20:
            tries+=1
            if len(hot)>=2:
                a,b = random.sample(hot,2)
            elif len(hot)==1:
                a,b = hot[0], random.choice(self.rule_pool)
            else: break
            pop_set.add((a,b))
        tries=0
        while len(pop_set)<n_hot+n_seeded and tries<(n_hot+n_seeded)*20:
            tries+=1
            depth = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.7 else random.randint(2,self.max_depth)
            if hot:
                tok = [random.choice(hot)] + [random.choice(self.rule_pool) for _ in range(depth-1)]
                random.shuffle(tok)
            else:
                tok = self._random_chain(depth)
            pop_set.add(tuple(tok))
        n_fill = self.pop_size - len(pop_set)
        fill_set = set()
        if self.seed_chains_sorted:
            novel = [s for s in self.seed_chains_sorted if s not in self.known_rules]
            known = [s for s in self.seed_chains_sorted if s in self.known_rules]
            pool = novel if len(novel)>=n_fill//2 else novel+known
            if pool:
                selected = random.sample(pool[:max(n_fill*3,100)], min(n_fill, len(pool)))
                for sc in selected:
                    tok = sc.split()
                    if 2<=len(tok)<=self.max_depth:
                        fill_set.add(tuple(tok))
        fill_tries=0
        while len(fill_set)<n_fill and fill_tries<n_fill*20:
            fill_tries+=1
            d = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.6 else random.randint(2,self.max_depth)
            fill_set.add(tuple(self._random_chain(d)))
        pop_set.update(fill_set)
        pop = [list(ind) for ind in pop_set]
        while len(pop)<self.pop_size:
            d = random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.5 else random.randint(2,self.max_depth)
            pop.append(self._random_chain(d))
        return pop[:self.pop_size]

    def _random_chain(self, depth=0):
        if depth<=0: depth = random.randint(2, self.max_depth)
        return [random.choice(self.rule_pool) for _ in range(depth)]

    def _random_depth_biased(self):
        return random.randint(3,self.max_depth) if self.max_depth>=3 and random.random()<0.5 else random.randint(2,self.max_depth)

    # -- Pipelining: overlap CPU random-chain generation with GPU eval -------
    # The GA's true dependency chain (eval -> fitness -> breed -> eval...) is
    # inherently serial per generation: next_pop can't be built before this
    # generation's fitness is known. But a meaningful slice of the CPU-side
    # breeding work is NOT dependent on this generation's fitness at all —
    # it's pure random exploration: the "fill remaining slots" loop, the
    # stagnation-refresh population, and the crossover-fallback branch (when
    # a bred child collides with an already-known signature) all just need
    # *some* fresh random chain, drawn from the same rule_pool/depth
    # distribution regardless of what generation N's results were. We
    # generate those in a background thread that runs continuously for the
    # whole evolve() call, so while the GPU is busy evaluating a population,
    # the CPU is already producing tomorrow's random candidates instead of
    # sitting idle waiting for the GPU and only then starting to generate
    # them. No staleness is introduced into the actual selection/fitness
    # logic — this only pipelines the parts of breeding that were always
    # independent of the current population.
    def _start_prefetch(self):
        self._prefetch_stop = threading.Event()
        self._prefetch_q = queue.Queue(maxsize=max(64, self.pop_size*4))
        def _worker():
            while not self._prefetch_stop.is_set():
                try:
                    ind = tuple(self._random_chain(self._random_depth_biased()))
                    self._prefetch_q.put(ind, timeout=0.5)
                except queue.Full:
                    continue
                except Exception:
                    continue
        self._prefetch_thread = threading.Thread(target=_worker, daemon=True, name='ga-prefetch')
        self._prefetch_thread.start()

    def _stop_prefetch(self):
        if self._prefetch_stop is not None:
            self._prefetch_stop.set()
        self._prefetch_thread = None

    def _get_prefetched(self, avoid_set, max_tries=8):
        """Pull a pre-generated random chain not already in avoid_set. Falls back
        to synchronous generation if the prefetch buffer is empty (e.g. right at
        the start of evolve() before the background thread has filled it, or if
        CPU breeding is briefly outpacing the prefetch thread) — correctness is
        never compromised, this is purely a speed optimization."""
        q = self._prefetch_q
        if q is not None:
            for _ in range(max_tries):
                try:
                    ind = q.get_nowait()
                except queue.Empty:
                    break
                if ind not in avoid_set:
                    return list(ind)
        return self._random_chain(self._random_depth_biased())

    def _clamp(self, tokens):
        if len(tokens)<2: tokens += self._random_chain(2-len(tokens))
        return tokens[:self.max_depth]

    def _tournament_select(self, fitness):
        k = min(self.tournament_k, len(fitness))
        contenders = random.sample(fitness, k)
        return list(max(contenders, key=lambda x:x[1])[0])

    def _sharing_denoms(self, pop):
        """Fitness sharing (Goldberg & Richardson) over token-set Jaccard distance.
        Individuals whose rule_pool tokens overlap heavily with many peers get
        their fitness divided by a larger 'niche count', so a population that
        has piled onto one structural neighborhood stops getting rewarded for
        near-duplicates of each other — complements the marginal-coverage
        novelty bonus, which only catches *functional* duplicates (same
        BUILTIN_PROBES signature), not structurally similar-but-distinct chains
        that haven't converged to the same signature yet.

        Vectorized over numpy instead of an O(n^2) pure-Python double loop:
        each individual's token set is encoded as a boolean row in an
        (n_pop x n_vocab) matrix, where n_vocab is the number of *distinct*
        tokens actually used in this generation's population (bounded by
        pop_size*max_depth, typically far smaller than the full rule_pool —
        so this stays cheap even when rule_pool itself is huge). Pairwise
        intersection counts for all pairs at once are then just a single
        matrix multiply (BLAS-backed, O(n^2*V) but done in C, not Python),
        and Jaccard distance/union follow from simple broadcasting. Same
        math as before — |A∩B| via matmul instead of set &, |A∪B| via
        |A|+|B|-|A∩B| instead of set | — just batched instead of looped
        pair-by-pair, which is what actually blew up at pop_size~1200
        (720k pairs/gen of pure-Python set ops)."""
        n = len(pop)
        sigma = self.sharing_sigma
        if sigma <= 0 or n <= 1:
            return [1.0] * n

        # Build a per-generation vocabulary (token -> column index) covering
        # only tokens that actually appear in this population, not the full
        # rule_pool. Keeps the matrix small regardless of rule_pool size.
        vocab = {}
        row_tokens = []
        for ind in pop:
            uniq = set(ind)
            row_tokens.append(uniq)
            for t in uniq:
                if t not in vocab:
                    vocab[t] = len(vocab)
        v = len(vocab)
        if v == 0:
            return [1.0] * n

        M = np.zeros((n, v), dtype=np.bool_)
        for i, uniq in enumerate(row_tokens):
            if uniq:
                M[i, [vocab[t] for t in uniq]] = True

        # float32, not int32/bool: numpy's matmul only gets the BLAS
        # (sgemm/dgemm) fast path for float dtypes. An int matmul silently
        # falls back to a slow generic C loop with no SIMD/threading, which
        # made this ~10x SLOWER than float32 despite doing the same op.
        Mf = M.astype(np.float32)
        sizes = Mf.sum(axis=1)                 # |set_i| for each individual
        inter = Mf @ Mf.T                      # |set_i ∩ set_j| for every pair, one BLAS gemm call
        union = sizes[:, None] + sizes[None, :] - inter
        # avoid div-by-zero for the (degenerate) empty-vs-empty case
        dist = 1.0 - np.divide(inter, union, out=np.zeros_like(union, dtype=np.float32),
                                where=union > 0)
        np.fill_diagonal(dist, np.inf)         # exclude self-pairs, same as j>i loop did

        sh = np.where(dist < sigma, 1.0 - dist / sigma, 0.0)
        denom = 1.0 + sh.sum(axis=1)
        return denom.tolist()

    def _crossover(self, p1, p2):
        if len(p1)<2 or len(p2)<2 or random.random()>self.crossover_p:
            return list(p1), list(p2)
        cut1 = random.randint(1,len(p1)-1)
        cut2 = random.randint(1,len(p2)-1)
        c1 = self._clamp(p1[:cut1] + p2[cut2:])
        c2 = self._clamp(p2[:cut2] + p1[cut1:])
        return c1, c2

    def _mutate(self, tokens):
        tokens = list(tokens)
        op = random.choices(['replace','insert','delete'], weights=self._mut_weights)[0]
        if op=='replace':
            idx = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)
        elif op=='insert' and len(tokens)<self.max_depth:
            idx = random.randint(0,len(tokens))
            tokens.insert(idx, random.choice(self.rule_pool))
        elif op=='delete' and len(tokens)>2:
            idx = random.randrange(len(tokens))
            tokens.pop(idx)
        else:
            idx = random.randrange(len(tokens))
            tokens[idx] = random.choice(self.rule_pool)
        return tokens

    def _eval_on_engine(self, eng, chains, hit_words):
        """Run one engine's slice of chains against the FULL base_words set (already
        cached on that engine via set_base_words) using the single high-level
        evaluate_chains_for_ga() RPC, and merge results (as global word indices)
        into the shared hit_words dict. Each chain only ever belongs to one
        engine's slice, so different threads never write the same hit_words key
        -- no locking needed.

        result's keys are expected to be exactly the chain strings in `chains`
        (hence also already present as keys in hit_words, pre-seeded by
        evaluate_population()). If the GPU/kernel ever reports a key that
        isn't one of ours -- e.g. a chain string corrupted by depth
        truncation -- treat it as a bad/unusable hit and warn instead of
        raising: a KeyError here would blow up an entire multi-hour GPU
        sweep over what's fundamentally bad data for a handful of chains.
        """
        result = eng.evaluate_chains_for_ga(chains)
        for rule_str, idx_set in result.items():
            bucket = hit_words.get(rule_str)
            if bucket is None:
                log_warn(
                    f"[S3] GPU returned hits for unrecognized chain string "
                    f"{rule_str!r} (not in the requested batch) — discarding "
                    f"{len(idx_set)} hit(s) instead of crashing."
                )
                continue
            bucket |= idx_set

    def _ensure_base_words_cached(self, eng):
        # Cache the flattened active_words once per engine per distinct word set
        # (same trick Stage 2 uses via _run_chains_against_words -> set_base_words).
        # active_words changes across generations when word subsampling (Feature 3)
        # is enabled — a fresh random subsample object each time a sampling block
        # starts, or self.base_words itself on full-sweep generations — so this
        # naturally re-flattens only when the word set actually changes, not
        # every single generation. Without this caching, _run_chain_kernel would
        # fall through to prepare_words_data()/_flatten() and re-encode the
        # entire word batch from scratch on every generation x chain-batch x
        # word-batch call, which is what made Stage 3 slow in the first place.
        if self._ga_cached_engines.get(id(eng)) is not self.active_words:
            eng.set_base_words(self.active_words)
            self._ga_cached_engines[id(eng)] = self.active_words

    def evaluate_population(self, population):
        """Returns {chain_str: set(local_word_index)} — the set of self.active_words
        indices each chain successfully hit (indices are local to whatever word set
        is active this generation — see evolve()'s word-subsampling logic — not
        necessarily the full base_words). Tracking word identity (not just a hit
        count) is what makes marginal-coverage-gain fitness possible in evolve().

        Fans out across every active GPU when self.gpu_engine is a MultiGPUEngine,
        the same way Stage 1/2/Seed do: split the (small) population's chain batch
        across engines and run each slice against all of self.active_words on its
        own GPU in parallel, then merge. Each chain only ever runs on one engine, so
        merging hit_words is just a dict union — no cross-engine set merging needed.

        Cross-generation cache: elites carried over unchanged from the previous
        generation (and any other chain that happens to repeat verbatim) are
        served from self._hit_cache instead of re-run on GPU, as long as
        active_words hasn't changed since the cached result was recorded (see
        __init__ for the invalidation rule). Only genuinely new/unseen chains,
        or ones whose cache entry belongs to a stale active_words set, hit the
        GPU this generation."""
        cs = [' '.join(tok) for tok in population]
        valid = [c for c in cs if HashcatRuleValidator.validate_rule_for_gpu(c)]
        hit_words = {c: set() for c in cs}
        if not valid: return hit_words

        aw_id = id(self.active_words)
        to_eval = []
        for c in valid:
            cached = self._hit_cache.get(c)
            if cached is not None and cached[0] == aw_id:
                hit_words[c] = cached[1]
            else:
                to_eval.append(c)
        if not to_eval:
            return hit_words

        engines = self.gpu_engine.engines if isinstance(self.gpu_engine, MultiGPUEngine) else [self.gpu_engine]
        engines = [e for e in engines if not getattr(e, 'disabled', False)]
        if not engines: return hit_words

        for eng in engines:
            self._ensure_base_words_cached(eng)

        def _remember(chains):
            for c in chains:
                self._hit_cache[c] = (aw_id, hit_words[c])
            if len(self._hit_cache) > self._hit_cache_max:
                for k in list(self._hit_cache.keys())[:len(self._hit_cache)-self._hit_cache_max]:
                    del self._hit_cache[k]

        if len(engines) == 1:
            self._eval_on_engine(engines[0], to_eval, hit_words)
            _remember(to_eval)
            return hit_words

        # Split the chain batch across GPUs, proportionally to each engine's
        # measured throughput for 'ga' work if MultiGPUEngine has calibrated it
        # (either from earlier GA generations, or bootstrapped from Stage 1/2/
        # Seed's 'words'/'chains' measurements the first time — see fallback
        # logic below). Population sizes are typically small (hundreds), so
        # slices can be tiny with many GPUs — that's fine, each GPU still
        # tests its slice against all of active_words, which is the expensive
        # axis. Every chain lands in exactly one slice, so no locking is
        # needed: each thread only ever writes to the hit_words[rule_str]
        # entries for its own chains.
        if isinstance(self.gpu_engine, MultiGPUEngine):
            slices = self.gpu_engine.weighted_split(to_eval, kind='ga', engines=engines)
        else:
            n = len(engines)
            chunk = max(1, (len(to_eval) + n - 1) // n)
            slices = [to_eval[i*chunk:(i+1)*chunk] for i in range(n)]

        def _timed_eval(eng, sl):
            t0 = time.time()
            self._eval_on_engine(eng, sl, hit_words)
            if isinstance(self.gpu_engine, MultiGPUEngine) and sl:
                self.gpu_engine.record_speed(eng, 'ga', len(sl), time.time()-t0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as pool:
            futs = [pool.submit(_timed_eval, eng, sl)
                    for eng, sl in zip(engines, slices) if sl]
            for f in futs:
                try:
                    f.result()
                except Exception as exc:
                    log_warn(f"[S3]    GPU worker error: {exc}")
        _remember(to_eval)
        return hit_words

    def evolve(self, hot_rules, generations, time_budget,
               word_sample_frac=0.25, full_sweep_every=4):
        """word_sample_frac / full_sweep_every implement Feature 3 (stratified
        word subsampling): most generations only evaluate the population
        against a systematic subsample of base_words (every Nth word, with a
        fresh random phase offset each time the block resamples) instead of
        the entire wordlist, which is normally the dominant per-generation
        cost. Every `full_sweep_every` generations (plus generation 0 and the
        very last generation) we do a full sweep instead, both to keep the
        fitness landscape honest (avoid drifting toward chains that only look
        good on a lucky subsample) and to record accurate hit counts. Hit
        counts recorded from a sampled generation are rescaled by
        len(base_words)/len(sample) so they stay roughly comparable in
        magnitude to full-sweep counts in all_new/the signature registry;
        final counts are corrected with one full-sweep verification pass
        below once evolution finishes."""
        if not self.rule_pool: log_warn("[S3] empty rule pool"); return Counter()
        if time_budget<=0: log_warn("[S3] no time budget"); return Counter()
        start = time.time()
        all_new = Counter()
        n_elite = max(1, int(self.pop_size*self.elite_frac))
        stagnation = 0
        best_ever = 0
        log_info(f"[S3]    pop={self.pop_size}  max_gen={generations}  elite={self.elite_frac:.0%}  budget={time_budget:.0f}s  pool={len(self.rule_pool):,}  known={len(self.known_rules):,}")
        do_subsample = 0.0 < word_sample_frac < 1.0 and len(self.base_words) >= 2000
        if do_subsample:
            log_info(f"[S3]    Word subsampling: ~{word_sample_frac:.0%}/gen, full sweep every {full_sweep_every} gens")
        pop = self.initial_population(hot_rules)
        last_gen=0
        ncols = shutil.get_terminal_size((80,24)).columns
        t0 = time.time()
        total_combos = 0
        self._start_prefetch()
        try:
            with tqdm(total=generations, desc=green("  STAGE 3 "), unit="gen", ncols=ncols,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
                for gen in range(generations):
                    last_gen=gen
                    if _kb.poll(): break
                    if time.time()-start >= time_budget: break

                    is_full = (not do_subsample) or gen==0 or gen==generations-1 or (gen % full_sweep_every == 0)
                    if is_full:
                        self.active_words = self.base_words
                        scale = 1.0
                    else:
                        step = max(2, int(round(1.0/word_sample_frac)))
                        # Only re-roll the subsample phase offset once per
                        # full_sweep_every block (at the block's first non-full
                        # generation), not every single generation. Randomizing
                        # it every generation created a *new* active_words list
                        # object each time (new id()), which silently defeated
                        # _hit_cache (keyed by (id(active_words), chain)) even
                        # for elites carried over unchanged from the previous
                        # generation, and in multi-GPU mode forced a fresh
                        # pickled re-upload of the subsampled wordlist to every
                        # worker process every generation instead of once per
                        # block.
                        if gen % full_sweep_every == 1 or self._sample_offset is None:
                            self._sample_offset = random.randint(0, step-1)
                        self.active_words = self.base_words[self._sample_offset::step]
                        scale = len(self.base_words) / max(1, len(self.active_words))

                    hit_words = self.evaluate_population(pop)
                    raw = {c: len(s) for c, s in hit_words.items()}
                    raw_scaled = raw if scale==1.0 else {c: int(round(h*scale)) for c,h in raw.items()}
                    self._update_sig_registry(raw_scaled)
                    for cs, hits in raw_scaled.items():
                        if hits>0 and HashcatRuleValidator.validate_rule_for_gpu(cs):
                            if hits > all_new[cs]: all_new[cs]=hits

                    # --- Marginal coverage gain (submodular greedy) ---
                    # Process individuals in descending order of raw hit count and credit each
                    # one only for the base_words it covers that no higher-priority individual
                    # in this generation already covered. This is what "efficiency" means in
                    # the concentrator/Pareto sense: a chain that duplicates coverage already
                    # provided by a better individual contributes ~0 fitness, regardless of its
                    # own raw hit count, so the population stops rewarding near-duplicates.
                    # (Uses raw unscaled hit_words — the sets are only ever compared against
                    # each other within this same generation/same word sample, so scaling
                    # doesn't matter here, only for the cross-generation bookkeeping above.)
                    order = sorted(pop, key=lambda ind: -len(hit_words.get(' '.join(ind), ())))
                    covered = set()
                    marginal = {}
                    for ind in order:
                        cs_i = ' '.join(ind)
                        s = hit_words.get(cs_i, set())
                        gain = len(s - covered)
                        marginal[cs_i] = gain
                        covered |= s

                    # Novelty bonus now checks functional-signature membership (_sig_is_covered)
                    # instead of an exact string match against known_rules. A syntactically
                    # different chain that is functionally identical (on BUILTIN_PROBES) to an
                    # already-known rule no longer gets rewarded as if it were novel.
                    #
                    # Parsimony pressure: divide by (1 + alpha*(depth-2)) so that among
                    # two chains covering the same words, the shorter (cheaper, more
                    # general) one wins ties instead of the GA drifting toward
                    # needlessly long chains that cover nothing extra.
                    def _base_fit(ind):
                        cs_i = ' '.join(ind)
                        gain = marginal.get(cs_i, 0)
                        novelty = 1 if self._sig_is_covered(cs_i) else 2
                        depth_penalty = 1.0 + self.parsimony_alpha * max(0, len(ind)-2)
                        return (gain * novelty) / depth_penalty

                    # _base_fit(ind) is computed once per individual here and
                    # reused below for shared_fitness — it used to be computed
                    # a second time per individual in a separate generator
                    # expression, doubling the fitness-evaluation cost of
                    # every generation for no reason (the raw base fitness is
                    # identical both times; only the sharing denominator
                    # differs).
                    base_fits = [_base_fit(ind) for ind in pop]
                    fitness = sorted(
                        zip((tuple(ind) for ind in pop), base_fits),
                        key=lambda x: -x[1])
                    best = fitness[0][1] if fitness else 0

                    # Fitness sharing: elites (used below for next_pop and reporting)
                    # are still picked from the raw `fitness` above, so the single
                    # best-found chain each generation is never discarded for being
                    # "too similar" to something else. Sharing only reshapes the
                    # odds during tournament selection of *parents*, spreading
                    # reproductive opportunity away from crowded structural niches.
                    share_denoms = self._sharing_denoms(pop)
                    shared_fitness = sorted(
                        [(tuple(ind), fitness_val / share_denoms[i])
                         for i, (ind, fitness_val) in enumerate(
                             zip(pop, base_fits))],
                        key=lambda x: -x[1])
                    total_combos += len(pop) * len(self.active_words)
                    elapsed = time.time()-t0
                    spd = _fmt_speed(total_combos / elapsed if elapsed>0 else 0)
                    if best>best_ever: best_ever=best; stagnation=0
                    else: stagnation+=1
                    if stagnation>=5:
                        stagnation=0
                        n_ref = max(1, int(self.pop_size*0.3))
                        refresh = []
                        refresh_set = {fitness[0][0]} if fitness else set()
                        rt=0
                        while len(refresh)<n_ref and rt<n_ref*20:
                            rt+=1
                            ind = tuple(self._get_prefetched(refresh_set))
                            if ind not in refresh_set:
                                refresh.append(list(ind)); refresh_set.add(ind)
                        keep = [list(ind) for ind,_ in fitness[:self.pop_size-n_ref]]
                        pop = keep+refresh
                        pbar.update(1)
                        pbar.set_postfix({"best":cyan(str(best)), "new":cyan(str(len(all_new))), "sigs":cyan(str(len(self._sig_to_best))), "spd":green(spd), "stag":yellow("REFRESH")}, refresh=False)
                        continue
                    elites = [list(ind) for ind,_ in fitness[:n_elite]]
                    next_pop = elites[:]
                    next_set = {tuple(e) for e in elites}
                    breed_attempts=0
                    while len(next_pop)<self.pop_size and breed_attempts<(self.pop_size-len(next_pop))*8:
                        breed_attempts+=1
                        p1 = self._tournament_select(shared_fitness)
                        p2 = self._tournament_select(shared_fitness)
                        c1,c2 = self._crossover(p1,p2)
                        c1 = self._mutate_adaptive(c1)
                        c2 = self._mutate_adaptive(c2)
                        for c in (c1,c2):
                            if len(next_pop)>=self.pop_size: break
                            key = tuple(c)
                            if key in next_set: continue
                            if self._sig_is_covered(' '.join(c)):
                                c = self._get_prefetched(next_set)
                                key = tuple(c)
                                if key in next_set: continue
                            next_pop.append(c); next_set.add(key)
                    fill_attempts=0
                    while len(next_pop)<self.pop_size and fill_attempts<self.pop_size*4:
                        fill_attempts+=1
                        ind = tuple(self._get_prefetched(next_set))
                        if ind not in next_set:
                            next_pop.append(list(ind)); next_set.add(ind)
                    pop = next_pop[:self.pop_size]
                    pbar.update(1)
                    pbar.set_postfix({"best":cyan(str(best)), "new":cyan(str(len(all_new))), "sigs":cyan(str(len(self._sig_to_best))), "spd":green(spd)}, refresh=False)
        finally:
            self._stop_prefetch()

        # Final correction pass: re-verify every distinct rule chain we ever
        # credited against the FULL base_words in one batch, so counts that
        # came from a subsampled generation (only ever an estimate, scaled by
        # 1/frac) get replaced with exact figures before we return. This is a
        # single bounded sweep over the deduped set of winners, not per
        # generation, so it stays cheap even though it's a full-wordlist pass.
        if do_subsample and all_new and not _kb.quit_requested:
            log_info(f"[S3]    Verifying {len(all_new):,} candidate chain(s) against the full wordlist...")
            self.active_words = self.base_words
            verify_chains = list(all_new.keys())
            hit_words = {}
            engines = self.gpu_engine.engines if isinstance(self.gpu_engine, MultiGPUEngine) else [self.gpu_engine]
            engines = [e for e in engines if not getattr(e, 'disabled', False)]
            if engines:
                for eng in engines:
                    self._ensure_base_words_cached(eng)
                hit_words = {c: set() for c in verify_chains}
                if len(engines) == 1:
                    self._eval_on_engine(engines[0], verify_chains, hit_words)
                else:
                    n = len(engines)
                    chunk = max(1, (len(verify_chains) + n - 1) // n)
                    slices = [verify_chains[i*chunk:(i+1)*chunk] for i in range(n)]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                        futs = [pool.submit(self._eval_on_engine, eng, sl, hit_words)
                                for eng, sl in zip(engines, slices) if sl]
                        for f in futs:
                            try: f.result()
                            except Exception as exc: log_warn(f"[S3]    Verify worker error: {exc}")
                all_new = Counter({c: len(s) for c, s in hit_words.items() if s})

        elapsed = time.time()-start
        log_info(f"[S3]    Evolution complete — {bold(green(str(len(all_new))))} unique chains passed bloom filter  ({bold(cyan(str(len(self._sig_to_best))))} distinct functional signatures)  ({elapsed:.1f}s, {last_gen+1} generation(s))")
        return all_new
