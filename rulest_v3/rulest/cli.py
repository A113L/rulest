"""
Command-line entry point: argument parsing and the Stage 0/1/S/2/3 pipeline
summary/report printing that wraps GPUExtractor.extract_rules().
"""
import os
import sys
import gc
import time
import shutil
import argparse
import datetime
import multiprocessing as mp
from collections import defaultdict

import numpy as np

from . import state
from .common import (
    print_banner, _print_controls, log_info, log_warn, log_error,
    bold, cyan, green, yellow, red, dim, _kb, safe_worker_count,
    MAX_HASHCAT_CHAIN, MINIMIZE_DISK_THRESHOLD, BUILTIN_PROBES,
)
from .devices import list_devices
from .extractor import GPUExtractor
from .multi_gpu import MultiGPUEngine
from .minimize import minimize_by_signature
from .selection import select_rules_by_marginal_coverage
from .io_utils import load_wordlist
from .profiling import stage_profiler

def main():
    ap = argparse.ArgumentParser(prog='rulest', description='GPU-Compatible Hashcat Rules Engine (optimized)',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every option below takes a single-letter short form alongside its long
    # name (e.g. --select-budget / -b), normalized the same way -o/--output
    # already worked, so any flag can be used either way. A handful of rare/
    # dangerous switches (--bloom-no-shard, --allow-reject-rules) and the
    # numbered --depthN-chains family intentionally have no short form —
    # one letter per depth (2..10) would eat most of the alphabet for
    # options nobody runs from memory anyway.
    ap.add_argument('base_wordlist', nargs='?')
    ap.add_argument('target_wordlist', nargs='?')
    ap.add_argument('-o','--output', default='rulest_output.txt')
    ap.add_argument('-d', '--device', default=None, metavar='SPEC',
                    help='Device index, name substring, "all" to run every detected GPU in parallel, '
                         'or a comma-separated list of indices (e.g. "0,1") to run only those GPUs in parallel '
                         '(default: auto-select best single GPU)')
    ap.add_argument('-L', '--list-devices', action='store_true')
    ap.add_argument('-m', '--max-depth', type=int, default=2)
    ap.add_argument('-t', '--target-hours', type=float, default=0.5)
    ap.add_argument('-c', '--max-chains', type=int, default=0)
    ap.add_argument('-B', '--bloom-mb', type=int, default=0)
    ap.add_argument('--bloom-no-shard', action='store_true')
    ap.add_argument('-r', '--seed-rules', default=None)
    for i in range(2,11):
        ap.add_argument(f'--depth{i}-chains', type=int, default=None, dest=f'depth{i}_chains')
    ap.add_argument('--allow-reject-rules', action='store_true')
    ap.add_argument('-n', '--no-builtin-seeds', action='store_true')
    pt = ap.add_argument_group('STAGE 0 — Token-Strip')
    pt.add_argument('-T', '--token-strip', action='store_true')
    pt.add_argument('--token-strip-min-stem', type=int, default=4)
    pt.add_argument('--token-strip-max-prefix', type=int, default=4)
    pt.add_argument('--token-strip-max-suffix', type=int, default=4)
    pt.add_argument('--token-strip-min-leet-amb', type=int, default=3)
    pt.add_argument('--token-strip-workers', type=int, default=0)
    pt.add_argument('--token-strip-chunk-size', type=int, default=0)
    ga = ap.add_argument_group('Genetic Algorithm')
    ga.add_argument('-g', '--genetic', action='store_true')
    ga.add_argument('-G', '--genetic-generations', type=int, default=50)
    ga.add_argument('-P', '--genetic-pop', type=int, default=200)
    ga.add_argument('-E', '--genetic-elite', type=float, default=0.15)
    sel = ap.add_argument_group('Rule Selection (post-processing)')
    sel.add_argument('-M', '--select-mode', choices=['frequency', 'greedy'], default='frequency',
                      help="frequency (default): sort by raw GPU hit count, as before. "
                           "greedy: lazy-greedy (CELF) marginal-coverage selection against "
                           "the real TARGET wordlist — maximizes |union of recovered words| "
                           "per rule budget instead of summed per-rule score.")
    sel.add_argument('-b', '--select-budget', type=int, default=0,
                      help="Max rules to keep with --select-mode greedy (0 = run to "
                           "saturation, i.e. until no remaining rule adds new coverage).")
    sel.add_argument('-s', '--select-budgets', default=None,
                      help="Comma-separated list of budgets (e.g. '100,1000,10000,150000') to "
                           "additionally emit as separate prefix files '<output>.<budget>.txt' "
                           "cut from the same greedy ordering — lets you compare recovery vs. "
                           "budget without rerunning coverage eval for each size.")
    sel.add_argument('-a', '--select-cost-alpha', type=float, default=0.0,
                      help="gain(r) = new_recovery(r) / depth(r)**alpha. 0 (default) = pure "
                           "marginal coverage. >0 favors shorter/cheaper chains among rules "
                           "with similar marginal gain.")
    sel.add_argument('-e', '--exact-recovery', action='store_true',
                      help="Explicitly report exact distinct target-word recovery after greedy "
                           "selection. Greedy coverage is already exact: bloom-positive GPU "
                           "hits are resolved through the exact verification kernel before CELF "
                           "selection, so this flag adds reporting only (no second GPU pass).")
    ap.add_argument('-D', '--debug', action='store_true')
    ap.add_argument('-p', '--profile', action='store_true',
                    help='Enable profiling: prints a per-stage wall-clock breakdown table '
                         '(STAGE 0-3, bloom build, minimization, selection/CELF) at the end '
                         'of the run, with GPU-level CELF detail folded into the Selection '
                         'row. Equivalent to setting RULEST_PROFILE=1 (and implies the finer '
                         'RULEST_CELF_PROFILE=1 GPU-event accounting); use for tuning, not '
                         'for normal runs, since the CELF side adds a little per-call overhead.')
    args = ap.parse_args()

    state.ALLOW_REJECT_RULES = args.allow_reject_rules
    state.VERBOSE = args.debug
    # Single switch for every profiling signal in the pipeline: the always-cheap
    # per-stage wall-clock table (STAGE 0-3, bloom build, minimize, selection) and
    # the heavier per-call GPU-event CELF accounting inside gpu_engine.py, which is
    # itself gated on this same env var so it stays off unless asked for. Either the
    # flag (-p/--profile) or the env var (for scripted/GUI launches that don't go
    # through argparse) turns both on together -- no more needing to know there
    # were two switches.
    profiling_on = args.profile or os.environ.get("RULEST_PROFILE") == "1" or \
                   os.environ.get("RULEST_CELF_PROFILE") == "1"
    if profiling_on:
        os.environ["RULEST_CELF_PROFILE"] = "1"
    stage_profiler.configure(profiling_on)
    print_banner()
    _print_controls()
    if args.list_devices: list_devices(); sys.exit(0)
    if not args.base_wordlist or not args.target_wordlist:
        ap.print_help(); print(); log_error("Both BASE and TARGET wordlists required"); sys.exit(1)
    if args.max_depth > MAX_HASHCAT_CHAIN:
        log_warn(f"Depth capped to {MAX_HASHCAT_CHAIN}"); args.max_depth=MAX_HASHCAT_CHAIN
    elif args.max_depth<1: log_error("--max-depth >=1"); sys.exit(1)

    log_info(f"  base      : {bold(args.base_wordlist)}")
    log_info(f"  target    : {bold(args.target_wordlist)}")
    shard_note = yellow('[sharding forced off]') if args.bloom_no_shard else dim('(auto: shard if >512MB)')
    log_info(f"  depth     : {bold(str(args.max_depth))}  |  hours: {bold(str(args.target_hours))}  |  bloom: {bold(str(args.bloom_mb or 'auto'))}MB  {dim('(GPU-only)')}  {shard_note}")
    if args.seed_rules: log_info(f"  seeds     : {bold(args.seed_rules)}")
    print()

    _EXCL = dim('[time budget: excluded]')
    _INCL = dim('[time budget: included]')

    if args.token_strip:
        _inj = green('→ STAGE S sbd → STAGE 2') if not args.no_builtin_seeds else yellow('→ STAGE 1 only → STAGE 2')
        log_info(f"  {bold(cyan('STAGE 0'))}  : {green('enabled')} — CPU exact-match (core+insert)  min-stem={args.token_strip_min_stem}  prefix={args.token_strip_max_prefix}  suffix={args.token_strip_max_suffix}  leet-amb={args.token_strip_min_leet_amb}  workers={safe_worker_count(args.token_strip_workers or mp.cpu_count())}  {_inj}  {_EXCL}")
    else:
        log_info(f"  {bold(dim('STAGE 0'))}  : {red('disabled')}")

    log_info(f"  {bold(cyan('STAGE 1'))}  : {green('enabled')} — single-rule GPU sweep  {_INCL}")

    if not args.no_builtin_seeds:
        log_info(f"  {bold(cyan('STAGE S'))}  : {green('enabled')} — numeric/pattern seed families A-M  {_EXCL}")
    else:
        log_info(f"  {bold(dim('STAGE S'))}  : {red('disabled')}  (--no-builtin-seeds)")

    log_info(f"  {bold(cyan('STAGE 2'))}  : {green('enabled')} — rule-chain GPU search  depth 2-{args.max_depth}  {_INCL}")

    if args.genetic:
        if not 0.0<args.genetic_elite<1.0: log_error("--genetic-elite must be between 0 and 1"); sys.exit(1)
        log_info(f"  {bold(cyan('STAGE 3'))}  : {green('enabled')} — genetic algorithm  pop={args.genetic_pop}  gen={args.genetic_generations}  elite={args.genetic_elite:.0%}  {_INCL}")
    else:
        log_info(f"  {bold(dim('STAGE 3'))}  : {red('disabled')}  (--genetic to enable)")

    if args.select_mode == 'greedy':
        budget_note = 'unbounded (saturation)' if args.select_budget <= 0 else f'{args.select_budget:,}'
        log_info(f"  {bold(cyan('SELECT'))}  : {green('greedy')} — CELF marginal-coverage over real TARGET  "
                 f"budget={budget_note}  cost_alpha={args.select_cost_alpha}")
        if args.select_budgets:
            log_info(f"            budget sweep: {args.select_budgets}")
    else:
        log_info(f"  {bold(dim('SELECT'))}  : {dim('frequency')} — sort by raw GPU hit count (--select-mode greedy to enable)")
    print()

    base_words = load_wordlist(args.base_wordlist)
    target_words = load_wordlist(args.target_wordlist)
    print()

    _kb.start()
    t_start = time.time()
    stage_profiler.begin_run()
    extractor = GPUExtractor(
        len(base_words), len(target_words), args.max_depth, args.device,
        args.target_hours, args.max_chains, args.seed_rules, args.bloom_mb,
        builtin_seeds=not args.no_builtin_seeds,
        bloom_no_shard=args.bloom_no_shard, genetic=args.genetic,
        genetic_generations=args.genetic_generations, genetic_pop=args.genetic_pop,
        genetic_elite=args.genetic_elite, token_strip=args.token_strip,
        token_strip_min_stem=args.token_strip_min_stem,
        token_strip_max_prefix=args.token_strip_max_prefix,
        token_strip_max_suffix=args.token_strip_max_suffix,
        token_strip_min_leet_amb=args.token_strip_min_leet_amb,
        token_strip_workers=args.token_strip_workers,
        token_strip_chunk_size=args.token_strip_chunk_size,
    )
    depth_overrides = {f'depth{i}_override': getattr(args, f'depth{i}_chains') for i in range(2,11)}
    # Greedy CELF selection (below) still needs one live GPU from this same
    # pool: it runs after extract_rules() returns, on extractor.gpu_engine
    # directly (CELF is single-GPU only -- see _resolve_single_engine in
    # selection.py -- but it reuses whichever device survived extraction
    # rather than opening a second one). So a MultiGPUEngine must NOT be
    # shut down here when selection is still going to use it: doing so
    # used to terminate every worker process right after Stage 0-3 and
    # leave Selection holding a dead engine, surfacing later as either a
    # hang or an unrelated-looking AttributeError deep in _GpuExactScorer.
    # Non-greedy ('frequency') mode never touches the GPU again, so it's
    # still safe -- and good practice -- to free the pool immediately.
    try:
        raw_counts = extractor.extract_rules(base_words, target_words, **depth_overrides)
    finally:
        if isinstance(extractor.gpu_engine, MultiGPUEngine) and args.select_mode != 'greedy':
            extractor.gpu_engine.shutdown()
    # Keep keyboard listener alive through minimize + coverage + CELF +
    # local-search so [p]/[r]/[q] work on every stage, not only STAGE 0–3.

    if args.select_mode != 'greedy':
        # Only the greedy selector needs the real target wordlist post-hoc;
        # frequency mode never touches it again, so free it as before.
        del target_words
        gc.collect()
    if _kb.quit_requested: log_warn("[QUIT]  Early exit — saving partial results")
    log_info(f"\n[GPU]  Raw bloom-filter candidates: {bold(cyan(str(len(raw_counts))))}"); print()

    with stage_profiler.track("Minimization (dedup)", unit="rule") as _mint:
        final_counts = minimize_by_signature(raw_counts, BUILTIN_PROBES)
        _mint.set_count(len(raw_counts))
    if ':' not in final_counts: final_counts[':'] = 0

    depth_dist = defaultdict(int)
    for r in final_counts:
        if r != ':': depth_dist[len(r.split())] += 1
    ds = '  '.join(f"d{d}:{depth_dist[d]:,}" for d in sorted(depth_dist))
    final_rules = len(final_counts) - (1 if ':' in final_counts else 0)
    removed = len(raw_counts) - len(final_counts)

    si = sorted(final_counts.items(), key=lambda kv: (-kv[1], len(kv[0].split()), kv[0]))
    def _write_ruleset_file(path, ordered_rules, sort_desc, extra_header=None):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# rulest — GPU-Compatible Hashcat Rules Engine (optimized)\n")
            f.write(f"# Generated      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Base           : {os.path.basename(args.base_wordlist)}\n")
            f.write(f"# Target         : {os.path.basename(args.target_wordlist)}\n")
            f.write(f"# Depth          : 1-{args.max_depth}\n")
            bloom_mb = args.bloom_mb if args.bloom_mb>0 else "auto"
            f.write(f"# Bloom          : {bloom_mb} MB  (sharding: {'disabled' if args.bloom_no_shard else 'auto'})\n")
            f.write(f"# STAGE 0        : core + insert mode  GPU-verified candidates\n")
            if args.genetic:
                f.write(f"# STAGE 3 GA     : pop={args.genetic_pop}  gen={args.genetic_generations}  elite={args.genetic_elite:.0%}\n")
            f.write("#\n")
            f.write(f"# GPU raw candidates      : {len(raw_counts):,}\n")
            mp2 = 'disk-backed SQLite' if len(raw_counts) > MINIMIZE_DISK_THRESHOLD else 'in-memory'
            f.write(f"# Minimization            : {mp2}  (threshold {MINIMIZE_DISK_THRESHOLD:,})\n")
            f.write(f"#   Probe words           : {len(BUILTIN_PROBES)}  (built-in)\n")
            f.write(f"#   Equiv. rules removed  : {removed:,}\n")
            f.write("#\n")
            f.write(f"# Rules kept     : {len(ordered_rules):,}\n")
            f.write(f"# Sorted by      : {sort_desc}\n")
            if extra_header:
                for line in extra_header:
                    f.write(f"# {line}\n")
            f.write(":\n")
            for r in ordered_rules:
                if r != ':': f.write(f"{r}\n")

    if args.select_mode == 'greedy':
        if isinstance(extractor.gpu_engine, MultiGPUEngine) and not extractor.gpu_engine.engines:
            log_error("[GPU]  No GPU device is available for selection -- every GPU "
                      "became unresponsive during extraction. Re-run with a single, "
                      "healthy --device.")
            sys.exit(1)
        candidate_rules = [r for r in final_counts if r != ':']
        n_target = len(set(target_words))
        budget = args.select_budget if args.select_budget > 0 else None

        try:
            with stage_profiler.track("Selection (CELF greedy)", unit="candidate") as _selt:
                selected, covered_mask = select_rules_by_marginal_coverage(
                    extractor.gpu_engine,
                    base_words,
                    target_words,
                    candidate_rules,
                    args.max_depth,
                    budget=budget,
                    cost_alpha=args.select_cost_alpha,
                )
                _selt.set_count(len(candidate_rules))
        finally:
            if isinstance(extractor.gpu_engine, MultiGPUEngine):
                extractor.gpu_engine.shutdown()

        ordered_rules = [rule for rule, _gain in selected]
        top_selected = list(selected)
        n_covered = int(covered_mask.sum())
        recovery_pct = n_covered / max(1, n_target)
        sort_desc = (
            f"greedy marginal coverage (CELF upper-bound + GPU recompute, "
            f"cost_alpha={args.select_cost_alpha})"
        )
        extra_header = [
            f"SELECT mode     : greedy (budget="
            f"{'unbounded/saturation' if budget is None else budget})",
            f"Target recovery : {n_covered:,}/{n_target:,} "
            f"({recovery_pct:.2%}) (exact distinct target coverage)",
        ]
        if args.exact_recovery:
            extra_header.append(
                f"Exact recovery  : {n_covered:,}/{n_target:,} "
                f"({recovery_pct:.2%}) (GPU exact verification; no coverage matrix)"
            )

        _write_ruleset_file(args.output, ordered_rules, sort_desc, extra_header)
        log_info(
            f"[OUT] Greedy ruleset written to: {bold(args.output)} "
            f"({bold(str(len(ordered_rules)))} rules, recovery {bold(f'{recovery_pct:.2%}')})"
        )

        if args.select_budgets:
            budgets = sorted({
                int(value) for value in args.select_budgets.split(',')
                if value.strip() and int(value) > 0
            })
            base, ext = os.path.splitext(args.output)
            ext = ext or '.txt'
            print()
            log_info(
                f"[OUT] Budget sweep — cutting the same greedy ordering at {len(budgets)} sizes:"
            )
            for size in budgets:
                sub = ordered_rules[:size]
                # CELF marginal gains are disjoint increments, so their sum
                # yields exact union recovery for each prefix.
                n_sub_covered = sum(gain for _rule, gain in selected[:size])
                sub_pct = n_sub_covered / max(1, n_target)
                sub_path = f"{base}.{size}{ext}"
                sub_extra = [
                    f"SELECT mode     : greedy (budget={size}, prefix of full greedy ordering)",
                    f"Target recovery : {n_sub_covered:,}/{n_target:,} ({sub_pct:.2%})",
                ]
                _write_ruleset_file(
                    sub_path, sub,
                    f"greedy marginal coverage (CELF GPU recompute), budget={size}",
                    sub_extra,
                )
                log_info(
                    f"       budget={size:>9,} rules={len(sub):>9,} "
                    f"recovery={sub_pct:.2%} -> {sub_path}"
                )

        del target_words
        gc.collect()
    else:
        ordered_rules = [r for r, _ in si]
        top_selected = None  # frequency mode: final summary uses GPU hit counts
        _write_ruleset_file(args.output, ordered_rules, "GPU frequency (descending, UTF-8)")
        log_info(f"[OUT]  Minimized rules written to: {bold(args.output)}")

        if args.select_budgets:
            # Cut the SAME frequency-descending ordering at the same sizes
            # used for the greedy budget sweep, so benchmark runs compare
            # like-for-like rule-COUNT (not "whatever frequency mode felt
            # like keeping"). No coverage/recovery % here — frequency mode
            # never builds the coverage matrix (that's greedy-only, and
            # expensive); if you need recovery numbers for frequency-mode
            # prefixes too, run them through evaluate_full_coverage
            # separately (or just diff actual hashcat/john crack counts
            # between the two rulesets, which is presumably the point of
            # the benchmark anyway).
            budgets = sorted({int(b) for b in args.select_budgets.split(',') if b.strip()})
            base, ext = os.path.splitext(args.output)
            ext = ext or '.txt'
            print()
            log_info(f"[OUT]  Budget sweep — cutting the same frequency ordering at {len(budgets)} sizes:")
            for b in budgets:
                sub = ordered_rules[:b]
                sub_path = f"{base}.{b}{ext}"
                sub_extra = [
                    f"SELECT mode     : frequency (budget={b}, prefix of full frequency ordering)",
                ]
                _write_ruleset_file(sub_path, sub, f"GPU frequency (descending, UTF-8), budget={b}", sub_extra)
                log_info(f"       budget={b:>9,}  rules={len(sub):>9,}  -> {sub_path}")

    elapsed = time.time()-t_start
    stage_profiler.report()
    sep = '─' * shutil.get_terminal_size((80,24)).columns
    print()
    log_info(cyan(sep))
    log_info(f"  {bold('DONE')}  rulest finished in {bold(f'{elapsed:.1f}s')}")
    log_info(cyan(sep))
    log_info(f"  GPU raw candidates : {bold(str(len(raw_counts)))}")
    log_info(f"  Rules kept         : {bold(green(str(final_rules)))}  {dim('('+ds+')')}")
    log_info(f"  Rules removed      : {bold(red(str(removed)))}")
    log_info(f"  Output file        : {bold(args.output)}")
    log_info(cyan(sep))
    if args.select_mode == 'greedy' and top_selected:
        # Show the greedy (or greedy+local-search) selection order — the
        # rules that actually made it into the output ruleset, ranked by
        # the order CELF picked them (plain greedy) or by each rule's final
        # since swaps happen in place and no longer leave position
        # meaningful — see local_search_improve()'s docstring). Gain is the
        # CELF marginal coverage at selection time for plain greedy, or
        # that same final exclusive-contribution count after local search.
        top = top_selected[:20]
        label = "greedy selection order"
        print(); log_info(f"  Top {len(top)} rules by {label}:")
        for i, (r, gain) in enumerate(top, 1):
            if isinstance(gain, (int, float)) and gain > 0:
                gain_s = f"{gain:.2f}" if isinstance(gain, float) else str(int(gain))
            else:
                gain_s = "—"
            freq = final_counts.get(r, 0)
            log_info(
                f"  {dim(str(i).rjust(3)+'.')}  {dim(f'd={len(r.split())}')}  "
                f"{r:<42s}  gain={cyan(gain_s)}  {dim(f'freq={freq}')}"
            )
    else:
        top = sorted(
            [(r, c) for r, c in final_counts.items() if r != ':'],
            key=lambda kv: (-kv[1], len(kv[0].split()), kv[0]),
        )[:20]
        if top:
            print(); log_info(f"  Top {len(top)} rules by GPU frequency:")
            for i, (r, c) in enumerate(top, 1):
                log_info(
                    f"  {dim(str(i).rjust(3)+'.')}  {dim(f'd={len(r.split())}')}  "
                    f"{r:<42s}  {cyan(str(c))}"
                )
    print()

    _kb.stop()
