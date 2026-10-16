"""
Post-processing: signature-based functional minimization of the raw rule
candidates GPU-extracted from earlier stages. Two rules that produce
identical outputs on a fixed probe-word set are functionally equivalent, so
only the best (highest-count, shortest, lexicographically-first) survivor
of each equivalence class is kept.

Falls back to a disk-backed SQLite table once the candidate count passes
MINIMIZE_DISK_THRESHOLD, since an in-memory dict of that many signature ->
[(rule,count),...] groups gets too large to hold comfortably.
"""
import os
import time
import shutil
import hashlib
import sqlite3
import tempfile
import multiprocessing as mp
from collections import Counter, defaultdict

from tqdm import tqdm

from .common import (
    log_section, log_info, log_warn, bold, cyan, green, red, dim, _fmt_speed,
    safe_worker_count, py_apply_chain,
    MINIMIZE_DISK_THRESHOLD, MINIMIZE_DISK_BATCH_SIZE, _kb,
)

# Module-level globals used by the multiprocessing worker so it is
# picklable (local/closure functions cannot be pickled on spawn/forkserver).
_minimize_worker_probe_words: list = []

def _minimize_worker_init(probe_words):
    """Pool initializer — stores probe words in each worker process."""
    global _minimize_worker_probe_words
    _minimize_worker_probe_words = probe_words

def _minimize_worker_sig(rule_count):
    """Top-level (picklable) worker: compute signature for one (rule, count) pair."""
    rule, count = rule_count
    sig = compute_rule_signature_hash(rule, _minimize_worker_probe_words)
    return sig, rule, count


def compute_rule_signature_hash(rule, probe_words):
    outputs = []
    for w in probe_words:
        out = py_apply_chain(rule, w)
        if out is None:
            # Return the sentinel string directly so callers can detect it with
            # sig.startswith('__UNSUPPORTED__').  Previously this returned a SHA1
            # digest of the sentinel, making the check unreachable.
            return f"__UNSUPPORTED__::{rule}"
        outputs.append(out)
    combined = '\x00'.join(outputs).encode('utf-8', errors='replace')
    # blake2b is a non-cryptographic-strength-need use case here (we only need
    # collision-avoidance for grouping, not security), and is noticeably
    # faster than sha1 on the short inputs this function deals with.
    return hashlib.blake2b(combined, digest_size=16).hexdigest()

def minimize_by_signature(rule_counter, probe_words):
    if not rule_counter: return Counter()
    log_section("POST-PROCESSING — Signature-Based Functional Minimization")
    n = len(rule_counter)
    log_info(f"[MINIMIZE] Candidates  : {bold(str(n))}")
    log_info(f"[MINIMIZE] Probe words : {bold(str(len(probe_words)))}")
    if n > MINIMIZE_DISK_THRESHOLD:
        return _minimize_disk(rule_counter, probe_words)
    else:
        return _minimize_mem(rule_counter, probe_words)

def _minimize_mem(rule_counter, probe_words):
    sig_map = defaultdict(list)
    rule_items = list(rule_counter.items())
    n_total = len(rule_items)
    ncols = shutil.get_terminal_size((80,24)).columns
    n_workers = safe_worker_count(min(mp.cpu_count(), n_total // 200 + 1))
    use_mp = n_total >= 500 and n_workers > 1
    log_info(f"[MINIMIZE] Workers    : {bold(str(n_workers if use_mp else 1))}")
    t0 = time.time()
    if use_mp:
        if hasattr(os, 'fork'):
            ctx = mp.get_context('fork')
        else:
            ctx = mp.get_context('spawn')
        with tqdm(total=n_total, desc=green("  Minimizing"), unit="rule", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            with ctx.Pool(processes=n_workers,
                          initializer=_minimize_worker_init,
                          initargs=(probe_words,)) as pool:
                for sig, rule, count in pool.imap_unordered(_minimize_worker_sig, rule_items, chunksize=64):
                    if _kb.poll():
                        log_warn("[MIN] Early exit — keeping minimization progress so far")
                        pool.terminate()
                        break
                    sig_map[sig].append((rule, count))
                    pbar.update(1)
                    elapsed = time.time() - t0
                    spd = _fmt_speed(pbar.n / elapsed if elapsed>0 else 0, "rules")
                    pbar.set_postfix({"unique_sigs": cyan(str(len(sig_map))), "spd": green(spd)}, refresh=False)
    else:
        with tqdm(total=n_total, desc=green("  Minimizing"), unit="rule", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            for rule, count in rule_items:
                if _kb.poll():
                    log_warn("[MIN] Early exit — keeping minimization progress so far")
                    break
                sig = compute_rule_signature_hash(rule, probe_words)
                sig_map[sig].append((rule, count))
                pbar.update(1)
                elapsed = time.time() - t0
                spd = _fmt_speed(pbar.n / elapsed if elapsed>0 else 0, "rules")
                pbar.set_postfix({"unique_sigs": cyan(str(len(sig_map))), "spd": green(spd)}, refresh=False)
    survivors = Counter()
    n_unsupported = 0
    for sig, group in sig_map.items():
        if sig.startswith('__UNSUPPORTED__'):
            n_unsupported += len(group) - 1  # count duplicates removed, not total kept
        best_rule, best_count = min(group, key=lambda x: (-x[1], len(x[0].split()), x[0]))
        survivors[best_rule] = best_count
    removed = n_total - len(survivors)
    log_info(f"[MINIMIZE] {green('Done')}")
    log_info(f"           Unique signatures : {bold(cyan(str(len(sig_map)))):>12s}")
    log_info(f"           Rules kept        : {bold(green(str(len(survivors)))):>12s}")
    log_info(f"           Rules removed     : {bold(red(str(removed))):>12s}  ({removed/max(1,n_total):.1%})")
    if n_unsupported:
        log_info(f"           Unsupported (kept 1 each group) : {bold(str(n_unsupported))}")
    return survivors

def _minimize_disk(rule_counter, probe_words):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.db', prefix='rulest_minimize_')
    os.close(tmp_fd)
    try:
        conn = sqlite3.connect(tmp_path)
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = OFF')
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('PRAGMA cache_size = -131072')
        conn.execute('CREATE TABLE sig_best (sig_hash TEXT PRIMARY KEY, rule TEXT NOT NULL, count INTEGER NOT NULL, depth INTEGER NOT NULL)')
        conn.commit()
        _UPSERT = '''
            INSERT INTO sig_best (sig_hash, rule, count, depth)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sig_hash) DO UPDATE SET
                rule  = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.rule
                    WHEN excluded.count = sig_best.count AND excluded.depth < sig_best.depth THEN excluded.rule
                    WHEN excluded.count = sig_best.count AND excluded.depth = sig_best.depth AND excluded.rule < sig_best.rule THEN excluded.rule
                    ELSE sig_best.rule
                END,
                count = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.count
                    ELSE sig_best.count
                END,
                depth = CASE
                    WHEN excluded.count > sig_best.count THEN excluded.depth
                    WHEN excluded.count = sig_best.count AND excluded.depth < sig_best.depth THEN excluded.depth
                    WHEN excluded.count = sig_best.count AND excluded.depth = sig_best.depth AND excluded.rule < sig_best.rule THEN excluded.depth
                    ELSE sig_best.depth
                END
        '''
        rule_items = list(rule_counter.items())
        n_total = len(rule_items)
        batch = []
        log_info(f"[MINIMIZE] Temp DB : {dim(tmp_path)}")
        ncols = shutil.get_terminal_size((80,24)).columns
        # Signature computation (pure CPU, one rule at a time) used to run in
        # the main thread here, unlike _minimize_mem's Pool-based version —
        # a serial bottleneck on exactly the workload (>MINIMIZE_DISK_THRESHOLD
        # candidates) where it hurts most. Reuse the same worker pattern: a
        # Pool computes signatures in parallel, the main thread only does the
        # (inherently serial) SQLite executemany batching.
        n_workers = safe_worker_count(min(mp.cpu_count(), n_total // 200 + 1))
        use_mp = n_total >= 500 and n_workers > 1
        log_info(f"[MINIMIZE] Workers    : {bold(str(n_workers if use_mp else 1))}")
        with tqdm(total=n_total, desc=green("  Minimizing"), unit="rule", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            t0 = time.time()
            done = 0
            if use_mp:
                if hasattr(os, 'fork'):
                    ctx = mp.get_context('fork')
                else:
                    ctx = mp.get_context('spawn')
                with ctx.Pool(processes=n_workers,
                              initializer=_minimize_worker_init,
                              initargs=(probe_words,)) as pool:
                    for sig, rule, count in pool.imap_unordered(_minimize_worker_sig, rule_items, chunksize=64):
                        if _kb.poll():
                            log_warn("[MIN] Early exit — keeping minimization progress so far")
                            pool.terminate()
                            break
                        depth = len(rule.split())
                        batch.append((sig, rule, count, depth))
                        if len(batch) >= MINIMIZE_DISK_BATCH_SIZE:
                            conn.executemany(_UPSERT, batch); conn.commit()
                            done += len(batch)
                            elapsed = time.time() - t0
                            spd = _fmt_speed(done / elapsed if elapsed>0 else 0, "rules")
                            n_sigs = conn.execute('SELECT COUNT(*) FROM sig_best').fetchone()[0]
                            pbar.set_postfix({"unique_sigs": cyan(str(n_sigs)), "spd": green(spd)}, refresh=False)
                            pbar.update(len(batch)); batch.clear()
            else:
                for rule, count in rule_items:
                    if _kb.poll():
                        log_warn("[MIN] Early exit — keeping minimization progress so far")
                        break
                    sig = compute_rule_signature_hash(rule, probe_words)
                    depth = len(rule.split())
                    batch.append((sig, rule, count, depth))
                    if len(batch) >= MINIMIZE_DISK_BATCH_SIZE:
                        conn.executemany(_UPSERT, batch); conn.commit()
                        done += len(batch)
                        elapsed = time.time() - t0
                        spd = _fmt_speed(done / elapsed if elapsed>0 else 0, "rules")
                        n_sigs = conn.execute('SELECT COUNT(*) FROM sig_best').fetchone()[0]
                        pbar.set_postfix({"unique_sigs": cyan(str(n_sigs)), "spd": green(spd)}, refresh=False)
                        pbar.update(len(batch)); batch.clear()
            if batch:
                conn.executemany(_UPSERT, batch); conn.commit()
                pbar.update(len(batch))
        survivors = Counter()
        for rule_str, cnt in conn.execute('SELECT rule, count FROM sig_best'):
            survivors[rule_str] = cnt
        n_unique_sigs = len(survivors)
        conn.close()
        removed = n_total - n_unique_sigs
        log_info(f"[MINIMIZE] {green('Done')}")
        log_info(f"           Unique signatures : {bold(cyan(str(n_unique_sigs))):>12s}")
        log_info(f"           Rules kept        : {bold(green(str(n_unique_sigs))):>12s}")
        log_info(f"           Rules removed     : {bold(red(str(removed))):>12s}  ({removed/max(1,n_total):.1%})")
        return survivors
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass
