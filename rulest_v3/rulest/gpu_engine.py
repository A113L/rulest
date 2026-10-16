"""
Single-GPU OpenCL engine: kernel compile/cache/retry, bloom-filter build and
upload, and the single-rule / rule-chain search kernels launches. Also owns
driver-hang recovery (kernel build timeout + retry with the optimizer
disabled, and full context recreation after a fatal kernel error).
"""
import time
import shutil
import random
import os
import sys
import threading
import itertools
from collections import Counter, defaultdict

import numpy as np
import pyopencl as cl
from tqdm import tqdm

from .common import (
    log_info, log_warn, log_error, log_debug, bold, green, cyan, _kb,
    fnv1a_32, HashcatRuleValidator, _fmt_speed,
    MAX_WORD_LEN, MAX_RULE_LEN, MAX_OUTPUT_LEN, MAX_CHAIN_STRING_LEN,
    MAX_HASHCAT_CHAIN, BLOOM_HASH_FUNCTIONS, HOT_RULE_RATIO,
    MAX_ATTEMPTS_MULTIPLIER, _BLOOM_ALREADY_ON_GPU,
    FNV1A_SEED1, FNV1A_SEED2, SPECIAL_CHARS_TOP, SPECIAL_CHARS_CORE, LEET_OPS,
)
from .devices import estimate_free_vram, get_max_allocation, get_device_by_spec
from .hashing import fnv1a64_batch, build_indexed_open_addressing_table, table_size_for_count
from .kernel_source import GPU_KERNEL_TEMPLATE
from .progress import ActivityHeartbeat  # CELF_PROGRESS_DIAGNOSTICS_PATCH_V2
from . import hashing as _rulest_hashing

class _KernelBuildTimeout(Exception):
    """Raised by GPUEngine._safe_build_program when the OpenCL driver's
    build() call doesn't return within the watchdog timeout."""
    pass

class GPUEngine:
    def __init__(self, params):
        self.params = params
        self.context = self.queue = self.device = self.program = None
        self.max_work_group_size = 512
        self.local_work_size = params.get('LOCAL_WORK_SIZE', 512)
        self.bloom_buf = None
        self.bloom_np = None
        self.rule_index = {}
        self._gpu_rules = []
        self._rules_version = 0
        self.kernel_single = None
        self.kernel_chain = None
        self.kernel_bloom = None
        self.kernel_verify = None
        self.kernel_celf_score = None
        self.kernel_celf_clear = None
        self._consecutive_errors = 0
        self._MAX_CONSECUTIVE_ERRORS = 5
        self._cached_free_vram = None
        self._cached_max_alloc = None
        self.disabled = False  # set True if a kernel build hangs/times out — this
                                # device's driver may be stuck holding an internal
                                # lock, so we stop issuing further work to it rather
                                # than risk every subsequent call blocking forever.
        self._program_cache = {}  # MAX_CHAIN_DEPTH -> (program, kernel_single, kernel_chain, kernel_bloom, kernel_verify)
        self._rules_buf = None
        self._rules_offsets_buf = None
        self._rules_lengths_buf = None
        self._rules_buf_key = None
        self._bloom_recovery_fn = None
        self._compiled_depth = None   # track compiled MAX_CHAIN_DEPTH
        self._base_words_precomputed = None  # (wf, wo, wl)
        self._celf_target_hashes_buf = None
        self._celf_slot_target_ids_buf = None
        self._celf_occupied_buf = None
        self._celf_active_buf = None
        self._celf_target_bytes_buf = None
        self._celf_target_offsets_buf = None
        self._celf_target_lengths_buf = None
        self._celf_target_count = 0
        self._celf_table_size = self._celf_table_mask = self._celf_bitmap_words = 0
        self._celf_slot_target_ids_host = np.zeros(0, dtype=np.int32)
        self._celf_target_slots_host = np.zeros(0, dtype=np.int32)
        self._celf_targets = []
        self._celf_pending_clear_buffers = []
        self._celf_pending_clear_host_arrays = []
        self._celf_clear_slots_buf = None
        self._celf_profile = {
            "calls": 0, "batch_calls": 0, "singleton_calls": 0, "candidate_scores": 0,
            "wall_s": 0.0, "host_enqueue_s": 0.0, "host_alloc_s": 0.0, "wait_wall_s": 0.0,
            "kernel_gpu_s": 0.0, "h2d_gpu_s": 0.0, "fill_gpu_s": 0.0,
            "readback_gpu_s": 0.0, "exact_record_s": 0.0,
        }
        self._celf_base_words_buf = None
        self._celf_base_offsets_buf = None
        self._celf_base_lengths_buf = None
        self._celf_word_flat_upload_buf = None
        self._celf_word_offsets_upload_buf = None
        self._celf_word_lengths_upload_buf = None
        self._celf_words_resident = False

    @property
    def gpu_rules(self):
        return self._gpu_rules

    @gpu_rules.setter
    def gpu_rules(self, value):
        # Any wholesale replacement of the rule list bumps the version
        # counter so _get_rules_buffers() can detect the change with a
        # cheap (int, int) comparison instead of materializing/comparing a
        # tuple() of the whole (~thousands-of-rules) list on every kernel
        # launch.
        self._gpu_rules = value
        self._rules_version += 1

    def get_free_vram(self): return estimate_free_vram(self.device)
    def get_max_allocation(self): return get_max_allocation(self.device)

    def _refresh_cached_limits(self):
        self._cached_free_vram = estimate_free_vram(self.device)
        self._cached_max_alloc = get_max_allocation(self.device)

    def safe_output_buffer_size(self, words_count, chains_count):
        fv = self._cached_free_vram or self.get_free_vram()
        ma = self._cached_max_alloc or self.get_max_allocation()
        avail = min(fv, ma) - 5*1024**2
        # Must match the kernel constant MAX_CHAINS_TO_FIND = MAX_SAFE_RESULTS_PER_BATCH;
        # do NOT cap below that value or the kernel will write past the end of fo.
        return max(1, min(avail//MAX_CHAIN_STRING_LEN,
                          self.params['MAX_SAFE_RESULTS_PER_BATCH'],
                          words_count*chains_count))

    def initialize_gpu(self, device_spec):
        try:
            self.device = get_device_by_spec(device_spec)
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, properties=(cl.command_queue_properties.PROFILING_ENABLE if os.environ.get("RULEST_CELF_PROFILE") == "1" else 0))
            self.max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            self.local_work_size = min(self.local_work_size, self.max_work_group_size)
            while self.max_work_group_size % self.local_work_size != 0 and self.local_work_size > 32:
                self.local_work_size //= 2
            self._refresh_cached_limits()
            log_info(f"[GPU]  {bold(self.device.name.strip())}")
            vgb = self._cached_free_vram/1024**3
            log_debug(f"       WGS={self.local_work_size}, VRAM~{vgb:.1f}GB, CU={self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)}")
            return True
        except Exception as e: log_error(f"GPU init failed: {e}"); return False

    def compile_kernel(self, force=False):
        current_depth = self.params.get('MAX_CHAIN_DEPTH', 2)
        if not force and self.program is not None and self._compiled_depth == current_depth:
            log_debug("kernel already compiled with same depth")
            return self.program
        # Reuse a previously-built program for this exact depth if we have
        # one cached. The pipeline flips MAX_CHAIN_DEPTH back and forth
        # (e.g. Stage2 depth -> Seed-pass depth -> back to Stage2 depth), so
        # this avoids re-triggering a slow/flaky driver compile — a real fix
        # for the "hang after seed pass" case, since the Stage 2 recompile is
        # very often just reusing what was already built at pipeline start.
        cached = self._program_cache.get(current_depth)
        if cached is not None:
            (self.program, self.kernel_single, self.kernel_chain, self.kernel_bloom,
             self.kernel_verify, self.kernel_celf_score, self.kernel_celf_clear) = cached
            self._compiled_depth = current_depth
            log_info(f"[GPU]  Reusing cached kernel for depth {current_depth} (skipping rebuild)")
            return self.program
        if self.program:
            log_info(f"[GPU]  Recompiling kernel for depth {current_depth} (was {self._compiled_depth})")
        else:
            log_info("[GPU]  Compiling kernel ...")
        try:
            src = GPU_KERNEL_TEMPLATE.format(
                BLOOM_FILTER_SIZE=self.params['BLOOM_FILTER_SIZE'],
                BLOOM_NUM_SHARDS=self.params['BLOOM_NUM_SHARDS'],
                BLOOM_SHARD_BITS=self.params['BLOOM_SHARD_BITS'],
                BLOOM_SHARD_BYTES=self.params['BLOOM_SHARD_BYTES'],
                MAX_SAFE_RESULTS_PER_BATCH=self.params['MAX_SAFE_RESULTS_PER_BATCH'],
                MAX_CHAIN_DEPTH=current_depth,
                MAX_CHAIN_STRING_LEN=MAX_CHAIN_STRING_LEN,
                MAX_WORD_LEN=MAX_WORD_LEN,
                MAX_RULE_LEN=MAX_RULE_LEN,
                MAX_OUTPUT_LEN=MAX_OUTPUT_LEN,
                BLOOM_HASH_FUNCTIONS=BLOOM_HASH_FUNCTIONS,
            )
        except Exception as e:
            log_error(f"Kernel compile failed: {e}")
            return None

        dev_name = self.device.name.strip() if self.device else "device"
        prog = None
        try:
            prog = self._safe_build_program(src, timeout=180)
        except _KernelBuildTimeout:
            # Many Intel NEO / iGPU driver hangs are caused by the optimizer
            # choking on a particular kernel rather than the device being
            # truly broken. Give it one real second chance on a brand-new
            # context (the old one may still have a build stuck inside it)
            # with the optimizer disabled — this is a well-known practical
            # workaround for exactly this class of hang.
            log_warn(f"[GPU]  Build stalled on {dev_name} — retrying once on a fresh context "
                     f"with the optimizer disabled (-cl-opt-disable)")
            if self._recreate_context():
                try:
                    prog = self._safe_build_program(src, timeout=120, build_options=['-cl-opt-disable'])
                except _KernelBuildTimeout:
                    prog = None
                except Exception as e:
                    log_error(f"[GPU]  Retry build failed on {dev_name}: {e}")
                    prog = None
            else:
                prog = None
        except Exception as e:
            log_error(f"Kernel compile failed: {e}")
            return None

        if prog is None:
            log_error(f"[GPU]  Disabling {dev_name} after repeated build failure/hang")
            self.disabled = True
            self.context = None
            self.queue = None
            self.program = self.kernel_single = self.kernel_chain = self.kernel_bloom = self.kernel_verify = None
            self.kernel_celf_score = self.kernel_celf_clear = None
            return None

        self.program = prog
        self.kernel_single = prog.find_single_rules_gpu
        self.kernel_chain = prog.find_rule_chains_gpu
        self.kernel_bloom = prog.build_bloom_filter_gpu
        self.kernel_verify = prog.verify_pairs_gpu
        self.kernel_celf_score = prog.celf_score_chains_gpu
        self.kernel_celf_clear = prog.celf_clear_active_slots
        self._compiled_depth = current_depth
        self._program_cache[current_depth] = (
            prog, self.kernel_single, self.kernel_chain, self.kernel_bloom,
            self.kernel_verify, self.kernel_celf_score, self.kernel_celf_clear,
        )
        log_info("[GPU]  Kernel compiled successfully")
        return prog

    def _recreate_context(self):
        """Create a fresh OpenCL context/queue on the same device, abandoning
        a previous one that may still have a build stuck inside it. Clears
        buffer/kernel state tied to the old context since it's no longer
        valid to use."""
        try:
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, properties=(cl.command_queue_properties.PROFILING_ENABLE if os.environ.get("RULEST_CELF_PROFILE") == "1" else 0))
            for buf in [self.bloom_buf, self._rules_buf, self._rules_offsets_buf, self._rules_lengths_buf]:
                if buf is not None:
                    try: buf.release()
                    except Exception: pass
            self.bloom_buf = self._rules_buf = self._rules_offsets_buf = self._rules_lengths_buf = None
            self._rules_buf_key = None
            self.gpu_rules = []
            self.rule_index = {}
            self.program = self.kernel_single = self.kernel_chain = self.kernel_bloom = self.kernel_verify = None
            self.kernel_celf_score = self.kernel_celf_clear = None
            self._program_cache = {}  # programs from the old context are invalid
            return True
        except Exception as e:
            log_error(f"[GPU]  Failed to create fresh context for retry: {e}")
            return False

    def _reset_gpu(self, error):
        log_warn(f"[GPU] Fatal kernel error: {error} — resetting context")
        # Release all buffers
        for buf in [self.bloom_buf, self._rules_buf, self._rules_offsets_buf, self._rules_lengths_buf]:
            if buf is not None:
                try: buf.release()
                except Exception: pass
        if self.queue is not None:
            try: self.queue.finish()
            except Exception: pass
        # Clear state
        self.bloom_buf = self._rules_buf = self._rules_offsets_buf = self._rules_lengths_buf = None
        self._rules_buf_key = None  # force re-upload of rules after reset
        self.program = self.kernel_single = self.kernel_chain = self.kernel_bloom = self.kernel_verify = None
        self.kernel_celf_score = self.kernel_celf_clear = None
        self._program_cache = {}  # programs from the old context are invalid
        self.context = None
        self.queue = None
        self.gpu_rules = []
        self.rule_index = {}
        self._rules_buf_key = None
        # Recreate — NVIDIA OpenCL on Windows needs ~1-2 s to exit TDR/error state
        for attempt in range(3):
            try:
                time.sleep(1.5)
                self.context = cl.Context([self.device])
                self.queue = cl.CommandQueue(self.context, properties=(cl.command_queue_properties.PROFILING_ENABLE if os.environ.get("RULEST_CELF_PROFILE") == "1" else 0))
                if not self.compile_kernel(force=True):
                    return False
                if self._bloom_recovery_fn:
                    bf = self._bloom_recovery_fn()
                    self.upload_bloom_filter(bf)
                elif self.bloom_np is not None:
                    self.upload_bloom_filter(self.bloom_np)
                return True
            except Exception as exc:
                log_error(f"Reset failed (attempt {attempt+1}/3): {exc}")
                self.context = None
                self.queue = None
        return False

    def _safe_queue_finish(self):
        if self.queue is None: return False
        res = []
        def fin():
            try:
                self.queue.finish()
                res.append(True)
            except Exception:
                res.append(False)
        t = threading.Thread(target=fin, daemon=True); t.start()
        t.join(timeout=90)
        if t.is_alive(): return False
        return bool(res) and res[0]

    def _safe_build_program(self, src, timeout=180, build_options=None):
        """Build an OpenCL program on a watchdog thread so a driver-level
        compile hang (seen on some multi-GPU / laptop-hybrid setups) can't
        freeze the whole run forever. Mirrors _safe_queue_finish's pattern:
        if the build thread is still alive after `timeout` seconds, we give
        up on it (the thread is left as a daemon and abandoned — OpenCL
        offers no safe way to cancel an in-flight build) and return None so
        the caller can fail gracefully instead of hanging at "100%"."""
        prog = cl.Program(self.context, src)
        result = {}
        def do_build():
            try:
                prog.build(options=build_options or [])
                result['ok'] = True
            except Exception as e:
                result['ok'] = False
                result['err'] = e
        t = threading.Thread(target=do_build, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise _KernelBuildTimeout(f"build did not return within {timeout}s")
        if not result.get('ok'):
            raise result.get('err', RuntimeError("kernel build failed"))
        return prog

    def generate_bloom_filter(self, target_words):
        total_bytes = self.params['BLOOM_NUM_SHARDS'] * self.params['BLOOM_SHARD_BYTES']
        bf = np.zeros(total_bytes, dtype=np.uint8)
        log_info(f"[BLOOM] CPU build: {total_bytes//1024//1024}MB, {len(target_words):,} words, {self.params['BLOOM_NUM_SHARDS']} shard(s)")
        shard_bits = self.params['BLOOM_SHARD_BITS']
        shard_bytes = self.params['BLOOM_SHARD_BYTES']
        ns = self.params['BLOOM_NUM_SHARDS']
        ncols = shutil.get_terminal_size((80,24)).columns
        for w in tqdm(target_words, desc=green("  Bloom filter"), unit="word", ncols=ncols, leave=False,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
            wb = w.encode('latin-1')
            for s in range(ns):
                seed1 = (FNV1A_SEED1 + s) & 0xFFFFFFFF
                seed2 = (FNV1A_SEED2 + s) & 0xFFFFFFFF
                h1 = fnv1a_32(wb, seed1); h2 = fnv1a_32(wb, seed2)
                for i in range(BLOOM_HASH_FUNCTIONS):
                    idx = (h1 + i*h2) % shard_bits
                    byte_off = s*shard_bytes + (idx//8)
                    bf[byte_off] |= 1 << (idx%8)
        return bf

    def generate_bloom_filter_gpu(self, target_words):
        ns = self.params['BLOOM_NUM_SHARDS']
        shard_bytes = self.params['BLOOM_SHARD_BYTES']
        total_bytes = ns * shard_bytes
        nw = len(target_words)
        if not self.context or not self.program or not self.kernel_bloom:
            reason = ("no GPU context" if not self.context else
                      "kernel not compiled" if not self.program else
                      "bloom kernel missing after compile")
            log_warn(f"[BLOOM] GPU not ready ({reason}) — falling back to CPU build")
            return self.generate_bloom_filter(target_words)
        log_info(f"[BLOOM] GPU build: {total_bytes//1024//1024}MB, {nw} words, {ns} shards")
        mf = cl.mem_flags
        bufs = []
        try:
            wf, wo, wl = self._flatten(target_words)
            words_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=wf)
            offsets_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=wo)
            lengths_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=wl)
            bufs.extend([words_buf, offsets_buf, lengths_buf])
            bf_int32_size = (total_bytes + 3)//4
            bf_init = np.zeros(bf_int32_size, dtype=np.int32)
            bf_buf = cl.Buffer(self.context, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=bf_init)
            bufs.append(bf_buf)  # register for cleanup on exception; removed below on success
            gs = ((nw + self.local_work_size - 1) // self.local_work_size) * self.local_work_size
            self.kernel_bloom.set_args(words_buf, offsets_buf, lengths_buf, np.int32(nw), bf_buf)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_bloom, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish(): raise RuntimeError("queue finish timeout")
            # Verify the GPU actually wrote something non-trivial before trusting
            # this buffer. A bloom filter can legitimately have a very low fill
            # ratio (e.g. fill=0.001% on a large --bloom-mb with a small
            # wordlist), in which case the set bits are sparse and can easily
            # fall entirely outside a small sampled region — sampling only the
            # first few KB of a 256MB+ buffer produced false "all-zero"
            # positives and silently discarded valid GPU results every time.
            # Downloading the full buffer is a one-off cost (once per bloom
            # build, not per batch) and gives an exact answer instead of a
            # statistical guess.
            verify = np.empty(bf_int32_size, dtype=np.int32)
            cl.enqueue_copy(self.queue, verify, bf_buf)
            all_zero = np.count_nonzero(verify) == 0
            del verify
            if all_zero and nw > 0:
                log_warn("[BLOOM] GPU build all-zero -> fallback to CPU")
                raise RuntimeError("all-zero")
            if self.bloom_buf: self.bloom_buf.release()
            bufs.remove(bf_buf)  # hand ownership to self.bloom_buf; don't release in finally
            self.bloom_buf = bf_buf
            self.bloom_np = None
            self._bloom_recovery_fn = lambda: self.generate_bloom_filter(target_words)
            return _BLOOM_ALREADY_ON_GPU
        except Exception as exc:
            log_warn(f"[BLOOM] GPU build failed ({exc}) -> CPU")
            return self.generate_bloom_filter(target_words)
        finally:
            for b in bufs: b.release()

    def upload_bloom_filter(self, bf):
        if bf is _BLOOM_ALREADY_ON_GPU: return
        if self.bloom_buf: self.bloom_buf.release()
        self.bloom_np = bf
        self.bloom_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY|cl.mem_flags.COPY_HOST_PTR, hostbuf=bf)

    def _flatten(self, items):
        if not items:
            return (np.array([], dtype=np.uint8), np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        enc = [x.encode('latin-1') for x in items]
        flat = b''.join(enc)
        offs = [0]; lens = [len(enc[0])]
        for b in enc[1:]: offs.append(offs[-1]+lens[-1]); lens.append(len(b))
        return (np.frombuffer(flat, dtype=np.uint8), np.array(offs, dtype=np.int32), np.array(lens, dtype=np.int32))

    def prepare_words_data(self, words):
        wf,wo,wl = self._flatten(words)
        return dict(words_flat=wf, word_offsets=wo, word_lengths=wl, num_words=len(words), num_rules=len(self.gpu_rules))

    def _get_rules_buffers(self, mf):
        # Cheap version-counter key instead of tuple(self.gpu_rules) — avoids
        # rebuilding + hashing/comparing a ~thousands-of-rules tuple on every
        # kernel launch. Bumped automatically by the gpu_rules setter above
        # whenever the rule list is wholesale-replaced.
        key = (self._rules_version, len(self._gpu_rules))
        if self._rules_buf_key != key:
            for attr in ('_rules_buf','_rules_offsets_buf','_rules_lengths_buf'):
                if getattr(self, attr): getattr(self, attr).release()
            rf,ro,rl = self._flatten(self.gpu_rules)
            self._rules_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=rf)
            self._rules_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=ro)
            self._rules_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY|mf.COPY_HOST_PTR, hostbuf=rl)
            self._rules_buf_key = key
        return self._rules_buf, self._rules_offsets_buf, self._rules_lengths_buf

    def set_base_words(self, words):
        self._base_words_precomputed = self._flatten(words)

    def get_word_batch_data(self, start, end):
        if self._base_words_precomputed is None:
            raise RuntimeError("Base words not precomputed — call set_base_words() first")
        full_wf, full_wo, full_wl = self._base_words_precomputed
        if start >= end or start >= len(full_wo):
            return dict(words_flat=np.array([], dtype=np.uint8),
                        word_offsets=np.array([], dtype=np.int32),
                        word_lengths=np.array([], dtype=np.int32),
                        num_words=0, num_rules=len(self.gpu_rules))
        end = min(end, len(full_wo))
        base_off = int(full_wo[start])
        end_off = int(full_wo[end-1]) + int(full_wl[end-1])
        wf = full_wf[base_off:end_off]
        wo = full_wo[start:end] - base_off
        wl = full_wl[start:end]
        return dict(words_flat=wf, word_offsets=wo, word_lengths=wl,
                    num_words=end-start, num_rules=len(self.gpu_rules))

    def process_all_words_single_rule(self, base_words, rules, bloom_filter, _silent=False,
                                       _shared_combos=None, _shared_lock=None, _position=None,
                                       _label=None):
        self.upload_bloom_filter(bloom_filter)
        if not self.compile_kernel(): return Counter()
        # Refresh GPU rules list
        self.gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(rules)
        self.rule_index = {r:i for i,r in enumerate(self.gpu_rules)}
        counter = Counter()
        bs = self.params['WORDS_PER_BATCH']
        ncols = shutil.get_terminal_size((80,24)).columns
        t0 = time.time(); words_done = 0; nr = len(self.gpu_rules)
        desc = f"  STAGE 1 [{_label}] " if _label else "  STAGE 1 "
        with tqdm(total=len(base_words), desc=green(desc), unit="word", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  disable=_silent, position=_position, leave=True) as pbar:
            for i in range(0, len(base_words), bs):
                if _kb.poll(): break
                batch = base_words[i:i+bs]
                if batch:
                    found = self._run_single_kernel(self.prepare_words_data(batch))
                    if found:
                        self._consecutive_errors = 0
                        counter.update(found)
                    elif self.queue is None:
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS: break
                    words_done += len(batch)
                    elapsed = time.time()-t0
                    delta = len(batch) * nr
                    if _shared_combos is not None and _shared_lock is not None:
                        with _shared_lock:
                            _shared_combos[0] += delta
                        combos_for_speed = _shared_combos[0]
                    else:
                        combos_for_speed = words_done * nr
                    spd = _fmt_speed(combos_for_speed / elapsed if elapsed>0 else 0)
                    pbar.set_postfix({"rules": cyan(str(len(counter))), "spd": green(spd)}, refresh=False)
                pbar.update(len(batch))
        log_info(f"[S1]    {bold(green(str(len(counter))))} unique rules passed bloom filter")
        return counter

    def _run_single_kernel(self, bd):
        if not self.context or not self.queue or not self.kernel_single:
            return []
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f|mf.COPY_HOST_PTR, hostbuf=arr); bufs.append(b); return b
            bb = B(bd['words_flat']); bbo = B(bd['word_offsets']); bbl = B(bd['word_lengths'])
            rb, rbo, rbl = self._get_rules_buffers(mf)
            outs = self.safe_output_buffer_size(bd['num_words'], bd['num_rules'])
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4); bufs.append(fc)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))
            tot = bd['num_words'] * bd['num_rules']
            gs = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_single.set_args(bb, bbo, bbl, rb, rbo, rbl, self.bloom_buf,
                                        np.int32(bd['num_words']), np.int32(bd['num_rules']), fo, fc)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_single, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish():
                self._reset_gpu(RuntimeError("queue.finish() timed out or failed"))
                return []
            cnt = np.zeros(1, dtype=np.int32); cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs); out = []
            if n>0:
                data = np.zeros(n*MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, data, fo)
                for i in range(n):
                    r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]).split(b'\0')[0].decode('latin-1', errors='ignore')
                    if r: out.append(r)
            return out
        except Exception as e:
            log_warn(f"Single kernel error: {e}")
            if not self._reset_gpu(e):
                log_error("GPU reset failed - aborting batch")
            return []
        finally:
            for b in bufs:
                try: b.release()
                except Exception: pass

    def _gen_random_chains(self, depth, count, valid, hot, existing, new_set):
        gen = set(); max_att = count * MAX_ATTEMPTS_MULTIPLIER
        hot_budget = int(count * HOT_RULE_RATIO) if hot else 0
        # Each pass targets an *absolute* size for gen, not a relative delta.
        # Original code used (count-hot_budget) as the cold target, meaning cold
        # pass exited immediately once len(gen) exceeded that threshold — which it
        # always did after the hot pass when HOT_RULE_RATIO >= 0.5.
        for target, use_hot in [(hot_budget, True), (count, False)]:
            att = 0
            while len(gen) < target and att < max_att:
                att += 1
                if use_hot and hot:
                    hp = random.randint(0, depth-1)
                    parts = [random.choice(hot) if i==hp else random.choice(valid) for i in range(depth)]
                else:
                    parts = [random.choice(valid) for _ in range(depth)]
                k = ' '.join(parts)
                if k not in existing and k not in gen and k not in new_set:
                    gen.add(k)
        return gen

    def build_numeric_seed_families(self, max_depth):
        digits = '0123456789'
        sbd = defaultdict(set)
        # A
        for depth in range(1, min(4, max_depth)+1):
            for combo in itertools.product(digits, repeat=depth):
                sbd[depth].add(' '.join(f'^{ch}' for ch in reversed(combo)))
        # B
        for depth in range(1, min(4, max_depth)+1):
            for combo in itertools.product(digits, repeat=depth):
                sbd[depth].add(' '.join(f'${ch}' for ch in combo))
        # C
        for depth in range(1, min(4, max_depth)+1):
            for ops in itertools.product(['^','$'], repeat=depth):
                for digs in itertools.product(digits, repeat=depth):
                    sbd[depth].add(' '.join(f'{o}{d}' for o,d in zip(ops,digs)))
        # D
        transform_ops = ['l','u','c','C','t','r','d','f','E','k','K','{','}','[',']']
        t_digit_ops = [f'^{d}' for d in digits] + [f'${d}' for d in digits] + ['[',']']
        for depth in range(2, min(4, max_depth)+1):
            for t_op in transform_ops:
                for ops in itertools.product(t_digit_ops, repeat=depth-1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed):
                        sbd[depth].add(seed)
        # E (dates)
        days = [f"{d:02d}" for d in range(1,32)]
        months = [f"{m:02d}" for m in range(1,13)]
        years2 = [f"{y:02d}" for y in range(60,100)] + [f"{y:02d}" for y in range(0,31)]
        years4 = [str(y) for y in range(1960,2031)]
        date4 = set(); date6 = set(); date8 = set()
        for d in days:
            for m in months:
                date4.add(d+m); date4.add(m+d)
        for y in years4: date4.add(y)
        for d in days:
            for m in months:
                for y in years2:
                    date6.add(d+m+y); date6.add(m+d+y)
        for d in days:
            for m in months:
                for y in years4:
                    date8.add(d+m+y); date8.add(m+d+y)
        for dset,bd in ((date4,4),(date6,6),(date8,8)):
            if bd>max_depth: continue
            for ds in dset:
                sbd[bd].add(' '.join(f'${c}' for c in ds))
                sbd[bd].add(' '.join(f'^{c}' for c in reversed(ds)))
        if max_depth>=5:
            for ds in date4:
                app = ' '.join(f'${c}' for c in ds)
                pre = ' '.join(f'^{c}' for c in reversed(ds))
                for t_op in transform_ops:
                    for chain in (f"{t_op} {app}", f"{t_op} {pre}"):
                        if HashcatRuleValidator.validate_rule_for_gpu(chain):
                            sbd[5].add(chain)
        bracket_ops = ['[',']']
        for dset,bd,br in ((date4,4,range(2,5)),(date6,6,range(1,3)),(date8,8,range(1,2))):
            for num_b in br:
                nd = bd+num_b
                if nd>max_depth: continue
                for brackets in itertools.product(bracket_ops, repeat=num_b):
                    bpfx = ' '.join(brackets)
                    for ds in dset:
                        app = ' '.join(f'${c}' for c in ds)
                        pre = ' '.join(f'^{c}' for c in reversed(ds))
                        for chain in (f"{bpfx} {app}", f"{bpfx} {pre}"):
                            if HashcatRuleValidator.validate_rule_for_gpu(chain):
                                sbd[nd].add(chain)
        # F
        for depth in range(1, min(3,max_depth)+1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                chain = ' '.join(f'${ch}' for ch in combo)
                if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[depth].add(chain)
        # G
        for depth in range(1, min(3,max_depth)+1):
            for combo in itertools.product(SPECIAL_CHARS_TOP, repeat=depth):
                chain = ' '.join(f'^{ch}' for ch in reversed(combo))
                if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[depth].add(chain)
        # H
        sp_ops_top = [f'${ch}' for ch in SPECIAL_CHARS_TOP] + [f'^{ch}' for ch in SPECIAL_CHARS_TOP]
        for depth in range(2, min(3,max_depth)+1):
            for t_op in transform_ops:
                for ops in itertools.product(sp_ops_top, repeat=depth-1):
                    seed = f"{t_op} {' '.join(ops)}"
                    if HashcatRuleValidator.validate_rule_for_gpu(seed): sbd[depth].add(seed)
        # I
        for depth in range(2, min(4,max_depth)+1):
            n_digits = depth-1
            for digit_combo in itertools.product(digits, repeat=n_digits):
                for sp in SPECIAL_CHARS_CORE:
                    app = ' '.join(f'${d}' for d in digit_combo) + f' ${sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(app): sbd[depth].add(app)
                    pre = f'^{sp} ' + ' '.join(f'^{d}' for d in reversed(digit_combo))
                    if HashcatRuleValidator.validate_rule_for_gpu(pre): sbd[depth].add(pre)
        # J
        for op in LEET_OPS:
            if HashcatRuleValidator.validate_rule_for_gpu(op): sbd[1].add(op)
        if max_depth>=2:
            leet_followup = [f'${d}' for d in digits] + [f'^{d}' for d in digits] + \
                            [f'${ch}' for ch in SPECIAL_CHARS_CORE] + [f'^{ch}' for ch in SPECIAL_CHARS_CORE]
            for leet_op in LEET_OPS:
                for follow in leet_followup:
                    chain = f"{leet_op} {follow}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[2].add(chain)
            for l1 in LEET_OPS:
                for l2 in LEET_OPS:
                    if l1!=l2 and HashcatRuleValidator.validate_rule_for_gpu(f"{l1} {l2}"):
                        sbd[2].add(f"{l1} {l2}")
        # K
        if max_depth>=2:
            for t1 in transform_ops:
                for t2 in transform_ops:
                    chain = f"{t1} {t2}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain): sbd[2].add(chain)
        # L
        for depth in range(2, min(3,max_depth)+1):
            n_digits = depth-1
            for sp in SPECIAL_CHARS_CORE:
                for digit_combo in itertools.product(digits, repeat=n_digits):
                    app = f'${sp} ' + ' '.join(f'${d}' for d in digit_combo)
                    if HashcatRuleValidator.validate_rule_for_gpu(app): sbd[depth].add(app)
                    pre = ' '.join(f'^{d}' for d in digit_combo) + f' ^{sp}'
                    if HashcatRuleValidator.validate_rule_for_gpu(pre): sbd[depth].add(pre)
        # M
        if max_depth>=2:
            for leet_op in LEET_OPS:
                for t_op in transform_ops:
                    if HashcatRuleValidator.validate_rule_for_gpu(f"{leet_op} {t_op}"):
                        sbd[2].add(f"{leet_op} {t_op}")
                    if HashcatRuleValidator.validate_rule_for_gpu(f"{t_op} {leet_op}"):
                        sbd[2].add(f"{t_op} {leet_op}")
        return dict(sbd)

    def run_seed_extraction_pass(self, base_words, sbd, bloom_filter, phase1_rules, _silent=False,
                                  _position=None, _label=None):
        if self.bloom_buf is None: self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(phase1_rules)
            self.rule_index = {r:i for i,r in enumerate(self.gpu_rules)}
        multi = []
        for depth, chains in sorted(sbd.items()):
            if depth >= 2:
                multi.extend(chains)
        if not multi:
            log_info("[SEED]    No multi-depth seeds")
            return Counter()
        total = sum(len(v) for d,v in sbd.items() if d>=2)
        d_levels = sorted(d for d in sbd if d>=2)
        depth_range_str = (f"d{d_levels[0]}–d{d_levels[-1]}" if len(d_levels) > 1
                           else (f"d{d_levels[0]}" if d_levels else "none"))
        log_info(f"[SEED]    Numeric seed pass: {total:,} chains across {len(d_levels)} depth level(s) ({depth_range_str})")
        counter = self._run_chains_against_words(base_words, multi, "  SEED PASS ", _silent,
                                                   _position=_position, _label=_label)
        log_info(f"[SEED]    {bold(green(str(len(counter))))} unique seed chains passed bloom filter")
        return counter

    def generate_informed_chains(self, rules, single_found, max_depth, seed_chains=None, prebuilt_sbd=None):
        max_depth = min(max_depth, MAX_HASHCAT_CHAIN)
        valid = [r for r in rules if HashcatRuleValidator.validate_rule_for_gpu(r)]
        if not valid: return []
        found_s = set(single_found.keys()) if single_found else set()
        hot = [r for r in valid if r in found_s]
        chains = set(valid)
        if seed_chains:
            for sc in seed_chains:
                if sc.count(' ') >= 1: chains.add(sc)
        for depth in range(2, max_depth+1):
            budget = self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 0)
            if budget <= 0: continue
            budget = min(budget, len(valid)**depth)
            new = self._gen_random_chains(depth, budget, valid, hot, chains, set())
            chains.update(new)
        return list(chains)

    def process_all_words_chain_rules(self, base_words, rules, max_depth, bloom_filter,
                                      single_counter, seed_chains=None, prebuilt_sbd=None, _silent=False,
                                      _position=None, _label=None):
        if self.bloom_buf is None: self.upload_bloom_filter(bloom_filter)
        if not self.program:
            if not self.compile_kernel(): return Counter()
        if not self.rule_index:
            self.rule_index = {r:i for i,r in enumerate(self.gpu_rules)}
        chains = self.generate_informed_chains(rules, single_counter, max_depth, seed_chains, prebuilt_sbd)
        if not chains: return Counter()
        counter = self._run_chains_against_words(base_words, chains, "  STAGE 2 ", _silent,
                                                   _position=_position, _label=_label)
        log_info(f"[S2]    {bold(green(str(len(counter))))} unique chain rules passed bloom filter")
        return counter

    def _run_chains_against_words(self, base_words, chains, desc="  CHAINS  ", _silent=False,
                                   _shared_combos=None, _shared_lock=None, _position=None,
                                   _label=None):
        """Shared inner loop: iterate chain batches x word sub-batches, collect matches.
        Uses the same pre-flattened word-batch fast path the original seed-extraction
        pass used (set_base_words/get_word_batch_data), avoiding repeated
        latin-1 re-encoding of base_words on every chain batch."""
        counter = Counter()
        cbs = self.params['CHAINS_PER_BATCH']
        wsb = self.params['WORD_SUB_BATCH']
        n_batches = (len(chains) + cbs - 1) // cbs
        ncols = shutil.get_terminal_size((80, 24)).columns
        t0 = time.time()
        total_combos = 0
        self.set_base_words(base_words)
        bar_desc = f"{desc}[{_label}] " if _label else desc
        with tqdm(total=n_batches, desc=green(bar_desc), unit="batch", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
                  disable=_silent, position=_position, leave=True) as pbar:
            for ci in range(0, len(chains), cbs):
                if _kb.poll(): break
                cb = chains[ci:ci + cbs]
                seqs, depths = self._compute_chain_seqs(cb)
                for wi in range(0, len(base_words), wsb):
                    end = min(wi + wsb, len(base_words))
                    bd = self.get_word_batch_data(wi, end)
                    if bd['num_words'] == 0: continue
                    found = self._run_chain_kernel(bd, cb, (seqs, depths))
                    if found is None:
                        self._consecutive_errors += 1
                        if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                            log_error(f"Too many consecutive GPU errors, aborting {desc.strip()}")
                            break
                    elif found:
                        self._consecutive_errors = 0
                        counter.update(found)
                if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS: break
                delta = len(cb) * len(base_words)
                total_combos += delta
                if _shared_combos is not None and _shared_lock is not None:
                    with _shared_lock:
                        _shared_combos[0] += delta
                    combos_for_speed = _shared_combos[0]
                else:
                    combos_for_speed = total_combos
                elapsed = time.time() - t0
                spd = _fmt_speed(combos_for_speed / elapsed if elapsed > 0 else 0)
                pbar.set_postfix({"rules": cyan(str(len(counter))), "spd": green(spd)}, refresh=False)
                pbar.update(1)
        return counter

    def evaluate_chains_for_ga(self, chains):
        """Stage 3 (genetic) companion to _run_chains_against_words above. Runs the
        full chain-batch x word-sub-batch sweep for one engine's slice of chains
        against whatever base_words were already cached via set_base_words(), and
        returns {chain_str: set(global_word_index)} for that slice only.

        This is exposed as a single high-level call (rather than the parent
        driving get_word_batch_data/_run_chain_kernel/etc one RPC at a time) so
        that in multi-GPU mode a non-primary engine's ProcessEngineProxy only
        needs ONE round trip per generation instead of dozens — each of those
        would otherwise pickle numpy word/result arrays across the process
        boundary redundantly (worker -> parent -> worker) for no reason, since
        the parent never needed to see the intermediate word batch at all."""
        cbs = self.params.get('CHAINS_PER_BATCH', 2000)
        wsb = self.params.get('WORD_SUB_BATCH', 20000)
        n_words_total = len(self._base_words_precomputed[1]) if getattr(self, '_base_words_precomputed', None) else 0
        hit_words = defaultdict(set)
        for ci in range(0, len(chains), cbs):
            cb = chains[ci:ci+cbs]
            seqs, depths = self._compute_chain_seqs(cb)
            for wi in range(0, n_words_total, wsb):
                end = min(wi + wsb, n_words_total)
                bd = self.get_word_batch_data(wi, end)
                if bd['num_words'] == 0: continue
                found = self._run_chain_kernel(bd, cb, (seqs, depths), return_word_idx=True)
                if found:
                    for local_idx, rule_str in found:
                        hit_words[rule_str].add(wi + local_idx)
            # _run_chain_kernel() above already calls _safe_queue_finish()
            # internally for every sub-batch it launches, so an extra call
            # here was pure redundant synchronization overhead.
        return dict(hit_words)

    def _compute_chain_seqs(self, chains):
        maxd = self.params['MAX_CHAIN_DEPTH']
        seqs = []
        depths = []
        truncated = 0
        for c in chains:
            parts = c.split()
            if len(parts) > maxd:
                truncated += 1
                parts = parts[:maxd]
            depths.append(len(parts))
            idxs = [self.rule_index.get(r, -1) for r in parts]
            while len(idxs) < maxd:
                idxs.append(-1)
            seqs.extend(idxs)
        if truncated:
            log_warn(f"[GPU] {truncated} chain(s) truncated to max depth {maxd}")
        return seqs, depths

    def _run_chain_kernel(self, words, chains, _precomputed=None, return_word_idx=False):
        # `words` may be either a raw list of word strings, or an already-flattened
        # batch dict as returned by get_word_batch_data()/prepare_words_data() —
        # accept both so callers can pre-flatten once (SEED PASS) and avoid re-flattening
        # every sub-batch, without silently corrupting the word count (that was the bug:
        # passing a dict straight to prepare_words_data() iterated over its *keys* as
        # if they were words, and len(dict) was used as the word count).
        if not self.context or not self.queue or not self.kernel_chain:
            return []
        if _precomputed is not None:
            seqs_np = np.array(_precomputed[0], dtype=np.int32)
            depths_np = np.array(_precomputed[1], dtype=np.int32)
        else:
            seqs, depths = self._compute_chain_seqs(chains)
            seqs_np = np.array(seqs, dtype=np.int32)
            depths_np = np.array(depths, dtype=np.int32)
        if isinstance(words, dict):
            bd = words
        else:
            bd = self.prepare_words_data(words)
        n_words = bd['num_words']
        if n_words == 0 or not chains:
            return []
        mf = cl.mem_flags; bufs = []
        try:
            def B(arr, f=mf.READ_ONLY):
                b = cl.Buffer(self.context, f|mf.COPY_HOST_PTR, hostbuf=arr); bufs.append(b); return b
            bb = B(bd['words_flat']); bbo = B(bd['word_offsets']); bbl = B(bd['word_lengths'])
            rb, rbo, rbl = self._get_rules_buffers(mf)
            csb = B(seqs_np); cdb = B(depths_np)
            outs = self.safe_output_buffer_size(n_words, len(chains))
            fo = cl.Buffer(self.context, mf.WRITE_ONLY, outs*MAX_CHAIN_STRING_LEN); bufs.append(fo)
            fc = cl.Buffer(self.context, mf.READ_WRITE, 4); bufs.append(fc)
            # foundw: parallel per-hit word-index buffer. Always allocated/passed since it's
            # now part of the compiled kernel's fixed signature; only decoded when the caller
            # (Stage 3 GA marginal-coverage fitness) actually needs per-word hit identity.
            fw = cl.Buffer(self.context, mf.WRITE_ONLY, outs*4); bufs.append(fw)
            cl.enqueue_copy(self.queue, fc, np.array([0], dtype=np.int32))
            tot = n_words*len(chains)
            gs = ((tot+self.local_work_size-1)//self.local_work_size)*self.local_work_size
            self.kernel_chain.set_args(bb, bbo, bbl, rb, rbo, rbl, csb, cdb, self.bloom_buf,
                                       np.int32(n_words), np.int32(len(chains)),
                                       np.int32(self.params['MAX_CHAIN_DEPTH']), fo, fc, fw)
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_chain, (gs,), (self.local_work_size,))
            if not self._safe_queue_finish():
                self._reset_gpu(RuntimeError("queue.finish() timed out or failed"))
                return None
            cnt = np.zeros(1, dtype=np.int32); cl.enqueue_copy(self.queue, cnt, fc)
            n = min(cnt[0], outs)
            if n<=0:
                return []
            data = np.zeros(n*MAX_CHAIN_STRING_LEN, dtype=np.uint8)
            cl.enqueue_copy(self.queue, data, fo)
            if return_word_idx:
                widx = np.zeros(n, dtype=np.int32)
                cl.enqueue_copy(self.queue, widx, fw)
                out = []
                for i in range(n):
                    r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]).split(b'\0')[0].decode('latin-1', errors='ignore')
                    if r: out.append((int(widx[i]), r))
                return out
            out = []
            for i in range(n):
                r = bytes(data[i*MAX_CHAIN_STRING_LEN:(i+1)*MAX_CHAIN_STRING_LEN]).split(b'\0')[0].decode('latin-1', errors='ignore')
                if r: out.append(r)
            return out
        except Exception as e:
            log_warn(f"Chain kernel error: {e}"); self._reset_gpu(e); return None
        finally:
            for b in bufs:
                try: b.release()
                except Exception: pass

    def celf_initialize_target_table(self, target_words):
        """Build a <=50%-loaded 64-bit fingerprint table and upload it once.

        The slot occupancy and active-target bitmaps are separate uint32 arrays.
        Each active bit is indexed by the same slot returned by open addressing.
        """
        _celf_hashmap_activity = ActivityHeartbeat(green("[CELF GPU] Building target hash map"), interval=0.5).start()  # CELF_PROGRESS_DIAGNOSTICS_PATCH_V2
        _table_t0 = time.perf_counter()
        if not self.context or not self.compile_kernel():
            raise RuntimeError("CELF GPU kernels are unavailable")
        targets = target_words if isinstance(target_words, list) else list(target_words)
        if not targets:
            raise ValueError("CELF target table needs at least one target")
        # Same sizing rule as before (next power of two at >=2x the target
        # count, i.e. <=50% load factor), expressed via the shared helper.
        table_size = table_size_for_count(len(targets), load_factor=0.50)
        bitmap_words = (table_size + 31) // 32
        # RULEST_CELF_V11_PACKED_FASTPATH: single-pass bytearray pack; hash directly from this payload.
        _celf_pack_t0 = time.perf_counter()
        target_slots = np.full(len(targets), -1, dtype=np.int32)
        target_lengths = np.zeros(len(targets), dtype=np.int32)
        target_offsets = np.zeros(len(targets), dtype=np.uint64)
        _valid_ids_storage = np.empty(len(targets), dtype=np.int32)
        _packed_target_payload = bytearray()
        _valid_count = 0
        for target_id, target in enumerate(targets):
            raw = target.encode("latin-1", errors="strict")
            length = len(raw)
            target_lengths[target_id] = length
            if 0 < length <= MAX_OUTPUT_LEN:
                target_offsets[target_id] = len(_packed_target_payload)
                _valid_ids_storage[_valid_count] = target_id
                _valid_count += 1
                _packed_target_payload.extend(raw)
        valid_ids = _valid_ids_storage[:_valid_count]
        packed_byte_count = len(_packed_target_payload)
        target_bytes = (np.frombuffer(_packed_target_payload, dtype=np.uint8) if packed_byte_count
                            else np.zeros(1, dtype=np.uint8))
        _celf_pack_seconds = time.perf_counter() - _celf_pack_t0
        if os.environ.get("RULEST_CELF_PROFILE") == "1":
            print(f"[CELF] packed target bytes: {packed_byte_count:,} bytes (dense stride would be {len(targets) * MAX_OUTPUT_LEN:,} bytes); pack={_celf_pack_seconds:.2f}s", flush=True)

        # Vectorized FNV-1a-64 over the padded byte matrix: one Python loop
        # over columns (bounded by MAX_OUTPUT_LEN), not one per target. See
        # hashing.fnv1a64_batch / build_indexed_open_addressing_table for the
        # rule_ranker-derived approach this replaces the old per-key Python
        # "hash then probe in a for loop" insertion with.
        #
        # progress=False on both calls below: this whole block already runs
        # under the "[CELF GPU] Building target hash map" ActivityHeartbeat
        # (started above), which owns one overwritten status line for this
        # phase. Letting these two helpers print their own `[HASHMAP] ...`
        # progress lines at the same time used to interleave two unrelated,
        # differently-styled progress reporters -- the heartbeat's single
        # \r-updated line kept getting interrupted by a wall of separate
        # scrolling prints, which is the "hashmap building" noise this phase
        # was mainly adding.
        if valid_ids.size:
            fingerprints = _rulest_hashing.fnv1a64_packed_records_chunked(
                target_bytes,
                target_offsets,
                target_lengths,
                valid_ids,
                progress=False,
                label="target hashes",
            )
            (
                target_hashes,
                slot_target_ids,
                occupied,
                table_size,
                inserted,
            ) = build_indexed_open_addressing_table(
                fingerprints, ids=valid_ids, table_size=table_size, progress=False
            )
            bitmap_words = occupied.size
            placed = slot_target_ids >= 0
            target_slots[slot_target_ids[placed]] = np.flatnonzero(placed)
        else:
            target_hashes = np.zeros(table_size, dtype=np.uint64)
            slot_target_ids = np.full(table_size, -1, dtype=np.int32)
            occupied = np.zeros(bitmap_words, dtype=np.uint32)
            inserted = 0

        table_mask = table_size - 1
        _table_host_build_s = time.perf_counter() - _table_t0
        if inserted == 0:
            raise ValueError("No non-empty targets fit MAX_OUTPUT_LEN")
        active = occupied.copy()
        # Drop old buffers only after new host tables have been constructed.
        self.celf_release_target_table()
        mf = cl.mem_flags
        _table_upload_t0 = time.perf_counter()
        self._celf_target_hashes_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_hashes)
        self._celf_slot_target_ids_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=slot_target_ids)
        self._celf_occupied_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=occupied)
        self._celf_active_buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=active)
        self._celf_clear_slots_buf = cl.Buffer(self.context, mf.READ_ONLY, max(4, len(targets) * 4))
        self._celf_target_bytes_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_bytes)
        self._celf_target_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_offsets)

        self._celf_target_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_lengths)
        self._celf_target_count = len(targets)
        self._celf_table_size = table_size
        self._celf_table_mask = table_mask
        self._celf_bitmap_words = bitmap_words  # open-addressed table-slot bitmap
        self._celf_seen_words = (len(targets) + 31) // 32  # compact target-ID bitmap
        self._celf_slot_target_ids_host = slot_target_ids
        self._celf_target_slots_host = target_slots
        self._celf_occupied_host = occupied.copy()
        self._celf_pending_clear_buffers = []
        self._celf_pending_clear_host_arrays = []
        self._celf_targets = targets
        _deferred_word_buffer_logs = self._celf_initialize_word_buffers()
        load = inserted / table_size
        _celf_hashmap_activity.stop()  # CELF_PROGRESS_DIAGNOSTICS_PATCH_V2
        # Printed only now, after the heartbeat's final "done" line -- these
        # used to fire from inside _celf_initialize_word_buffers() while the
        # heartbeat was still live, landing mid-line on top of its
        # \r-overwritten status with no separating newline.
        for log_fn, message in _deferred_word_buffer_logs:
            log_fn(message)
        log_info(f"[CELF GPU] Target hash table: {inserted:,} entries, {table_size:,} slots, load={load:.1%}; FNV-1a 64 + SplitMix64; separate occupied/active bitmaps")
        if os.environ.get("RULEST_CELF_PROFILE") == "1":
            log_info(
                f"[CELF PROFILE] target-table host build/arrays={_table_host_build_s:.3f}s; "
                f"GPU buffer allocation+upload={time.perf_counter() - _table_upload_t0:.3f}s"
            )

    def _celf_initialize_word_buffers(self):
        """Keep base words on the GPU where possible; otherwise reuse upload buffers.

        Returns (log_fn, message) instead of printing directly: this runs
        from inside celf_initialize_target_table while the
        "[CELF GPU] Building target hash map" ActivityHeartbeat is still
        live, and a print() here used to land mid-line on top of the
        heartbeat's \\r-overwritten status (no newline between them -- the
        two outputs visibly ran together). The caller holds onto the
        message and prints it only after the heartbeat has stopped.
        """
        if self._base_words_precomputed is None:
            raise RuntimeError("CELF base words are not cached")
        wf, wo, wl = self._base_words_precomputed
        mf = cl.mem_flags
        self._celf_words_resident = False
        for name in ("_celf_base_words_buf", "_celf_base_offsets_buf", "_celf_base_lengths_buf",
                     "_celf_word_flat_upload_buf", "_celf_word_offsets_upload_buf", "_celf_word_lengths_upload_buf"):
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
                setattr(self, name, None)

        arrays = (wf, wo, wl)
        total_bytes = sum(int(array.nbytes) for array in arrays)
        try:
            free_vram = int(self.get_free_vram() or 0)
        except Exception:
            free_vram = 0
        try:
            max_alloc = int(self.get_max_allocation() or 0)
        except Exception:
            max_alloc = 0
        allocations_fit = max_alloc <= 0 or all(int(array.nbytes) <= max_alloc for array in arrays)
        memory_budget_ok = free_vram <= 0 or total_bytes <= max(16 * 1024 * 1024, int(free_vram * 0.30))

        if allocations_fit and memory_budget_ok and all(int(array.nbytes) > 0 for array in arrays):
            try:
                self._celf_base_words_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wf)
                self._celf_base_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wo)
                self._celf_base_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wl)
                self._celf_words_resident = True
                return [(log_info, f"[CELF GPU] Base word list resident on GPU ({total_bytes / (1024 * 1024):.1f} MiB); reused across CELF rescoring")]
            except Exception as exc:
                residency_warning = (log_warn, f"[CELF GPU] Full word-list residency unavailable; using reusable upload buffers: {exc}")
                for name in ("_celf_base_words_buf", "_celf_base_offsets_buf", "_celf_base_lengths_buf"):
                    buf = getattr(self, name, None)
                    if buf is not None:
                        try:
                            buf.release()
                        except Exception:
                            pass
                    setattr(self, name, None)

        word_batch = max(1, int(self.params.get("WORD_SUB_BATCH", 20000)))
        self._celf_word_flat_upload_buf = cl.Buffer(self.context, mf.READ_ONLY, max(1, word_batch * MAX_WORD_LEN))
        self._celf_word_offsets_upload_buf = cl.Buffer(self.context, mf.READ_ONLY, max(4, word_batch * 4))
        self._celf_word_lengths_upload_buf = cl.Buffer(self.context, mf.READ_ONLY, max(4, word_batch * 4))
        fallback_message = (log_info, "[CELF GPU] Base word list uses reusable GPU upload buffers (low-memory fallback)")
        if "residency_warning" in locals():
            # Surface both: why residency was abandoned, then what was used
            # instead. Caller prints them in this order, one per line.
            return [residency_warning, fallback_message]
        return [fallback_message]

    def celf_prepare_candidate_pool(self, chains):
        """Prepare token buffers and candidate chain sequences once per CELF run."""
        _pool_t0 = time.perf_counter()
        chains = list(chains)
        if not chains:
            raise ValueError("CELF candidate pool cannot be empty")
        self.prepare_rule_pool(chains)
        seqs, depths = self._compute_chain_seqs(chains)
        seqs_np = np.ascontiguousarray(seqs, dtype=np.int32)
        depths_np = np.ascontiguousarray(depths, dtype=np.int32)
        _pool_prepare_s = time.perf_counter() - _pool_t0
        _pool_alloc_t0 = time.perf_counter()
        mf = cl.mem_flags
        # Build/upload the atomic token buffer once, outside the CELF hot loop.
        self._celf_rules_buf, self._celf_rules_offsets_buf, self._celf_rules_lengths_buf = self._get_rules_buffers(mf)
        self._celf_candidate_seq_buf = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seqs_np
        )
        self._celf_candidate_depths_buf = cl.Buffer(
            self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=depths_np
        )
        self._celf_candidate_index = {chain: i for i, chain in enumerate(chains)}
        self._celf_candidate_count = len(chains)
        scratch_limit = max(4, int(self.params.get("CELF_SEEN_SCRATCH_MB", 32))) * 1024 * 1024
        by_scratch = max(1, scratch_limit // max(4, self._celf_seen_words * 4))
        configured = max(1, int(self.params.get("CELF_RULE_BATCH_SIZE", 32)))
        self._celf_batch_capacity = min(len(chains), configured, by_scratch)
        self._celf_candidate_indices_buf = cl.Buffer(
            self.context, mf.READ_ONLY, self._celf_batch_capacity * 4
        )
        self._celf_gains_buf = cl.Buffer(
            self.context, mf.READ_WRITE, self._celf_batch_capacity * 4
        )
        self._celf_seen_buf = cl.Buffer(
            self.context, mf.READ_WRITE,
            max(4, self._celf_batch_capacity * self._celf_seen_words * 4),
        )
        self._celf_hit_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)
        self._celf_hit_slots_buf = cl.Buffer(
            self.context, mf.READ_WRITE, max(1, self._celf_target_count) * 4
        )
        self._celf_active_version = getattr(self, "_celf_active_version", 0) + 1
        self._celf_last_single_cache = None
        log_info(
            f"[CELF GPU] Candidate pool prepared once: {len(self.gpu_rules):,} atomic tokens, "
            f"{self._celf_candidate_count:,} chains; persistent batch workspace={self._celf_batch_capacity}"
        )
        if os.environ.get("RULEST_CELF_PROFILE") == "1":
            log_info(
                f"[CELF PROFILE] candidate-pool seq/host prep={_pool_prepare_s:.3f}s; "
                f"GPU buffer allocation+upload={time.perf_counter() - _pool_alloc_t0:.3f}s"
            )

    def celf_score_chains_gpu(self, chains, record_hits=False, progress_callback=None):
        """Score via resident candidate sequences and persistent scratch/result buffers.

        Singleton scoring records hit slots on the GPU in the same pass. The slot
        list is not copied to host unless this candidate becomes the selected winner.
        """
        _profile_on = os.environ.get("RULEST_CELF_PROFILE") == "1"
        _wall_t0 = time.perf_counter()
        _host_enqueue_s = 0.0
        _host_alloc_s = 0.0
        _wait_wall_s = 0.0
        _h2d_events = []
        _fill_events = []
        _kernel_events = []
        _read_events = []
        if getattr(self, "_celf_active_buf", None) is None:
            raise RuntimeError("CELF target table has not been initialized")
        chains = list(chains)
        if not chains:
            return np.zeros(0, dtype=np.int64), []
        if record_hits and len(chains) != 1:
            raise ValueError("record_hits=True requires exactly one candidate chain")
        if not getattr(self, "_celf_candidate_index", None):
            raise RuntimeError("CELF candidate pool has not been prepared")
        if not self.context or not self.kernel_celf_score:
            raise RuntimeError("CELF GPU score kernel is unavailable")
        if len(chains) > self._celf_batch_capacity:
            raise ValueError(
                f"CELF batch of {len(chains)} exceeds persistent capacity {self._celf_batch_capacity}"
            )
        try:
            indices = [self._celf_candidate_index[chain] for chain in chains]
        except KeyError as exc:
            raise RuntimeError(f"Candidate was not in the prepared CELF pool: {exc}") from exc

        n_rules = len(chains)
        # Batched calls only compute a fast raw-hit upper bound. Exact singleton
        # calls explicitly request recording and deduplicate target IDs with seen[].
        do_record = bool(record_hits)
        self._celf_last_single_cache = None
        _enqueue_t0 = time.perf_counter()
        _alloc_t0 = time.perf_counter()
        indices_np = np.ascontiguousarray(indices, dtype=np.int32)
        _host_alloc_s += time.perf_counter() - _alloc_t0
        event = cl.enqueue_copy(self.queue, self._celf_candidate_indices_buf, indices_np, is_blocking=False)
        if _profile_on: _h2d_events.append(event)
        event = cl.enqueue_fill_buffer(self.queue, self._celf_gains_buf, np.int32(0), 0, n_rules * 4)
        if _profile_on: _fill_events.append(event)
        if do_record:
            seen_bytes = n_rules * self._celf_seen_words * 4
            event = cl.enqueue_fill_buffer(self.queue, self._celf_seen_buf, np.uint32(0), 0, max(4, seen_bytes))
            if _profile_on: _fill_events.append(event)
            event = cl.enqueue_fill_buffer(self.queue, self._celf_hit_count_buf, np.int32(0), 0, 4)
            if _profile_on: _fill_events.append(event)

        n_words_total = len(self._base_words_precomputed[1])  # CELF_N_WORDS_TOTAL_SCOPE_FIX
        word_batch = n_words_total if self._celf_words_resident else max(1, int(self.params.get("WORD_SUB_BATCH", 20000)))
        n_words_total = len(self._base_words_precomputed[1])
        for word_start in range(0, n_words_total, word_batch):
            word_end = min(word_start + word_batch, n_words_total)
            if self._celf_words_resident:
                num_words = word_end - word_start
                if num_words <= 0:
                    continue
                bb, bo, bl = self._celf_base_words_buf, self._celf_base_offsets_buf, self._celf_base_lengths_buf
                kernel_word_start = word_start
            else:
                bd = self.get_word_batch_data(word_start, word_end)
                num_words = int(bd["num_words"])
                if num_words <= 0:
                    continue
                event = cl.enqueue_copy(self.queue, self._celf_word_flat_upload_buf, bd["words_flat"], is_blocking=False)
                if _profile_on: _h2d_events.append(event)
                event = cl.enqueue_copy(self.queue, self._celf_word_offsets_upload_buf, bd["word_offsets"], is_blocking=False)
                if _profile_on: _h2d_events.append(event)
                event = cl.enqueue_copy(self.queue, self._celf_word_lengths_upload_buf, bd["word_lengths"], is_blocking=False)
                if _profile_on: _h2d_events.append(event)
                bb, bo, bl = self._celf_word_flat_upload_buf, self._celf_word_offsets_upload_buf, self._celf_word_lengths_upload_buf
                kernel_word_start = 0

            total = num_words * n_rules
            global_size = ((total + self.local_work_size - 1) // self.local_work_size) * self.local_work_size
            self.kernel_celf_score.set_args(
                bb, bo, bl,
                self._celf_rules_buf, self._celf_rules_offsets_buf, self._celf_rules_lengths_buf,
                self._celf_candidate_seq_buf, self._celf_candidate_depths_buf,
                self._celf_candidate_indices_buf,
                self._celf_target_hashes_buf, self._celf_slot_target_ids_buf,
                self._celf_occupied_buf, self._celf_active_buf,
                self._celf_target_bytes_buf, self._celf_target_offsets_buf, self._celf_target_lengths_buf,
                self._celf_seen_buf, self._celf_gains_buf,
                self._celf_hit_slots_buf, self._celf_hit_count_buf,
                np.int32(num_words), np.int32(n_rules), np.int32(kernel_word_start),
                np.int32(self.params["MAX_CHAIN_DEPTH"]),
                np.uint32(self._celf_table_mask), np.int32(self._celf_seen_words),
                np.int32(1 if do_record else 0),
            )
            event = cl.enqueue_nd_range_kernel(self.queue, self.kernel_celf_score, (global_size,), (self.local_work_size,))
            if _profile_on: _kernel_events.append(event)
            # NOTE: do NOT report `word_end` here. The kernel above is enqueued
            # asynchronously (no .wait()) -- it has not run yet, let alone
            # finished, at this point. Reporting completion here used to make
            # the CELF progress bar's "scan words" field jump to N/N (fully
            # scanned) the instant a kernel was merely *submitted*, which for
            # the common GPU-resident-wordlist case (word_batch ==
            # n_words_total, so this loop body runs exactly once) meant the
            # bar claimed "100% scanned" before the real, potentially
            # multi-minute GPU compute + readback below had even started --
            # then sat on that stale 100% with no further update until this
            # whole call returned. That read as the tool being frozen/broken
            # (CELF's "selected" count is genuinely 0 until round 1 finishes
            # -- that part is real, not a display bug -- but "scan words"
            # claiming to already be done was not). Report what has actually
            # been dispatched SO FAR (word_start, i.e. before this chunk),
            # which is honest about what has not yet executed.
            if progress_callback is not None:
                progress_callback(word_start, n_words_total)

        _host_enqueue_s += time.perf_counter() - _enqueue_t0
        _alloc_t0 = time.perf_counter()
        host_gains = np.zeros(n_rules, dtype=np.int32)
        host_count = np.zeros(1, dtype=np.int32)
        _host_alloc_s += time.perf_counter() - _alloc_t0
        gain_event = cl.enqueue_copy(self.queue, host_gains, self._celf_gains_buf, is_blocking=False)
        if _profile_on: _read_events.append(gain_event)
        if do_record:
            event = cl.enqueue_copy(self.queue, host_count, self._celf_hit_count_buf, is_blocking=False)
            if _profile_on: _read_events.append(event)
            # One queue synchronization covers stats plus every queued word chunk.
            _wait_t0 = time.perf_counter()
            self.queue.finish()
            _wait_wall_s += time.perf_counter() - _wait_t0
        else:
            _wait_t0 = time.perf_counter()
            gain_event.wait()
            _wait_wall_s += time.perf_counter() - _wait_t0
        # The real work (and the real wait) is above. Only now has the GPU
        # actually produced gains for every dispatched word chunk, so this is
        # the one point where reporting full completion is honest -- unlike
        # the per-chunk calls in the dispatch loop above, which report what
        # was *submitted*, never what has *finished*.
        if progress_callback is not None:
            progress_callback(n_words_total, n_words_total)
        self._celf_release_pending_clear_buffers()
        hit_slots = []
        if do_record:
            n_hits = min(max(0, int(host_count[0])), self._celf_target_count)
            self._celf_last_hit_count = n_hits
            if n_rules == 1:
                self._celf_last_single_cache = (
                    chains[0], int(host_gains[0]), self._celf_active_version, None, n_hits
                )
            if record_hits and n_hits:
                _alloc_t0 = time.perf_counter()
                host_slots = np.zeros(n_hits, dtype=np.int32)
                _host_alloc_s += time.perf_counter() - _alloc_t0
                event = cl.enqueue_copy(
                    self.queue, host_slots,
                    self._celf_hit_slots_buf.get_sub_region(0, n_hits * 4),
                    is_blocking=False,
                )
                if _profile_on: _read_events.append(event)
                _wait_t0 = time.perf_counter()
                event.wait()
                _wait_wall_s += time.perf_counter() - _wait_t0
                hit_slots = [int(slot) for slot in host_slots]
                if n_rules == 1:
                    self._celf_last_single_cache = (
                        chains[0], int(host_gains[0]), self._celf_active_version, hit_slots, n_hits
                    )
        else:
            self._celf_last_hit_count = 0

        if _profile_on:
            def _event_seconds(events):
                total = 0.0
                for ev in events:
                    try:
                        total += max(0, int(ev.profile.end) - int(ev.profile.start)) / 1e9
                    except Exception:
                        pass
                return total
            stats = self._celf_profile
            stats["calls"] += 1
            stats["batch_calls"] += int(n_rules > 1)
            stats["singleton_calls"] += int(n_rules == 1)
            stats["candidate_scores"] += n_rules
            stats["wall_s"] += time.perf_counter() - _wall_t0
            stats["host_enqueue_s"] += _host_enqueue_s
            stats["host_alloc_s"] += _host_alloc_s
            stats["wait_wall_s"] += _wait_wall_s
            stats["kernel_gpu_s"] += _event_seconds(_kernel_events)
            stats["h2d_gpu_s"] += _event_seconds(_h2d_events)
            stats["fill_gpu_s"] += _event_seconds(_fill_events)
            stats["readback_gpu_s"] += _event_seconds(_read_events)
            if do_record:
                stats["exact_record_s"] += _event_seconds(_kernel_events)
            _now = time.perf_counter()
            if _now - getattr(self, "_celf_profile_last_log", 0.0) >= 10.0:
                self._celf_profile_last_log = _now
                self.celf_report_profile()
        return host_gains.astype(np.int64), hit_slots

    def celf_take_cached_single_score(self, chain):
        """Reuse a just-scored singleton without launching another full wordlist scan."""
        cached = getattr(self, "_celf_last_single_cache", None)
        if cached is None:
            return None
        cached_chain, gain, version, host_slots, n_hits = cached
        if cached_chain != chain or version != self._celf_active_version:
            return None
        if host_slots is None:
            if n_hits:
                host = np.zeros(n_hits, dtype=np.int32)
                event = cl.enqueue_copy(
                    self.queue, host,
                    self._celf_hit_slots_buf.get_sub_region(0, n_hits * 4),
                    is_blocking=False,
                )
                event.wait()
                host_slots = [int(slot) for slot in host]
            else:
                host_slots = []
            self._celf_last_single_cache = (cached_chain, gain, version, host_slots, n_hits)
        return int(gain), list(host_slots)

    def celf_clear_last_hit_slots(self):
        """Clear the winning target slots directly from the reusable GPU buffer."""
        n_slots = int(getattr(self, "_celf_last_hit_count", 0))
        if n_slots > 0:
            global_size = ((n_slots + self.local_work_size - 1) // self.local_work_size) * self.local_work_size
            self.kernel_celf_clear.set_args(self._celf_active_buf, self._celf_hit_slots_buf, np.int32(n_slots))
            cl.enqueue_nd_range_kernel(self.queue, self.kernel_celf_clear, (global_size,), (self.local_work_size,))
        self._celf_active_version = getattr(self, "_celf_active_version", 0) + 1
        self._celf_last_single_cache = None

    def celf_target_ids_for_slots(self, slots):
        """Translate returned active table slots into original target IDs."""
        result = set()
        host_map = self._celf_slot_target_ids_host
        for slot in slots:
            if 0 <= int(slot) < len(host_map):
                target_id = int(host_map[int(slot)])
                if target_id >= 0:
                    result.add(target_id)
        return result

    def _celf_release_pending_clear_buffers(self):
        """Release temporary slot buffers only after queued GPU work has completed."""
        pending = getattr(self, "_celf_pending_clear_buffers", [])
        self._celf_pending_clear_buffers = []
        self._celf_pending_clear_host_arrays = []
        for buf in pending:
            try:
                buf.release()
            except Exception:
                pass

    def celf_reset_active_targets(self):
        """Restore the GPU active bitmap after a rejected speculative CELF attempt."""
        if not getattr(self, "_celf_active_buf", None):
            raise RuntimeError("CELF target table has not been initialized")
        self._celf_release_pending_clear_buffers()
        cl.enqueue_copy(self.queue, self._celf_active_buf, self._celf_occupied_host).wait()
        self._celf_active_version = getattr(self, "_celf_active_version", 0) + 1
        self._celf_last_single_cache = None

    def celf_clear_target_ids(self, target_ids):
        """Clear exact winner target IDs using a persistent upload buffer."""
        target_ids = sorted({int(target_id) for target_id in target_ids})
        slots = [int(self._celf_target_slots_host[target_id]) for target_id in target_ids
                 if 0 <= target_id < self._celf_target_count
                 and int(self._celf_target_slots_host[target_id]) >= 0]
        if not slots:
            return
        slots_np = np.ascontiguousarray(slots, dtype=np.int32)
        if self._celf_clear_slots_buf is None or len(slots) > self._celf_target_count:
            raise RuntimeError("CELF clear-slot workspace is not available or too small")
        cl.enqueue_copy(self.queue, self._celf_clear_slots_buf, slots_np, is_blocking=False)
        global_size = ((len(slots) + self.local_work_size - 1) // self.local_work_size) * self.local_work_size
        self.kernel_celf_clear.set_args(self._celf_active_buf, self._celf_clear_slots_buf, np.int32(len(slots)))
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_celf_clear, (global_size,), (self.local_work_size,))
        # Preserve the async H2D source array until the next queue synchronization.
        self._celf_pending_clear_host_arrays.append(slots_np)
        self._celf_active_version = getattr(self, "_celf_active_version", 0) + 1
        self._celf_last_single_cache = None

    def celf_report_profile(self):
        """Log recompute costs by host enqueue, GPU events and synchronization."""
        if os.environ.get("RULEST_CELF_PROFILE") != "1":
            return
        p = getattr(self, "_celf_profile", {})
        if not p or not p.get("calls"):
            log_info("[CELF PROFILE] no scoring calls collected")
            return
        calls = max(1, int(p["calls"]))
        log_info(
            "[CELF PROFILE] score calls={calls:,} (batch={batch:,}, singleton={single:,}), "
            "candidates={candidates:,}; wall={wall:.3f}s; host alloc={alloc:.3f}s; host enqueue={enqueue:.3f}s; "
            "GPU kernel={kernel:.3f}s; H2D={h2d:.3f}s; fills={fills:.3f}s; "
            "readback={readback:.3f}s; synchronization wait={wait:.3f}s; "
            "avg wall/call={avg:.2f}ms".format(
                calls=int(p["calls"]), batch=int(p["batch_calls"]), single=int(p["singleton_calls"]),
                candidates=int(p["candidate_scores"]), wall=float(p["wall_s"]),
                alloc=float(p["host_alloc_s"]), enqueue=float(p["host_enqueue_s"]), kernel=float(p["kernel_gpu_s"]),
                h2d=float(p["h2d_gpu_s"]), fills=float(p["fill_gpu_s"]),
                readback=float(p["readback_gpu_s"]), wait=float(p["wait_wall_s"]),
                avg=1000.0 * float(p["wall_s"]) / calls,
            )
        )

    def celf_release_target_table(self):
        """Release GPU-resident CELF table and associated host lookup state."""
        # RULEST_CELF_PACKED_TARGETS_V10_1: release packed-target offsets buffer idempotently.
        _celf_offsets_buf = getattr(self, "_celf_target_offsets_buf", None)
        if _celf_offsets_buf is not None:
            try:
                _celf_offsets_buf.release()
            except Exception:
                pass
        self._celf_target_offsets_buf = None

        try:
            if getattr(self, "queue", None) is not None:
                self.queue.finish()
        except Exception:
            pass
        self._celf_release_pending_clear_buffers()
        names = (
            "_celf_target_hashes_buf", "_celf_slot_target_ids_buf", "_celf_occupied_buf",
            "_celf_active_buf", "_celf_target_bytes_buf", "_celf_target_lengths_buf",
            "_celf_base_words_buf", "_celf_base_offsets_buf", "_celf_base_lengths_buf",
            "_celf_word_flat_upload_buf", "_celf_word_offsets_upload_buf", "_celf_word_lengths_upload_buf",
            "_celf_candidate_seq_buf", "_celf_candidate_depths_buf", "_celf_candidate_indices_buf",
            "_celf_gains_buf", "_celf_seen_buf", "_celf_hit_count_buf", "_celf_hit_slots_buf",
            "_celf_clear_slots_buf",
        )
        for name in names:
            buf = getattr(self, name, None)
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
            setattr(self, name, None)
        self._celf_target_count = 0
        self._celf_table_size = 0
        self._celf_table_mask = 0
        self._celf_bitmap_words = 0
        self._celf_seen_words = 0
        self._celf_candidate_index = {}
        self._celf_candidate_count = 0
        self._celf_batch_capacity = 0
        self._celf_last_hit_count = 0
        self._celf_last_single_cache = None
        self._celf_active_version = getattr(self, "_celf_active_version", 0) + 1
        self._celf_slot_target_ids_host = np.zeros(0, dtype=np.int32)
        self._celf_target_slots_host = np.zeros(0, dtype=np.int32)
        self._celf_occupied_host = np.zeros(0, dtype=np.uint32)
        self._celf_targets = []
        self._celf_words_resident = False
        self._celf_pending_clear_host_arrays = []

    def verify_pairs_exact(self, base_words, selected_chains, pairs, batch_size=200000):
        """--exact-recovery: for a small set of (chain_idx, base_word_idx)
        pairs (chain_idx indexes into `selected_chains`, base_word_idx
        indexes into `base_words`), compute the ACTUAL mangled output string
        on GPU (same apply() as the bloom-gated kernels, just unconditional
        + no bloom) and return them aligned with `pairs`, so the caller can
        check each one against a real (non-bloom) set of target_words. Meant
        to be called only with the already-small selected ruleset x its
        recorded hits — NOT a full base_words x rules sweep."""
        if not self.compile_kernel() or not self.kernel_verify:
            return [None] * len(pairs)
        # Atomic-token universe for these chains only (not the full phase1
        # pool) — keeps _get_rules_buffers()/rule_index scoped to exactly
        # what's needed for this verification pass.
        atomic = sorted({tok for c in selected_chains for tok in c.split()})
        # If the caller (evaluate_full_coverage's loop in selection.py)
        # already called prepare_rule_pool() with this same chain batch,
        # rule_index already covers every token we need — skip rebuilding
        # gpu_rules/rule_index (and the resulting rules-buffer re-upload in
        # _get_rules_buffers) a second time for nothing.
        if not all(tok in self.rule_index for tok in atomic):
            self.gpu_rules = atomic
            self.rule_index = {r: i for i, r in enumerate(atomic)}
        seqs, depths = self._compute_chain_seqs(selected_chains)
        mcd = self.params['MAX_CHAIN_DEPTH']
        seqs_np = np.array(seqs, dtype=np.int32)
        depths_np = np.array(depths, dtype=np.int32)
        bwf, bwo, bwl = self._flatten(base_words)
        results = [None] * len(pairs)
        mf = cl.mem_flags
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            npairs = len(chunk)
            if npairs == 0:
                continue
            pw = np.array([p[1] for p in chunk], dtype=np.int32)  # base_word idx
            pc = np.array([p[0] for p in chunk], dtype=np.int32)  # chain idx
            bufs = []
            try:
                def B(arr, f=mf.READ_ONLY):
                    b = cl.Buffer(self.context, f | mf.COPY_HOST_PTR, hostbuf=arr); bufs.append(b); return b
                bb = B(bwf); bbo = B(bwo); bbl = B(bwl)
                rb, rbo, rbl = self._get_rules_buffers(mf)
                csb = B(seqs_np); cdb = B(depths_np)
                pwb = B(pw); pcb = B(pc)
                out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, npairs * MAX_OUTPUT_LEN); bufs.append(out_buf)
                out_len_buf = cl.Buffer(self.context, mf.WRITE_ONLY, npairs * 4); bufs.append(out_len_buf)
                gs = ((npairs + self.local_work_size - 1) // self.local_work_size) * self.local_work_size
                self.kernel_verify.set_args(bb, bbo, bbl, rb, rbo, rbl, csb, cdb, pwb, pcb,
                                             np.int32(npairs), np.int32(mcd), out_buf, out_len_buf)
                cl.enqueue_nd_range_kernel(self.queue, self.kernel_verify, (gs,), (self.local_work_size,))
                if not self._safe_queue_finish():
                    self._reset_gpu(RuntimeError("queue.finish() timed out or failed"))
                    continue
                lens = np.zeros(npairs, dtype=np.int32)
                cl.enqueue_copy(self.queue, lens, out_len_buf)
                data = np.zeros(npairs * MAX_OUTPUT_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, data, out_buf)
                for i in range(npairs):
                    ol = int(lens[i])
                    if ol <= 0:
                        results[start + i] = ''
                        continue
                    s = bytes(data[i * MAX_OUTPUT_LEN:i * MAX_OUTPUT_LEN + ol]).decode('latin-1', errors='ignore')
                    results[start + i] = s
            except Exception as e:
                log_warn(f"Verify kernel error: {e}")
                self._reset_gpu(e)
            finally:
                for b in bufs:
                    try: b.release()
                    except Exception: pass
        return results

    def prepare_rule_pool(self, chains):
        """Force-rebuild gpu_rules/rule_index to cover exactly the atomic
        tokens used by `chains` (chain strings, e.g. "^t X u").

        Unlike ensure_rules_ready(), this always rebuilds — it does not
        early-return when rule_index is already non-empty. That matters
        because verify_pairs_exact() intentionally narrows rule_index to
        whatever chains it was last called with (see its docstring). If a
        caller then feeds a *different* batch of chains straight into
        evaluate_chains_for_ga()/_compute_chain_seqs() without re-syncing
        rule_index first, any token from the new batch that wasn't part of
        the old narrowed set silently resolves to -1 in rule_index.get(r,
        -1). The GPU chain kernel still appends that token's separator
        space to the found-string buffer *before* checking the index and
        bailing out, so the reported hit is a truncated, space-padded
        string (e.g. "^t ") that no longer matches the original chain
        string used as the Python-side dict key — the caller sees a
        KeyError instead of a clean hit. evaluate_full_coverage() (see
        selection.py) calls this once per batch, right before
        evaluate_population(), specifically to close that gap.
        """
        atomic = sorted({tok for c in chains for tok in c.split()})
        self.gpu_rules = atomic
        self.rule_index = {r: i for i, r in enumerate(atomic)}
        return True

    def ensure_rules_ready(self, phase1_rules):
        """Populate gpu_rules/rule_index on demand. Split out of the various
        call sites so it can also be invoked remotely as a single RPC by
        ProcessEngineProxy (a remote caller can't just poke .gpu_rules /
        .rule_index attributes and expect them to affect the real engine
        living in another process)."""
        if not self.rule_index:
            self.gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(phase1_rules)
            self.rule_index = {r: i for i, r in enumerate(self.gpu_rules)}
        return True
