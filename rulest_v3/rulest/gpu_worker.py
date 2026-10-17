"""
Process-isolated GPU worker used by multi-GPU mode.

A hung OpenCL driver call blocks the Python *thread* it runs on and cannot
be killed; a *process* can be (SIGTERM/SIGKILL). So each GPU in multi-GPU
mode gets its own OS process running a GPUEngine, talked to over
multiprocessing queues with a heartbeat; if a device's driver wedges, the
parent notices the silence and kills just that process instead of hanging
forever. Single-GPU runs skip all of this and use GPUEngine directly.
"""
import os
import time
import threading
import multiprocessing as mp
import queue as queue_module
from collections import Counter

import numpy as np
import pyopencl as cl

from .common import log_warn, _BLOOM_ALREADY_ON_GPU
from .gpu_engine import GPUEngine

_WORKER_HEARTBEAT_SECS  = 3     # how often the child pings "still alive"
_WORKER_HANG_TIMEOUT    = 480   # no heartbeat for this long -> declare the
                                 # device hung and kill its process (covers
                                 # the existing 180s+120s compile-retry path
                                 # with headroom)
_WORKER_STARTUP_TIMEOUT = 120   # time allowed for initialize_gpu() to reply

# Methods whose return value is either not picklable (a live pyopencl
# Program) or needs server-side resolution before it can cross the
# process boundary (the "already uploaded to this GPU's VRAM" sentinel).
# Handled specially in the worker loop instead of via plain getattr-forward.
_WORKER_SPECIAL_METHODS = {'compile_kernel', 'generate_bloom_filter_gpu'}

def _gpu_worker_entry(device_spec, params, cmd_q, res_q, hb_q):
    """Runs inside the child process. Owns exactly one real GPUEngine and
    executes RPC-style commands sent by the parent over cmd_q/res_q, while
    a background thread pings hb_q so the parent can tell 'still working'
    apart from 'driver wedged'."""
    try:
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
        eng = GPUEngine(params)
        ok = False
        try:
            ok = eng.initialize_gpu(device_spec)
        except Exception as e:
            res_q.put(('__init__', 'err', f"init crashed: {e}"))
            return
        dev_name = eng.device.name.strip() if (ok and eng.device) else None
        res_q.put(('__init__', 'ok' if ok else 'err', dev_name))
        if not ok:
            return

        stop_evt = threading.Event()
        def _ticker():
            while not stop_evt.is_set():
                try: hb_q.put(time.time())
                except Exception: pass
                stop_evt.wait(_WORKER_HEARTBEAT_SECS)
        threading.Thread(target=_ticker, daemon=True, name='hb-ticker').start()

        while True:
            try:
                call_id, method, args, kwargs = cmd_q.get()
            except Exception:
                break
            if method == '__shutdown__':
                break
            if method == '__set_params__':
                eng.params = args[0]
                try: res_q.put((call_id, 'ok', None))
                except Exception: pass
                continue
            try:
                if method == 'compile_kernel':
                    prog = eng.compile_kernel(*args, **kwargs)
                    result = prog is not None
                elif method == 'generate_bloom_filter_gpu':
                    bf = eng.generate_bloom_filter_gpu(*args, **kwargs)
                    if bf is _BLOOM_ALREADY_ON_GPU:
                        buf = eng.bloom_buf
                        bf_np = np.empty(buf.size // 4, dtype=np.int32)
                        cl.enqueue_copy(eng.queue, bf_np, buf)
                        eng.queue.finish()
                        bf = bf_np
                    result = bf
                elif method == '__getattr__':
                    # Only scalar CELF workspace metadata is exposed to the
                    # parent process. Never try to proxy live OpenCL buffers,
                    # kernels, queues, or other process-local GPU objects.
                    name = args[0]
                    if name not in {'_celf_seen_words', '_celf_batch_capacity'}:
                        raise AttributeError(f"Remote GPU attribute is not exposed: {name}")
                    result = getattr(eng, name)
                else:
                    fn = getattr(eng, method)
                    result = fn(*args, **kwargs)
                res_q.put((call_id, 'ok', result))
            except Exception as e:
                try:
                    res_q.put((call_id, 'err', f"{type(e).__name__}: {e}"))
                except Exception:
                    pass
        stop_evt.set()
    except Exception as e:
        try:
            res_q.put(('__init__', 'err', f"worker crashed before init: {e}"))
        except Exception:
            pass

class _RemoteDeviceHandle:
    """Stand-in for a pyopencl Device across the process boundary — callers
    only ever read .name off the primary engine's .device for logging."""
    def __init__(self, name):
        self.name = name or "remote device"

class ProcessEngineProxy:
    """Drop-in stand-in for a GPUEngine that actually lives in its own OS
    process, used by MultiGPUEngine/GeneticRuleEvolver in place of a real
    GPUEngine whenever more than one GPU is active. Mirrors the subset of
    the GPUEngine surface those callers use (.device, .disabled,
    .bloom_buf/.program/.rule_index as truthy readiness flags, .queue, and
    method calls). Any call that goes silent (no heartbeat) for longer
    than _WORKER_HANG_TIMEOUT gets its process hard-killed and the device
    marked disabled — a legitimately long GPU sweep keeps pinging the
    whole time, so this only trips on an actual wedged driver."""

    def __init__(self, params, device_spec):
        self._params = params
        self.device_spec = device_spec
        self.disabled = False
        self.device = None
        self._proc = None
        self._cmd_q = None
        self._res_q = None
        self._hb_q = None
        self._call_id = 0
        self._lock = threading.Lock()
        self._bloom_ready = False
        self._compiled = False
        self._rules_ready = False
        self._last_heartbeat = None
        self._hb_drain_stop = None
        self._hb_drain_thread = None

    # -- lifecycle --------------------------------------------------------
    def initialize_gpu(self, device_spec=None):
        if device_spec is not None:
            self.device_spec = device_spec
        ctx = mp.get_context('spawn')
        self._cmd_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._hb_q = ctx.Queue()
        self._proc = ctx.Process(target=_gpu_worker_entry,
                                  args=(self.device_spec, self._params,
                                        self._cmd_q, self._res_q, self._hb_q),
                                  daemon=True, name=f'gpu-worker-{self.device_spec}')
        self._proc.start()
        try:
            tag, status, payload = self._res_q.get(timeout=_WORKER_STARTUP_TIMEOUT)
        except Exception:
            log_warn(f"[GPU]  Worker for device {self.device_spec} did not respond to init in "
                     f"{_WORKER_STARTUP_TIMEOUT}s — killing")
            self._kill()
            return False
        if status != 'ok':
            log_warn(f"[GPU]  Worker for device {self.device_spec} failed to init: {payload}")
            self._kill()
            return False
        self.device = _RemoteDeviceHandle(payload)
        self._last_heartbeat = time.time()
        self._start_heartbeat_drain()
        return True

    def _start_heartbeat_drain(self):
        """Continuously drain hb_q in the background, independent of whether
        an RPC call is currently in flight. Without this, an engine that
        sits idle for a long stretch (e.g. every non-primary GPU during
        STAGE 3's genetic algorithm, which can legitimately run for hours
        and only ever talks to engines[0]) still gets a heartbeat ping every
        _WORKER_HEARTBEAT_SECS from its worker. Nobody was reading those
        pings between calls, so they piled up in the pipe's OS buffer; once
        that buffer filled, the child's ticker thread blocked on hb_q.put()
        forever, and that half-stuck queue could then hang the parent at
        shutdown (multiprocessing.Queue.join_thread() blocks until a
        queue's buffered data is flushed). Keeping the queue drained at all
        times avoids both problems."""
        stop_evt = threading.Event()
        def _drain():
            while not stop_evt.is_set():
                try:
                    self._hb_q.get(timeout=1.0)
                    self._last_heartbeat = time.time()
                except Exception:
                    pass
        t = threading.Thread(target=_drain, daemon=True, name=f'hb-drain-{self.device_spec}')
        self._hb_drain_stop = stop_evt
        self._hb_drain_thread = t
        t.start()

    def _kill(self):
        self.disabled = True
        if self._hb_drain_stop is not None:
            self._hb_drain_stop.set()
            self._hb_drain_stop = None
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
            except Exception:
                pass
        # Never let a leftover queue block interpreter/process shutdown —
        # cancel_join_thread() tells the Queue's internal feeder thread not
        # to wait for a full flush, which is the standard fix for
        # multiprocessing hanging on exit when a queue still has buffered
        # (or partially written) data nobody is going to read anymore.
        for q in (self._cmd_q, self._res_q, self._hb_q):
            if q is None: continue
            try: q.close()
            except Exception: pass
            try: q.cancel_join_thread()
            except Exception: pass

    def terminate(self):
        if self.disabled or self._proc is None:
            self._kill()
            return
        try:
            self._cmd_q.put((None, '__shutdown__', (), {}))
        except Exception:
            pass
        self._kill()

    # -- generic RPC --------------------------------------------------------
    def _call(self, method, *args, _raise_on_error=False, **kwargs):
        if self.disabled or self._proc is None:
            if _raise_on_error:
                raise RuntimeError(f"GPU worker is unavailable during '{method}'")
            return None
        with self._lock:
            self._call_id += 1
            call_id = self._call_id
            try:
                self._cmd_q.put((call_id, method, args, kwargs))
            except Exception as e:
                log_warn(f"[GPU]  Failed to send '{method}' to worker: {e}")
                self._kill()
                if _raise_on_error:
                    raise RuntimeError(f"Failed to send '{method}' to GPU worker") from e
                return None
            while True:
                try:
                    reply = self._res_q.get(timeout=1.0)
                except queue_module.Empty:
                    reply = None
                except Exception as e:
                    if _raise_on_error:
                        self._kill()
                        raise RuntimeError(f"Could not receive reply for '{method}' from GPU worker") from e
                    reply = None
                if reply is not None:
                    tag, status, payload = reply
                    if tag != call_id:
                        continue  # stale reply from an earlier, since-abandoned call
                    if status == 'ok':
                        return payload
                    log_warn(f"[GPU]  Worker error in '{method}': {payload}")
                    if _raise_on_error:
                        raise RuntimeError(f"GPU worker failed in '{method}': {payload}")
                    return None
                if self._proc is None or not self._proc.is_alive():
                    log_warn(f"[GPU]  Worker process died during '{method}' — dropping device")
                    self._kill()
                    if _raise_on_error:
                        raise RuntimeError(f"GPU worker died during '{method}'")
                    return None
                last_beat = self._last_heartbeat or 0
                if time.time() - last_beat > _WORKER_HANG_TIMEOUT:
                    log_warn(f"[GPU]  Worker unresponsive for {_WORKER_HANG_TIMEOUT}s during "
                             f"'{method}' — killing device (driver hang)")
                    self._kill()
                    if _raise_on_error:
                        raise RuntimeError(f"GPU worker became unresponsive during '{method}'")
                    return None

    # -- GPUEngine-compatible surface --------------------------------------
    @property
    def params(self): return self._params

    @params.setter
    def params(self, value):
        self._params = value
        if not self.disabled and self._proc is not None:
            self._call('__set_params__', value)

    @property
    def bloom_buf(self):
        return True if self._bloom_ready else None

    @property
    def program(self):
        return True if self._compiled else None

    @property
    def rule_index(self):
        return {'ready': True} if self._rules_ready else {}

    @property
    def queue(self):
        return True if (not self.disabled and self._proc is not None) else None

    def compile_kernel(self, force=False):
        ok = bool(self._call('compile_kernel', force=force))
        self._compiled = ok
        return ok

    def upload_bloom_filter(self, bf):
        r = self._call('upload_bloom_filter', bf)
        self._bloom_ready = not self.disabled
        return r

    def generate_bloom_filter_gpu(self, target_words):
        bf = self._call('generate_bloom_filter_gpu', target_words)
        if bf is not None:
            self._bloom_ready = True
        return bf

    def ensure_rules_ready(self, phase1_rules):
        r = self._call('ensure_rules_ready', phase1_rules)
        self._rules_ready = bool(r)
        return r

    def build_numeric_seed_families(self, max_depth):
        return self._call('build_numeric_seed_families', max_depth=max_depth) or {}

    def process_all_words_single_rule(self, base_words, rules, bloom_filter, _silent=False,
                                       _shared_combos=None, _shared_lock=None, _position=None,
                                       _label=None):
        # _shared_combos/_shared_lock are in-process threading primitives from
        # MultiGPUEngine._parallel_run and can't cross a process boundary — each
        # worker renders its own tqdm bar (pinned to _position, tagged _label)
        # instead, so every device gets a visible, independent line.
        return self._call('process_all_words_single_rule', base_words, rules, bloom_filter,
                           _silent=_silent, _position=_position, _label=_label) or Counter()

    def run_seed_extraction_pass(self, base_words, sbd, bloom_filter, phase1_rules, _silent=False,
                                  _position=None, _label=None):
        return self._call('run_seed_extraction_pass', base_words, sbd, bloom_filter,
                           phase1_rules, _silent=_silent, _position=_position, _label=_label) or Counter()

    def process_all_words_chain_rules(self, base_words, rules, max_depth, bloom_filter,
                                       single_counter, seed_chains=None, prebuilt_sbd=None, _silent=False,
                                       _position=None, _label=None):
        r = self._call('process_all_words_chain_rules', base_words, rules, max_depth, bloom_filter,
                        single_counter, seed_chains=seed_chains, prebuilt_sbd=prebuilt_sbd, _silent=_silent,
                        _position=_position, _label=_label)
        if r is not None: self._compiled = True; self._bloom_ready = True; self._rules_ready = True
        return r or Counter()

    def generate_informed_chains(self, rules, single_found, max_depth, seed_chains=None, prebuilt_sbd=None):
        return self._call('generate_informed_chains', rules, single_found, max_depth,
                           seed_chains, prebuilt_sbd) or []

    def _run_chains_against_words(self, base_words, chains, desc="  CHAINS  ", _silent=False,
                                   _shared_combos=None, _shared_lock=None, _position=None, _label=None):
        return self._call('_run_chains_against_words', base_words, chains, desc, _silent,
                           _position=_position, _label=_label) or Counter()

    def set_base_words(self, words):
        # Stage 3 (genetic) calls this once per engine up front, then reuses the
        # cached flattened copy in the worker process across every generation via
        # evaluate_chains_for_ga() below — avoids re-encoding the wordlist (and
        # re-sending it across the process boundary) every generation.
        return self._call('set_base_words', words)

    # CELF greedy-selection API. These calls deliberately stay on this one
    # worker for the whole selection run so its target table, active bitmap,
    # candidate pool, and hit-slot cache remain resident on the same GPU.
    @property
    def _celf_seen_words(self):
        return int(self._call('__getattr__', '_celf_seen_words', _raise_on_error=True))

    @property
    def _celf_batch_capacity(self):
        return int(self._call('__getattr__', '_celf_batch_capacity', _raise_on_error=True))

    def celf_initialize_target_table(self, target_words):
        return self._call('celf_initialize_target_table', target_words, _raise_on_error=True)

    def celf_prepare_candidate_pool(self, candidate_rules):
        return self._call('celf_prepare_candidate_pool', candidate_rules, _raise_on_error=True)

    def celf_score_chains_gpu(self, chains, record_hits=False, progress_callback=None, exact_counts=False):
        # `progress_callback` is a parent-process closure and must not be
        # pickled across multiprocessing queues. The GPU method's own result
        # is returned synchronously; progress reporting remains at the caller.
        call_kwargs = {'record_hits': record_hits, '_raise_on_error': True}
        if exact_counts:
            call_kwargs['exact_counts'] = True
        result = self._call('celf_score_chains_gpu', chains, **call_kwargs)
        if result is None:
            raise RuntimeError('GPU worker returned no CELF score result')
        return result

    def celf_take_cached_single_score(self, chain):
        return self._call('celf_take_cached_single_score', chain, _raise_on_error=True)

    def celf_target_ids_for_slots(self, slots):
        result = self._call('celf_target_ids_for_slots', slots, _raise_on_error=True)
        if result is None:
            raise RuntimeError('GPU worker returned no CELF target IDs')
        return result

    def celf_clear_target_ids(self, target_ids):
        return self._call('celf_clear_target_ids', target_ids, _raise_on_error=True)

    def celf_reset_active_targets(self):
        """Reset CELF coverage to the full target universe before a safe retry."""
        return self._call('celf_reset_active_targets', _raise_on_error=True)

    def celf_release_target_table(self):
        return self._call('celf_release_target_table', _raise_on_error=True)

    def celf_report_profile(self):
        return self._call('celf_report_profile', _raise_on_error=True)

    def celf_profile_summary(self):
        return self._call('celf_profile_summary', _raise_on_error=True)

    def evaluate_chains_for_ga(self, chains):
        r = self._call('evaluate_chains_for_ga', chains)
        return r if r is not None else {}

    def _compute_chain_seqs(self, chains):
        return self._call('_compute_chain_seqs', chains) or ([], [])

    def _run_chain_kernel(self, words, chains, _precomputed=None, return_word_idx=False):
        return self._call('_run_chain_kernel', words, chains, _precomputed, return_word_idx) or []

    def verify_pairs_exact(self, base_words, selected_chains, pairs, batch_size=200000):
        return self._call('verify_pairs_exact', base_words, selected_chains, pairs,
                           batch_size=batch_size) or [None] * len(pairs)

    def prepare_rule_pool(self, chains):
        return bool(self._call('prepare_rule_pool', chains))

    def _safe_queue_finish(self):
        return bool(self._call('_safe_queue_finish'))
