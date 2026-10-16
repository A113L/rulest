"""
OpenCL device enumeration/selection and the dynamic-parameter calculator
that sizes batches, bloom filters, and time budgets from a device's compute
units and estimated free VRAM.
"""
import math

import pyopencl as cl

import sys

from .common import (
    log_info, log_warn, log_error, cyan, dim, blue,
    VRAM_USAGE_FACTOR, BASELINE_COMBOS_PER_SEC,
    LOW_END_COMPUTE_UNITS_THRESHOLD, LOW_END_COMBOS_PER_SEC,
    POSSIBLE_WORK_GROUP_SIZES, BASE_WORDS_PER_BATCH, BASE_CHAINS_PER_BATCH,
    BASE_WORD_SUB_BATCH, BASE_MAX_SAFE_RESULTS, MAX_SAFE_RESULTS_CAP,
    OPTIMAL_GLOBAL_MULTIPLIER_BASE, BLOOM_HASH_FUNCTIONS,
)

def get_all_devices():
    devs = []
    for plat in cl.get_platforms():
        for dtype in (cl.device_type.GPU, cl.device_type.CPU):
            try:
                for d in plat.get_devices(dtype): devs.append((plat,d))
            except Exception: pass
    return devs

def list_devices():
    devs = get_all_devices()
    if not devs: log_error("No OpenCL devices."); sys.exit(1)
    log_info(f"\n{blue('Available OpenCL devices:')}")
    for i,(p,d) in enumerate(devs):
        try:
            t = cl.device_type.to_string(d.get_info(cl.device_info.TYPE))
        except Exception:
            dtype = d.get_info(cl.device_info.TYPE)
            if dtype == cl.device_type.GPU: t = "GPU"
            elif dtype == cl.device_type.CPU: t = "CPU"
            else: t = "OTHER"
        log_info(f"  {cyan(str(i)+':')} {d.get_info(cl.device_info.NAME)} ({t}) — {p.name}")
    gpu_count = len(get_all_gpu_only_devices())
    if gpu_count > 1:
        log_info(f"\n  {dim('Tip:')} use {cyan('--device all')} to run on all {gpu_count} GPU(s) in parallel, "
                 f"or {cyan('--device 0,1')} to pick specific GPUs")
    print()

def get_all_gpu_only_devices():
    devs = []
    for p in cl.get_platforms():
        try:
            for d in p.get_devices(cl.device_type.GPU):
                devs.append(d)
        except Exception: pass
    return devs

def get_device_by_spec(spec):
    if spec is None: return get_best_gpu_device()
    spec = spec.strip()
    devs = get_all_devices()
    if not devs: raise RuntimeError("No OpenCL devices")
    if spec.isdigit():
        i = int(spec)
        if 0<=i<len(devs): return devs[i][1]
        raise RuntimeError(f"Index {i} out of range")
    lo = spec.lower()
    match = [d for _,d in devs if lo in d.get_info(cl.device_info.NAME).lower()]
    if len(match)==1: return match[0]
    if match: return match[0]
    raise RuntimeError(f"No device matching '{spec}'")

def get_best_gpu_device():
    best = None; best_score = -1
    for p in cl.get_platforms():
        try: devs = p.get_devices(cl.device_type.GPU)
        except Exception: continue
        for d in devs:
            name = d.get_info(cl.device_info.NAME).upper()
            score = (10 if 'NVIDIA' in name or 'AMD' in name else 0)
            score += (5 if 'RTX' in name or 'GTX' in name else 0)
            score += d.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            if score > best_score: best_score = score; best = d
    if best is None:
        for p in cl.get_platforms():
            try: return p.get_devices(cl.device_type.GPU)[0]
            except Exception: pass
    if best is None: raise RuntimeError("No GPU found")
    return best

def estimate_free_vram(device):
    try: return int(device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * VRAM_USAGE_FACTOR)
    except: return 1<<30

def get_max_allocation(device):
    try: return device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
    except: return 1<<30

# ----------------------------------------------------------------------
# Bloom size & dynamic parameters (power-of-two & sharding logic)
# ----------------------------------------------------------------------
def get_auto_bloom_mb(free_vram_gb):
    if free_vram_gb >= 8.0: return 512
    elif free_vram_gb >= 4.0: return 256
    else: return 128

def calculate_dynamic_parameters(base_count, target_count, device=None, target_hours=0.5,
                                  bloom_mb_override=None, bloom_no_shard=False):
    if device:
        try:
            mwgs = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            mcu = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            fv = estimate_free_vram(device)
            vgb = fv / 1024**3
            name_up = device.get_info(cl.device_info.NAME).upper()
            lws = max(s for s in POSSIBLE_WORK_GROUP_SIZES if s <= mwgs)
            if 'NVIDIA' in name_up and mcu >= 38: lws = min(512, lws)
            est = LOW_END_COMBOS_PER_SEC if mcu < LOW_END_COMPUTE_UNITS_THRESHOLD else BASELINE_COMBOS_PER_SEC
            log_info(f"[GPU]  CU={mcu}, VRAM~{vgb:.1f}GB, WGS={lws}, est={est//1_000_000}M/s")
        except:
            lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2<<30; vgb = 2.0
    else:
        lws = 256; est = BASELINE_COMBOS_PER_SEC; mcu = 38; fv = 2<<30; vgb = 2.0

    vram_scale = max(0.25, min(1.0, vgb/8.0))
    target_sec = target_hours * 3600
    if bloom_mb_override and bloom_mb_override > 0:
        eff_bloom = bloom_mb_override
    else:
        eff_bloom = get_auto_bloom_mb(vgb)

    total_bits = eff_bloom * 1024 * 1024 * 8
    if total_bits & (total_bits - 1):
        total_bits = 1 << total_bits.bit_length()
        eff_bloom = total_bits // (1024*1024*8)
    SHARD_BITS = 1 << 32
    if not bloom_no_shard and total_bits > SHARD_BITS:
        num_shards = (total_bits + SHARD_BITS - 1) // SHARD_BITS
        shard_bits = SHARD_BITS
        bloom_bits = num_shards * SHARD_BITS
        log_info(f"[BLOOM] Sharding enabled: {eff_bloom}MB -> {num_shards} shards of 512MB, total {bloom_bits//(1024*1024*8)}MB")
    else:
        num_shards = 1
        if total_bits & (total_bits - 1):
            total_bits = 1 << total_bits.bit_length()
            eff_bloom  = total_bits // (1024 * 1024 * 8)
        shard_bits = total_bits
        bloom_bits = total_bits
        log_info(f"[BLOOM] Single shard (fast path), size {bloom_bits//(1024*1024*8)}MB")

    if target_count > 0:
        fill = 1.0 - math.exp(-BLOOM_HASH_FUNCTIONS * target_count / bloom_bits)
        fpr = fill ** BLOOM_HASH_FUNCTIONS
        log_info(f"[BLOOM] {bloom_bits//8//1024//1024}MB, shards={num_shards}, fill={fill:.3%}, FPR={fpr:.6%}")
        if fpr > 0.01: log_warn(f"High FPR {fpr:.3%} — increase --bloom-mb")

    # ------------------------------------------------------------------
    # CELF GPU batch workspace (persistent "seen" scratch buffer + per-call
    # candidate batch capacity). Ranker's own GPU CELF path runs comfortably
    # at batch=1024 on an 8GB card; the old rulest default of 32 was a fixed
    # fallback never scaled by device/target size, which is why profiling
    # showed batch=32 regardless of VRAM headroom. Size both from the actual
    # target set so a 3060Ti (~8GB) lands near 1024 and a 1050Ti (~4GB)
    # lands near 512, while never exceeding a safe slice of free VRAM.
    seen_words_est = max(1, (int(target_count) + 31) // 32) if target_count else 4096
    if vgb >= 7.0:
        desired_celf_batch = 1024
    elif vgb >= 3.5:
        desired_celf_batch = 512
    else:
        desired_celf_batch = 256
    desired_celf_batch = max(32, min(desired_celf_batch, 4096))
    needed_scratch_bytes = desired_celf_batch * seen_words_est * 4
    # Don't let the CELF "seen" scratch buffer eat more than ~25% of the
    # VRAM budget we already estimated as usable.
    scratch_budget_bytes = max(32 * 1024 * 1024, int(fv * 0.25))
    celf_scratch_bytes = min(needed_scratch_bytes, scratch_budget_bytes)
    if celf_scratch_bytes < needed_scratch_bytes:
        # Budget is tighter than the target batch needs — shrink the batch
        # to fit rather than silently truncating it later per-call.
        desired_celf_batch = max(32, celf_scratch_bytes // max(4, seen_words_est * 4))
        celf_scratch_bytes = desired_celf_batch * seen_words_est * 4
    celf_scratch_mb = max(32, (celf_scratch_bytes + (1024 * 1024 - 1)) // (1024 * 1024))
    log_info(
        f"[CELF] GPU workspace sized from VRAM~{vgb:.1f}GB: batch={desired_celf_batch}, "
        f"seen-scratch={celf_scratch_mb}MB (targets={target_count:,})"
    )

    return {
        'BLOOM_FILTER_SIZE': bloom_bits,
        'BLOOM_NUM_SHARDS': num_shards,
        'BLOOM_SHARD_BITS': shard_bits,
        'BLOOM_SHARD_BYTES': shard_bits // 8,
        'WORDS_PER_BATCH': max(1000, int(BASE_WORDS_PER_BATCH * vram_scale)),
        'CHAINS_PER_BATCH': max(500, int(BASE_CHAINS_PER_BATCH * vram_scale)),
        'WORD_SUB_BATCH': max(5000, int(BASE_WORD_SUB_BATCH * vram_scale)),
        'MAX_SAFE_RESULTS_PER_BATCH': min(MAX_SAFE_RESULTS_CAP, max(5000, int(BASE_MAX_SAFE_RESULTS * vram_scale))),
        'MAX_CHAINS_TO_FIND': 2**31-1,
        'LOCAL_WORK_SIZE': lws,
        'OPTIMAL_GLOBAL_MULTIPLIER': mcu * OPTIMAL_GLOBAL_MULTIPLIER_BASE,
        'EST_COMBOS_PER_SEC': est,
        'TARGET_SECONDS': target_sec,
        'vram_scale': vram_scale,
        'free_vram': fv,
        'CELF_RULE_BATCH_SIZE': int(desired_celf_batch),
        'CELF_SEEN_SCRATCH_MB': int(celf_scratch_mb),
    }

# ----------------------------------------------------------------------
# GPU-compatible atomic rules generator
# ----------------------------------------------------------------------
