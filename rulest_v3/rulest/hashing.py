"""Shared 64-bit fingerprinting and vectorized open-addressing helpers.

Ported from rule_ranker's ``hashing.py``: the CELF target hash map used to be
built with a pure-Python ``for target in targets: ... for _probe in
range(table_size): ...`` loop (one Python-level iteration per key, plus a
nested Python-level probe loop on every collision). That does not scale past
a few hundred thousand targets. rule_ranker solves the same problem â€” turning
a batch of 64-bit keys into a linear-probed open-addressing table â€” entirely
in numpy: every key's first-choice slot is computed in one vectorized pass,
colliding keys are retried in lockstep (array-at-a-time, not key-at-a-time),
and the loop only runs once per *round* of collisions instead of once per
key. This module mirrors that approach for rulest.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

FNV1A64_OFFSET = 14695981039346656037
FNV1A64_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


def fast_fnv1a_hash_64(data: bytes) -> int:
    """Pure-Python FNV-1a-64 over a bytes-like object (single key)."""
    h = FNV1A64_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV1A64_PRIME) & UINT64_MASK
    return h


def fnv1a64_batch(byte_matrix, lengths) -> np.ndarray:
    """Vectorized FNV-1a-64 over a padded (n, max_len) uint8 byte matrix.

    ``lengths[i]`` is the true byte length of row ``i`` (padding past that
    point is ignored). Replaces the "one Python loop per string" pattern with
    one Python loop over *columns* (bounded by ``max_len``, not by ``n``),
    each iteration vectorized across every row via numpy.
    """
    byte_matrix = np.asarray(byte_matrix, dtype=np.uint8)
    lengths = np.asarray(lengths, dtype=np.int64)
    n, max_len = byte_matrix.shape
    h = np.full(n, np.uint64(FNV1A64_OFFSET), dtype=np.uint64)
    prime = np.uint64(FNV1A64_PRIME)
    with np.errstate(over="ignore"):
        for col in range(max_len):
            active = lengths > col
            if not np.any(active):
                break
            col_bytes = byte_matrix[:, col].astype(np.uint64)
            updated = (h ^ col_bytes) * prime
            h = np.where(active, updated, h)
    return h
# RULEST_HASH_STRINGS_CHUNKED_V8
def fnv1a64_records_chunked(records, batch_size=32768, progress=True, label="target hashes"):
    """Hash str/bytes records in bounded batches, preserving FNV-1a-64 bytes.

    The output is one uint64 per input record. Scratch memory is bounded by a
    single batch, not by (number_of_targets x max_word_length).
    """
    import time as _hash_time
    import numpy as _np
    if not hasattr(records, "__len__") or not hasattr(records, "__getitem__"):
        records = list(records)
    n = len(records)
    batch_size = max(1, min(1_000_000, int(batch_size)))
    result = _np.empty(n, dtype=_np.uint64)
    started = _hash_time.monotonic()
    last_report = started
    offset = _np.uint64(FNV1A64_OFFSET)
    prime = _np.uint64(FNV1A64_PRIME)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        part = records[start:end]
        encoded = [x if isinstance(x, bytes) else str(x).encode("latin-1", errors="strict") for x in part]
        lengths = _np.fromiter((len(x) for x in encoded), dtype=_np.int64, count=len(encoded))
        offsets = _np.empty(len(encoded), dtype=_np.int64)
        if len(encoded):
            offsets[0] = 0
            if len(encoded) > 1:
                _np.cumsum(lengths[:-1], out=offsets[1:])
        raw = _np.frombuffer(b"".join(encoded), dtype=_np.uint8)
        h = _np.full(len(encoded), offset, dtype=_np.uint64)
        max_len = int(lengths.max()) if len(encoded) else 0
        for col in range(max_len):
            active = _np.flatnonzero(lengths > col)
            if active.size == 0:
                break
            byte_values = raw[offsets[active] + col].astype(_np.uint64, copy=False)
            with _np.errstate(over="ignore"):
                h[active] = (h[active] ^ byte_values) * prime
        result[start:end] = h
        now = _hash_time.monotonic()
        if progress and (now - last_report >= 0.75 or end == n):
            elapsed = max(now - started, 1e-9)
            print(f"[HASHMAP] {label}: {end:,}/{n:,} ({100.0*end/max(n,1):.1f}%) | "
                  f"{end/elapsed:,.0f} hashes/s | {elapsed:.1f}s", flush=True)
            last_report = now
    if progress and n == 0:
        print(f"[HASHMAP] {label}: 0/0 (empty)", flush=True)
    return result



def _slot_for_key(keys: np.ndarray, mask: int) -> np.ndarray:
    """SplitMix64 finalizer mix, matching the OpenCL ``mix64`` slot mixer."""
    keys_u = np.asarray(keys, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = keys_u + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(mask)).astype(np.int64)


def table_size_for_count(count: int, load_factor: float = 0.50) -> int:
    """Return a power-of-two table size at or below *load_factor*."""
    if count < 1:
        return 2
    target = max(2, int(np.ceil(count / load_factor)))
    size = 1
    while size < target:
        size <<= 1
    return size


def build_open_addressing_table_uint64(keys, table_size: Optional[int] = None):
    """Build a uint64 linear-probing table without Python per-key loops.

    Returns ``(hash_table, states)`` where ``states`` is uint32 and uses:
    0 = empty, 2 = occupied/ready. Duplicate keys are collapsed before
    insertion. See ``build_indexed_open_addressing_table`` for a variant that
    also tracks which original row landed in each slot (needed when slots
    must be mapped back to e.g. a target id).
    """
    arr = np.unique(np.asarray(keys, dtype=np.uint64))
    if table_size is None:
        table_size = table_size_for_count(int(arr.size))
    table_size = int(table_size)
    if table_size < table_size_for_count(int(arr.size)):
        raise ValueError("table_size is too small for the requested load factor")
    mask = table_size - 1

    table = np.zeros(table_size, dtype=np.uint64)
    states = np.zeros(table_size, dtype=np.uint32)
    if arr.size == 0:
        return table, states

    pending_keys = arr
    pending_slots = _slot_for_key(pending_keys, mask)
    while pending_keys.size:
        _, first = np.unique(pending_slots, return_index=True)
        winner = np.zeros(pending_keys.size, dtype=bool)
        winner[first] = True

        cand_slots = pending_slots[winner]
        cand_keys = pending_keys[winner]
        free = states[cand_slots] == 0
        if np.any(free):
            table[cand_slots[free]] = cand_keys[free]
            states[cand_slots[free]] = 2

        retry = ~winner
        winner_pos = np.flatnonzero(winner)
        blocked = ~free
        if np.any(blocked):
            retry[winner_pos[blocked]] = True

        pending_keys = pending_keys[retry]
        pending_slots = (pending_slots[retry] + 1) & mask

    return table, states


def build_indexed_open_addressing_table(
    keys, ids=None, table_size: Optional[int] = None, progress=True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Build an indexed uint64 open-addressing table with bounded insertion scratch.

    slot_ids == -1 is the empty marker, avoiding a table-sized bool occupancy
    array. Collision resolution operates on bounded input chunks. The GPU-facing
    return format remains compatible with the existing CELF lookup kernels.
    """
    import os as _os
    import time as _time
    keys_u = np.asarray(keys, dtype=np.uint64)
    if keys_u.ndim != 1:
        raise ValueError("keys must be one-dimensional")
    n = int(keys_u.size)
    i32_max = int(np.iinfo(np.int32).max)
    if n > i32_max:
        raise ValueError("target IDs use int32; target count exceeds capacity")
    if ids is None:
        ids_u = np.arange(n, dtype=np.int32)
    else:
        source_ids = np.asarray(ids)
        if source_ids.ndim != 1 or source_ids.size != n:
            raise ValueError("ids must be one-dimensional and the same length as keys")
        if source_ids.size and (int(source_ids.min()) < 0 or int(source_ids.max()) > i32_max):
            raise ValueError("target IDs must be non-negative int32 values")
        ids_u = source_ids.astype(np.int32, copy=False)

    if table_size is None:
        table_size = table_size_for_count(n)
    table_size = int(table_size)
    if table_size < 2 or (table_size & (table_size - 1)) != 0:
        raise ValueError("table_size must be a power of two >= 2")
    if n and table_size < table_size_for_count(n):
        raise ValueError("table_size is too small for the requested load factor")
    mask = table_size - 1
    bitmap_words = (table_size + 31) // 32
    hash_table = np.zeros(table_size, dtype=np.uint64)
    slot_ids = np.full(table_size, -1, dtype=np.int32)
    occupied_bitmap = np.zeros(bitmap_words, dtype=np.uint32)
    if n == 0:
        return hash_table, slot_ids, occupied_bitmap, table_size, 0

    try:
        chunk_size = int(_os.environ.get("RULEST_HASHMAP_BUILD_BATCH", "250000"))
    except ValueError:
        chunk_size = 250000
    chunk_size = max(4096, min(2_000_000, chunk_size))
    started = _time.monotonic()
    last_report = started
    inserted = 0
    if progress:
        print(f"[HASHMAP] indexed build: targets={n:,}, slots={table_size:,}, "
              f"table arrays={(hash_table.nbytes + slot_ids.nbytes + occupied_bitmap.nbytes)/(1024**2):.1f} MiB, "
              f"chunk={chunk_size:,}", flush=True)

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(n, chunk_start + chunk_size)
        pending_keys = keys_u[chunk_start:chunk_end]
        pending_ids = ids_u[chunk_start:chunk_end]
        pending_slots = _slot_for_key(pending_keys, mask)
        inserted_chunk = 0
        rounds = 0
        while pending_keys.size:
            rounds += 1
            if rounds > table_size:
                raise RuntimeError("hash table insertion exceeded table-size probe limit")
            _, first = np.unique(pending_slots, return_index=True)
            winner = np.zeros(pending_slots.size, dtype=bool)
            winner[first] = True
            cand_slots = pending_slots[winner]
            cand_keys = pending_keys[winner]
            cand_ids = pending_ids[winner]
            free = slot_ids[cand_slots] < 0
            if np.any(free):
                placed = cand_slots[free]
                slot_ids[placed] = cand_ids[free]
                hash_table[placed] = cand_keys[free]
                count = int(np.count_nonzero(free))
                inserted += count
                inserted_chunk += count
            retry = ~winner
            winner_positions = np.flatnonzero(winner)
            blocked = ~free
            if np.any(blocked):
                retry[winner_positions[blocked]] = True
            pending_keys = pending_keys[retry]
            pending_ids = pending_ids[retry]
            pending_slots = (pending_slots[retry] + 1) & mask
            now = _time.monotonic()
            if progress and now - last_report >= 1.0:
                elapsed = max(now - started, 1e-9)
                print(f"[HASHMAP] inserted {inserted:,}/{n:,} targets "
                      f"({100.0*inserted/n:.1f}%) | {inserted/elapsed:,.0f} targets/s | "
                      f"chunk {chunk_start:,}:{chunk_end:,}, collisions round {rounds} | {elapsed:.1f}s", flush=True)
                last_report = now
        now = _time.monotonic()
        if progress and (now - last_report >= 0.75 or chunk_end == n):
            elapsed = max(now - started, 1e-9)
            print(f"[HASHMAP] inserted {inserted:,}/{n:,} targets "
                  f"({100.0*inserted/n:.1f}%) | {inserted/elapsed:,.0f} targets/s | {elapsed:.1f}s", flush=True)
            last_report = now

    if inserted != n:
        raise RuntimeError(f"indexed hashmap inserted {inserted:,} keys for {n:,} rows")

    # Pack bitmap in slices. Do not allocate a persistent table-sized bool array.
    slots_per_pack = 8 * 1024 * 1024
    for slot_start in range(0, table_size, slots_per_pack):
        slot_end = min(table_size, slot_start + slots_per_pack)
        packed = np.packbits(slot_ids[slot_start:slot_end] >= 0, bitorder="little")
        if packed.size % 4:
            packed = np.concatenate((packed, np.zeros((-packed.size) % 4, dtype=np.uint8)))
        word_start = slot_start // 32
        word_count = (slot_end - slot_start + 31) // 32
        occupied_bitmap[word_start:word_start + word_count] = packed.view(np.uint32)[:word_count]
    elapsed = max(_time.monotonic() - started, 1e-9)
    if progress:
        print(f"[HASHMAP] ready: {inserted:,} slots in {elapsed:.1f}s | {inserted/elapsed:,.0f} targets/s", flush=True)
    return hash_table, slot_ids, occupied_bitmap, table_size, inserted

# RULEST_CELF_V11_PACKED_FNV64: hash the packed target payload without re-encoding
# millions of Python strings or joining another temporary bytes buffer.
def fnv1a64_packed_records_chunked(packed_bytes, target_offsets, target_lengths,
                                   target_ids, batch_size=None, progress=True,
                                   label="target hashes"):
    """Return FNV-1a-64 hashes in target_ids order from packed variable-length bytes.

    ``packed_bytes`` is a contiguous uint8 payload, ``target_offsets`` and
    ``target_lengths`` index records in that payload, and ``target_ids`` gives
    the records to hash. Scratch is O(batch_size), never O(N * max_record_len).
    This intentionally matches the scalar/batched FNV-1a-64 byte order used by
    the existing fnv1a64_records_chunked() function and the OpenCL kernel.
    """
    import os as _os
    import time as _time
    import numpy as _np

    packed = _np.asarray(packed_bytes, dtype=_np.uint8)
    offsets = _np.asarray(target_offsets, dtype=_np.uint64)
    lengths = _np.asarray(target_lengths, dtype=_np.int32)
    ids = _np.asarray(target_ids, dtype=_np.int32)
    if offsets.ndim != 1 or lengths.ndim != 1 or ids.ndim != 1:
        raise ValueError("packed FNV inputs must be one-dimensional")
    if offsets.size != lengths.size:
        raise ValueError("target_offsets and target_lengths must have equal length")
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= offsets.size):
        raise IndexError("target_ids contains an out-of-range target index")
    if batch_size is None:
        batch_size = int(_os.environ.get("RULEST_HASH_BATCH_SIZE", "32768"))
    batch_size = max(1, int(batch_size))
    total = int(ids.size)
    out = _np.empty(total, dtype=_np.uint64)
    if total == 0:
        return out

    offset_basis = _np.uint64(14695981039346656037)
    prime = _np.uint64(1099511628211)
    started = _time.perf_counter()
    last_reported = 0
    report_every = max(batch_size, 1_000_000)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = ids[start:end]
        lens = lengths[batch_ids]
        offs = offsets[batch_ids]
        if _np.any(lens < 0):
            raise ValueError("negative target byte length")
        if _np.any(offs > packed.size) or _np.any(lens > (packed.size - _np.minimum(offs, packed.size))):
            raise ValueError("target offset/length exceeds packed byte payload")

        n = end - start
        h = _np.full(n, offset_basis, dtype=_np.uint64)
        max_len = int(lens.max()) if n else 0
        for col in range(max_len):
            rows = _np.flatnonzero(lens > col)
            if rows.size == 0:
                break
            current = h[rows]
            current ^= packed[offs[rows] + _np.uint64(col)].astype(_np.uint64, copy=False)
            current *= prime
            h[rows] = current
        out[start:end] = h
        done = end
        if progress and (done == total or done - last_reported >= report_every):
            elapsed = max(_time.perf_counter() - started, 1e-9)
            pct = 100.0 * done / total
            print(f"[HASHMAP] {label}: {done:,}/{total:,} ({pct:.1f}%) | "
                  f"{done / elapsed:,.0f} hashes/s | {elapsed:.1f}s", flush=True)
            last_reported = done
    return out

