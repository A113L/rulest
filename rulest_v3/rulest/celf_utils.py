"""Small, dependency-light helpers for CELF target bitmap handling."""
from __future__ import annotations


def target_ids_from_seen_words(words, target_count: int) -> set[int]:
    """Decode a little-endian uint32 target bitmap into a set of target IDs.

    Bit ``b`` in word ``i`` represents target ID ``i * 32 + b``. The OpenCL
    CELF scorer uses the same layout with ``atomic_or``. Values outside the
    logical target range (padding bits in the final word) are ignored.
    """
    target_count = max(0, int(target_count))
    result: set[int] = set()
    for word_index, raw_word in enumerate(words):
        bits = int(raw_word) & 0xFFFFFFFF
        while bits:
            least_bit = bits & -bits
            bit_index = least_bit.bit_length() - 1
            target_id = word_index * 32 + bit_index
            if target_id < target_count:
                result.add(target_id)
            bits ^= least_bit
    return result
