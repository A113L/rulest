"""In-RAM, shared-memory wordlist loading.

Wordlists are normalized and deduplicated once while loading.  The unique
Latin-1 bytes and uint64 byte offsets then live in OS shared memory.  The
parent and spawned workers attach to the same segments; no temporary wordlist
files, per-worker copies of millions of strings, or disk-backed random reads
are involved after initial loading.
"""
from __future__ import annotations

import os
import sys
import struct
from collections.abc import Sequence
from multiprocessing.shared_memory import SharedMemory

from .common import log_info, log_error, bold, cyan, MAX_WORD_LEN
from .tokenstrip_hashset import FlatHashSet64


class WordListView(Sequence):
    """Lazy range/stride view over a :class:`SharedWordList`."""
    __slots__ = ("_words", "_indices")

    def __init__(self, words, indices):
        self._words = words
        self._indices = indices if isinstance(indices, range) else range(*indices)

    def __len__(self):
        return len(self._indices)

    def __bool__(self):
        return len(self._indices) != 0

    def __iter__(self):
        get = self._words.__getitem__
        for index in self._indices:
            yield get(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return WordListView(self._words, self._indices[index])
        return self._words[self._indices[index]]

    def numpy_flatten(self):
        """Return zero-copy bytes and compact GPU offset/length arrays."""
        if self._indices.step == 1:
            start = self._indices.start
            stop = self._indices.stop
            return self._words.numpy_flatten_range(start, stop)
        return self._words._flatten_sequence(self)


class SharedWordList(Sequence):
    """Unique words stored as Latin-1 bytes + uint64 offsets in Shared Memory.

    Pickling transfers only shared-memory names and dimensions.  This is safe
    for Windows ``spawn``: child processes attach to the same physical data.
    ``membership_index`` is optional and is retained only on the owner object;
    Stage 0 workers attach to it separately through its small descriptor.
    """
    __slots__ = (
        "_data_shm", "_offsets_shm", "_offsets", "_count", "_data_size",
        "_owner", "_closed", "membership_index", "_rulest_unique_words",
    )

    def __init__(self, data_shm, offsets_shm, count, data_size,
                 membership_index=None, *, owner=True):
        self._data_shm = data_shm
        self._offsets_shm = offsets_shm
        self._offsets = offsets_shm.buf.cast("Q")
        self._count = int(count)
        self._data_size = int(data_size)
        self._owner = bool(owner)
        self._closed = False
        self.membership_index = membership_index
        self._rulest_unique_words = True
        if len(self._offsets) < self._count + 1:
            self.close()
            raise ValueError("Shared word offsets are shorter than the word count")

    def __len__(self):
        return self._count

    def __bool__(self):
        return self._count != 0

    def __eq__(self, other):
        if self is other:
            return True
        try:
            if len(self) != len(other):
                return False
            return all(a == b for a, b in zip(self, other))
        except (TypeError, AttributeError):
            return False

    def __iter__(self):
        get = self.__getitem__
        for index in range(self._count):
            yield get(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return WordListView(self, range(*index.indices(self._count)))
        if self._closed:
            raise ValueError("wordlist is closed")
        index = int(index)
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("wordlist index out of range")
        start = self._offsets[index]
        end = self._offsets[index + 1]
        return self._data_shm.buf[start:end].tobytes().decode("latin-1")

    @property
    def word_encoding(self):
        # Wordlists are normalized with the existing latin-1 loader contract.
        return "latin-1"

    def shared_descriptor(self):
        """Return the immutable descriptor needed to attach in another process."""
        return (
            self._data_shm.name,
            self._offsets_shm.name,
            self._count,
            self._data_size,
        )

    def numpy_flatten(self):
        return self.numpy_flatten_range(0, self._count)

    def numpy_flatten_range(self, start, stop):
        """Expose packed bytes directly and build only the GPU int32 indices."""
        import numpy as np

        start = max(0, min(self._count, int(start)))
        stop = max(start, min(self._count, int(stop)))
        offsets64 = self._offsets
        first_byte = int(offsets64[start])
        end_byte = int(offsets64[stop])
        byte_count = end_byte - first_byte
        if end_byte > 0x7FFFFFFF:
            raise OverflowError(
                "GPU word offsets use int32; this wordlist exceeds the 2 GiB limit"
            )
        n = stop - start
        flat = np.frombuffer(
            self._data_shm.buf, dtype=np.uint8, count=byte_count, offset=first_byte
        )
        word_offsets = np.empty(n, dtype=np.int32)
        word_lengths = np.empty(n, dtype=np.int32)
        if n:
            source_offsets = np.frombuffer(
                self._offsets_shm.buf, dtype=np.uint64,
                count=n + 1, offset=start * 8,
            )
            np.copyto(word_offsets, source_offsets[:-1], casting="unsafe")
            word_offsets -= first_byte
            np.subtract(
                source_offsets[1:], source_offsets[:-1],
                out=word_lengths, casting="unsafe",
            )
        return flat, word_offsets, word_lengths

    @staticmethod
    def _flatten_sequence(words):
        """Low-peak-memory fallback for non-contiguous lazy views."""
        import numpy as np

        n = len(words)
        lengths = np.empty(n, dtype=np.int32)
        total = 0
        for i, word in enumerate(words):
            size = len(word.encode("latin-1"))
            lengths[i] = size
            total += size
        if total > 0x7FFFFFFF:
            raise OverflowError("GPU word offsets use int32; this view exceeds 2 GiB")
        flat_bytes = bytearray(total)
        offsets = np.empty(n, dtype=np.int32)
        position = 0
        for i, word in enumerate(words):
            raw = word.encode("latin-1")
            offsets[i] = position
            flat_bytes[position:position + len(raw)] = raw
            position += len(raw)
        return np.frombuffer(flat_bytes, dtype=np.uint8), offsets, lengths

    def __getstate__(self):
        return self.shared_descriptor()

    def __setstate__(self, state):
        data_name, offsets_name, count, data_size = state
        self._data_shm = SharedMemory(name=data_name, create=False)
        try:
            self._offsets_shm = SharedMemory(name=offsets_name, create=False)
        except Exception:
            self._data_shm.close()
            raise
        self._offsets = self._offsets_shm.buf.cast("Q")
        self._count = int(count)
        self._data_size = int(data_size)
        self._owner = False
        self._closed = False
        self.membership_index = None
        self._rulest_unique_words = True

    def __repr__(self):
        return f"SharedWordList({self._count:,} words, {self._data_size:,} bytes in RAM)"

    def close(self):
        if self._closed:
            return
        offsets = getattr(self, "_offsets", None)
        self._offsets = None
        if offsets is not None:
            try:
                offsets.release()
            except (AttributeError, BufferError):
                pass
        pending_exports = False
        for attr in ("_offsets_shm", "_data_shm"):
            shared = getattr(self, attr, None)
            if shared is None:
                continue
            try:
                shared.close()
            except BufferError:
                # A live NumPy view can keep the exported data buffer alive.
                # Keep the handle so a later close() can finish cleanup.
                pending_exports = True
            else:
                setattr(self, attr, None)
            if self._owner:
                try:
                    shared.unlink()
                except FileNotFoundError:
                    pass
        membership = getattr(self, "membership_index", None)
        self.membership_index = None
        if membership is not None and self._owner:
            membership.close(unlink=True)
        self._closed = not pending_exports

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _normalize_line(line):
    word = line.decode("latin-1", errors="ignore").strip()
    if word and len(word) <= MAX_WORD_LEN:
        return word
    return None


def _count_valid_lines(filename):
    """First sequential pass: size the shared uint64 offset index/hash table."""
    count = 0
    with open(filename, "rb") as stream:
        for line in stream:
            if _normalize_line(line) is not None:
                count += 1
    return count


def load_wordlist(filename, *, retain_membership_index=False):
    """Load a newline-delimited wordlist into shared RAM.

    The input is scanned once to size allocations, then once to normalize,
    deduplicate, and copy the unique bytes into Shared Memory.  All subsequent
    indexing (including Stage 0 and GPU preparation) reads RAM only.

    If ``retain_membership_index`` is true, the compact shared 63-bit hash set
    is kept for Stage 0.  This avoids building a Python ``set[str]`` or sending
    that set to each Windows worker.
    """
    filename = os.fspath(filename)
    try:
        candidate_count = _count_valid_lines(filename)
        source_size = os.path.getsize(filename)
    except FileNotFoundError:
        log_error(f"File not found: {filename}")
        sys.exit(1)

    data_shm = offsets_shm = membership = None
    offsets_view = None
    try:
        # The compacted bytes can never exceed the original file size.
        data_shm = SharedMemory(create=True, size=max(1, source_size))
        offsets_shm = SharedMemory(
            create=True, size=max(8, (candidate_count + 1) * 8)
        )
        offsets_view = offsets_shm.buf.cast("Q")
        offsets_view[0] = 0
        membership = FlatHashSet64(
            (), expected_count=candidate_count, shared=True
        )

        data_size = 0
        unique_count = 0
        with open(filename, "rb") as stream:
            for line in stream:
                word = _normalize_line(line)
                if word is None or not membership.add(word):
                    continue
                raw = word.encode("latin-1")
                next_size = data_size + len(raw)
                data_shm.buf[data_size:next_size] = raw
                data_size = next_size
                unique_count += 1
                offsets_view[unique_count] = data_size

        offsets_view.release()
        offsets_view = None
        if not retain_membership_index:
            membership.close(unlink=True)
            membership = None

        result = SharedWordList(
            data_shm,
            offsets_shm,
            unique_count,
            data_size,
            membership_index=membership,
            owner=True,
        )
        data_shm = offsets_shm = membership = None
        data_reserved = result._data_shm.size
        offsets_reserved = result._offsets_shm.size
        membership_reserved = (
            result.membership_index.size_bytes
            if result.membership_index is not None else 0
        )
        total_reserved = data_reserved + offsets_reserved + membership_reserved
        log_info(
            f"[LOAD] {bold(os.path.basename(filename))}: "
            f"{bold(cyan(f'{len(result):,}'))} unique words "
            f"| shared RAM reserved: data={data_reserved / 1048576:.1f} MiB, "
            f"offsets={offsets_reserved / 1048576:.1f} MiB, "
            f"membership={membership_reserved / 1048576:.1f} MiB, "
            f"total={total_reserved / 1048576:.1f} MiB"
        )
        return result
    except Exception:
        if offsets_view is not None:
            try:
                offsets_view.release()
            except Exception:
                pass
        for shared in (offsets_shm, data_shm):
            if shared is not None:
                try:
                    shared.close()
                finally:
                    try:
                        shared.unlink()
                    except FileNotFoundError:
                        pass
        if membership is not None:
            try:
                membership.close(unlink=True)
            except Exception:
                pass
        raise
