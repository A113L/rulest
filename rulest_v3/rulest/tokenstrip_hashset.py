"""Compact 64-bit hash membership table used by Stage 0 and wordlist loading.

The table stores one 63-bit FNV-1a fingerprint per occupied slot.  It is an
exact-in-practice membership index (the same collision model used by the
original compact Stage-0 implementation), but unlike ``set[str]`` it uses
roughly 9--12 bytes per input word and can be backed by shared memory so
Windows ``spawn`` workers attach to one physical table instead of receiving
one pickled copy each.
"""
from array import array
from multiprocessing.shared_memory import SharedMemory

_MASK63 = 0x7FFFFFFFFFFFFFFF
_FNV_OFFSET = 1469598103934665603
_FNV_PRIME = 1099511628211


def _fnv1a_63(s):
    h = _FNV_OFFSET
    for ch in s:
        h ^= ord(ch)
        h = (h * _FNV_PRIME) & _MASK63
    return h


class FlatHashSet64:
    __slots__ = ("table", "mask", "size", "capacity", "_shm", "_owner", "_closed")

    def __init__(self, words=(), load=0.70, *, expected_count=None, shared=False):
        if expected_count is None:
            expected_count = len(words)
        n = max(0, int(expected_count))
        cap = 16
        while cap < max(1, int(n / max(0.10, min(float(load), 0.90)))):
            cap <<= 1

        self.capacity = cap
        self.mask = cap - 1
        self.size = 0
        self._owner = bool(shared)
        self._closed = False
        self._shm = SharedMemory(create=True, size=cap * 8) if shared else None
        if shared:
            self.table = self._shm.buf.cast("Q")
            # Newly-created shared memory is zero-filled by the OS.
        else:
            self.table = array('Q', bytes(8 * cap))
        for word in words:
            self.add(word)

    @classmethod
    def attach(cls, name, capacity, size):
        instance = cls.__new__(cls)
        instance.capacity = int(capacity)
        instance.mask = instance.capacity - 1
        instance.size = int(size)
        instance._owner = False
        instance._closed = False
        instance._shm = SharedMemory(name=name, create=False)
        instance.table = instance._shm.buf.cast("Q")
        if len(instance.table) != instance.capacity:
            instance.close()
            raise ValueError("Shared hash table has an unexpected size")
        return instance

    @property
    def shared_name(self):
        return self._shm.name if self._shm is not None else None

    def descriptor(self):
        if self._shm is None:
            raise RuntimeError("Hash table is not backed by shared memory")
        return (self._shm.name, self.capacity, self.size)

    def add(self, s):
        h = _fnv1a_63(s)
        slot = h & self.mask
        table = self.table
        want = h + 1  # zero marks an empty slot
        while True:
            value = table[slot]
            if value == 0:
                table[slot] = want
                self.size += 1
                return True
            if value == want:
                return False
            slot = (slot + 1) & self.mask

    def __contains__(self, s):
        h = _fnv1a_63(s)
        slot = h & self.mask
        table = self.table
        want = h + 1
        while True:
            value = table[slot]
            if value == 0:
                return False
            if value == want:
                return True
            slot = (slot + 1) & self.mask

    def __len__(self):
        return self.size

    @property
    def size_bytes(self):
        return self.capacity * 8

    def close(self, unlink=False):
        if self._closed:
            return
        self._closed = True
        table = self.table
        self.table = None
        try:
            table.release()
        except (AttributeError, BufferError):
            pass
        if self._shm is not None:
            try:
                self._shm.close()
            finally:
                if unlink or self._owner:
                    try:
                        self._shm.unlink()
                    except FileNotFoundError:
                        pass
            self._shm = None

    def __del__(self):
        try:
            self.close(unlink=False)
        except Exception:
            pass
