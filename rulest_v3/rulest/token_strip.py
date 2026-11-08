"""
Stage 0 (--token-strip): cheap CPU pre-pass that tries to explain each
target word as a base-wordlist word plus a short, exact-match transform
(case toggling, leet substitution, reversal, duplication, edge deletion,
insertion, prefix/suffix boundary stripping) before any GPU work happens.
Runs across a multiprocessing pool for throughput on large target lists.

Membership structure for the base wordlist can be either a plain set[str]
(exact, higher RAM) or a Bloom filter (compact, false-positive safe
because every hit is still verified by py_apply_chain).
"""
import os
import atexit
import time
import shutil
import itertools
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from array import array
from collections import defaultdict

from tqdm import tqdm

from . import worker_state
from .common import (
    log_info, log_warn, bold, green, cyan, yellow, dim, _fmt_speed,
    safe_worker_count, HashcatRuleValidator, py_apply_chain,
    MAX_WORD_LEN, MAX_HASHCAT_CHAIN,
    _TOKEN_STRIP_LEET_BY_CHAR,
    TOKEN_STRIP_LEET_CHARS, TOKEN_STRIP_BOUNDARY, TOKEN_STRIP_ALPHA_BOUNDARY,
    _kb,
)
from .hashing import fast_fnv1a_hash_64
from .tokenstrip_hashset import FlatHashSet64

# ---------------------------------------------------------------------------
# Compact Bloom filter for base-wordlist membership (Stage 0 only)
# ---------------------------------------------------------------------------

class BloomFilter:
    """Compact Bloom filter optionally backed by shared memory."""
    __slots__ = ("n_bits", "n_hashes", "bits", "mask", "count", "_shm", "_owner")

    def __init__(self, n_bits: int, n_hashes: int = 5, *, shared=False):
        p = 64
        while p < max(64, int(n_bits)):
            p <<= 1
        self.n_bits = p
        self.mask = p - 1
        self.n_hashes = max(1, min(int(n_hashes), 16))
        self._owner = bool(shared)
        self._shm = SharedMemory(create=True, size=p // 8) if shared else None
        self.bits = self._shm.buf if shared else bytearray(p // 8)
        self.count = 0

    @classmethod
    def from_size_mb(cls, mb: float, n_hashes: int = 5, *, shared=False):
        n_bits = max(64, int(mb * 1024 * 1024 * 8))
        return cls(n_bits, n_hashes=n_hashes, shared=shared)

    @classmethod
    def for_capacity(cls, n_items: int, bits_per_item: float = 8.0,
                     n_hashes: int = 5, *, shared=False):
        return cls(max(64, int(n_items * bits_per_item)),
                   n_hashes=n_hashes, shared=shared)

    @classmethod
    def attach(cls, name, n_bits, n_hashes, count):
        instance = cls.__new__(cls)
        instance.n_bits = int(n_bits)
        instance.mask = instance.n_bits - 1
        instance.n_hashes = int(n_hashes)
        instance.count = int(count)
        instance._owner = False
        instance._shm = SharedMemory(name=name, create=False)
        instance.bits = instance._shm.buf
        return instance

    def descriptor(self):
        if self._shm is None:
            raise RuntimeError("Bloom filter is not backed by shared memory")
        return ("bloom", self._shm.name, self.n_bits, self.n_hashes, self.count)

    def _hashes(self, key: str):
        data = key.encode('utf-8', errors='ignore')
        h1 = fast_fnv1a_hash_64(data)
        h2 = h1 ^ (h1 >> 30)
        h2 = (h2 * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
        h2 ^= (h2 >> 27)
        h2 = (h2 * 0x94d049bb133111eb) & ((1 << 64) - 1)
        h2 ^= (h2 >> 31)
        if h2 == 0:
            h2 = 0x9e3779b97f4a7c15
        for i in range(self.n_hashes):
            yield (h1 + i * h2) & self.mask

    def add(self, key: str) -> None:
        for h in self._hashes(key):
            self.bits[h >> 3] |= (1 << (h & 7))
        self.count += 1

    def __contains__(self, key: str) -> bool:
        for h in self._hashes(key):
            if not (self.bits[h >> 3] & (1 << (h & 7))):
                return False
        return True

    def __len__(self):
        return self.count

    @property
    def size_bytes(self) -> int:
        return self.n_bits // 8

    def estimated_fpr(self) -> float:
        if self.count <= 0 or self.n_bits <= 0:
            return 0.0
        kn_over_m = self.n_hashes * self.count / self.n_bits
        return (1.0 - (2.718281828459045 ** (-kn_over_m))) ** self.n_hashes

    def close(self, unlink=False):
        shm = self._shm
        if shm is None:
            return
        self._shm = None
        bits = self.bits
        self.bits = None
        try:
            bits.release()
        except (AttributeError, BufferError):
            pass
        try:
            shm.close()
        except BufferError:
            pass
        if unlink or self._owner:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def _build_membership(base_words, bloom_mb=0):
    """Build a compact, shared-memory membership index without ``set[str]``.

    The exact path stores one 63-bit fingerprint per open-addressed slot.  Its
    collision behavior is the same compact fingerprint model already used by
    the loader; memory use is roughly 9--12 bytes per input word, rather than
    hundreds of bytes per Python string plus the set table.
    """
    n = len(base_words)
    if bloom_mb == 0:
        membership = FlatHashSet64(base_words, expected_count=n, shared=True)
        return membership, f"shared exact hash table ({n:,} words, {membership.size_bytes / 1024 / 1024:.1f} MiB)"

    if bloom_mb < 0:
        if n < 400_000:
            membership = FlatHashSet64(base_words, expected_count=n, shared=True)
            return membership, f"shared exact hash table (auto, {n:,} words, {membership.size_bytes / 1024 / 1024:.1f} MiB)"
        bf = BloomFilter.for_capacity(n, bits_per_item=8.0, n_hashes=5, shared=True)
        mode = f"shared Bloom auto (~8 bits/item, {bf.size_bytes / 1024 / 1024:.1f} MiB)"
    else:
        bf = BloomFilter.from_size_mb(float(bloom_mb), n_hashes=5, shared=True)
        mode = f"shared Bloom {bloom_mb} MB ({bf.n_bits:,} bits, k={bf.n_hashes})"

    for word in base_words:
        if word:
            bf.add(word)
    return bf, mode + f"  inserted={bf.count:,}  est.FP≈{bf.estimated_fpr()*100:.3f}%"


def _hashcat_title_case(s):
    """Mirror hashcat's 'E' rule: capitalize the first letter of each space-separated word."""
    res = list(s); cap = True
    for i,c in enumerate(res):
        if cap and 'a'<=c<='z': res[i]=c.upper(); cap=False
        elif not cap and 'A'<=c<='Z': res[i]=c.lower()
        if c == ' ': cap=True
    return ''.join(res)

def _infer_case_rules(cased_stem):
    lower = cased_stem.lower()
    if cased_stem == lower: return [[]]
    cand = []
    if cased_stem == lower.upper(): cand.append(['u'])
    if len(cased_stem)>=1 and cased_stem[0].isupper() and cased_stem[1:].islower(): cand.append(['c'])
    if len(cased_stem)>=1 and cased_stem[0].islower() and cased_stem[1:].isupper(): cand.append(['C'])
    toggled = ''.join(c.lower() if c.isupper() else c.upper() for c in cased_stem)
    if cased_stem == toggled: cand.append(['t'])
    if cased_stem == _hashcat_title_case(lower): cand.append(['E'])
    if not cand:
        up = [i for i,c in enumerate(cased_stem) if c != lower[i]]
        if up and all(p<=9 for p in up): cand.append([f'T{p}' for p in up])
    return cand

def _leet_decode_variants(mid, max_amb=3):
    for ch in mid:
        if not ch.isalpha() and ch not in TOKEN_STRIP_LEET_CHARS: return
    leet = [(i,ch,_TOKEN_STRIP_LEET_BY_CHAR[ch]) for i,ch in enumerate(mid) if ch in TOKEN_STRIP_LEET_CHARS]
    if not leet:
        yield (mid, frozenset()); return
    if sum(1 for _,_,opts in leet if len(opts)>1) > max_amb: return
    for combo in itertools.product(*[opts for _,_,opts in leet]):
        dec = list(mid); rules = set()
        for (pos,_,_), (b,r) in zip(leet,combo):
            dec[pos]=b; rules.add(r)
        dstr = ''.join(dec)
        if all(ch.isalpha() for ch in dstr): yield (dstr, frozenset(rules))

def _decode_middle(mid, max_amb=3):
    for leet_dec, leet_rules in _leet_decode_variants(mid, max_amb):
        stem = leet_dec.lower()
        case_cand = _infer_case_rules(leet_dec)
        yield (stem, leet_rules, case_cand if case_cand else [[]])

def _rule_chain_orderings(case_ops, leet_ops, prepend_ops, append_ops, lead=None):
    lead = lead or []
    seen = set()
    def _add(ops):
        full = lead + ops
        key = tuple(full)
        if key not in seen:
            seen.add(key); yield full
    yield from _add(case_ops+leet_ops+prepend_ops+append_ops)
    if case_ops and leet_ops: yield from _add(leet_ops+case_ops+prepend_ops+append_ops)
    if prepend_ops and (case_ops or leet_ops): yield from _add(prepend_ops+case_ops+leet_ops+append_ops)

def _boundary_scan(word, bound, max_pre, max_suf, min_len):
    L = len(word)
    for p in range(min(max_pre+1, L+1)):
        if p>0 and word[p-1] not in bound: break
        for s in range(min(max_suf+1, L-p+1)):
            if s==0: mid = word[p:]; suf=''
            else: mid = word[p:L-s]; suf = word[L-s:]
            if s>0 and word[L-s] not in bound: break
            if len(mid) >= min_len:
                yield (word[:p], mid, suf)

def _chains_from_middle(mid, pre, suf, base_set, max_depth, min_len, max_leet_amb, lead=None):
    target = pre+mid+suf
    lead = lead or []
    lead_depth = len(lead)
    prepend = [f'^{c}' for c in reversed(pre)]
    append = [f'${c}' for c in suf]
    found = set()
    for stem, leet_rules, case_cand in _decode_middle(mid, max_leet_amb):
        if len(stem) < min_len or stem not in base_set: continue
        leet_ops = sorted(leet_rules)
        for case_ops in case_cand:
            if lead_depth + len(case_ops)+len(leet_ops)+len(prepend)+len(append) > max_depth:
                continue
            for ops in _rule_chain_orderings(case_ops, leet_ops, prepend, append, lead=lead):
                if not ops or len(ops)>max_depth: continue
                chain = ' '.join(ops)
                if not HashcatRuleValidator.validate_rule_for_gpu(chain): continue
                if py_apply_chain(chain, stem) == target:
                    found.add(chain)
    return found

def _extract_letter_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb):
    found = set()
    for pre,mid,suf in _boundary_scan(word, TOKEN_STRIP_BOUNDARY, max_pre, max_suf, min_len):
        found |= _chains_from_middle(mid, pre, suf, base_set, max_depth, min_len, max_amb)
    return found

def _extract_digit_mode(word, base_set, max_depth, min_len, max_pre, max_suf):
    found = set()
    for pre,mid,suf in _boundary_scan(word, TOKEN_STRIP_ALPHA_BOUNDARY, max_pre, max_suf, min_len):
        if not mid.isdigit(): continue
        if mid not in base_set: continue
        prepend = [f'^{c}' for c in reversed(pre)]
        append = [f'${c}' for c in suf]
        if len(prepend)+len(append)==0 or len(prepend)+len(append)>max_depth: continue
        chain = ' '.join(prepend+append)
        if HashcatRuleValidator.validate_rule_for_gpu(chain) and py_apply_chain(chain, mid)==word:
            found.add(chain)
    return found

def _extract_reverse_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb):
    if max_depth<1: return set()
    found = set()
    for pre,mid,suf in _boundary_scan(word, TOKEN_STRIP_BOUNDARY, max_pre, max_suf, min_len):
        rev = mid[::-1]
        found |= _chains_from_middle(rev, pre, suf, base_set, max_depth, min_len, max_amb, lead=['r'])
    return found

def _extract_duplicate_mode(word, base_set, max_depth, min_len):
    found = set()
    L = len(word)
    for op,builder in (('d',lambda s:s+s), ('f',lambda s:s+s[::-1])):
        for h in range(min_len, L//2+1):
            stem = word[:h]
            if builder(stem)==word and stem in base_set:
                if HashcatRuleValidator.validate_rule_for_gpu(op) and py_apply_chain(op, stem)==word:
                    found.add(op)
    return found

def _extract_delete_edge_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb):
    if max_depth<1 or len(word)<min_len+1: return set()
    found = set()
    for op,trim in (('[',word[1:]), (']',word[:-1])):
        for pre,mid,suf in _boundary_scan(trim, TOKEN_STRIP_BOUNDARY, max_pre, max_suf, min_len):
            found |= _chains_from_middle(mid, pre, suf, base_set, max_depth, min_len, max_amb, lead=[op])
    return found

def _extract_insert_mode(word, base_set, max_depth, min_len):
    if max_depth<2: return set()
    found = set()
    L = len(word)
    if L-1 >= min_len:
        for pos in range(min(L,10)):
            cand = word[:pos]+word[pos+1:]
            if cand in base_set:
                rule = f"i{pos}{word[pos]}"
                if HashcatRuleValidator.validate_rule_for_gpu(rule) and py_apply_chain(rule, cand)==word:
                    found.add(rule)
    if max_depth>=3 and L-2 >= min_len:
        for i in range(min(L,10)):
            for j in range(i+1, min(L,10)):
                cand = word[:i]+word[i+1:j]+word[j+1:]
                if len(cand) < min_len: continue
                if cand in base_set:
                    # After inserting word[i] at position i, the string grows by 1,
                    # so the second insert position must be j+1 (not j) to land correctly.
                    chain = f"i{i}{word[i]} i{j+1}{word[j]}"
                    if HashcatRuleValidator.validate_rule_for_gpu(chain) and py_apply_chain(chain, cand)==word:
                        found.add(chain)
    return found

class PackedWordRange:
    """Lazy view over a contiguous or strided range of packed words."""

    __slots__ = ("_buffer", "_indices")

    def __init__(self, buffer, indices):
        self._buffer = buffer
        self._indices = indices if isinstance(indices, range) else range(*indices)

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        for index in self._indices:
            yield self._buffer[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PackedWordRange(self._buffer, self._indices[index])
        return self._buffer[self._indices[index]]


class PackedWordBuffer:
    """UTF-8 word storage backed by shared memory.

    Words are concatenated without separators.  A shared uint64 array with
    ``word_count + 1`` entries stores each word's start/end byte boundaries.
    Pool tasks therefore need only send integer ranges, never Python strings.

    The process that constructs the buffer owns and unlinks the segments;
    workers attach by name and only close their local handles.
    """

    __slots__ = (
        "word_count", "byte_length", "data_shm", "offsets_shm",
        "_offsets", "_owner", "_closed", "encoding",
    )

    def __init__(self, words):
        offsets = array("Q", [0])
        if offsets.itemsize != 8:
            raise RuntimeError("PackedWordBuffer requires 64-bit unsigned offsets")

        packed = bytearray()
        word_count = len(words)
        for index, word in enumerate(words):
            if not isinstance(word, str):
                raise TypeError(
                    f"target word at index {index} is {type(word).__name__}, expected str"
                )
            packed.extend(word.encode("utf-8"))
            offsets.append(len(packed))

        self.word_count = word_count
        self.byte_length = len(packed)
        self.encoding = "utf-8"
        self._owner = True
        self._closed = False
        self.data_shm = SharedMemory(create=True, size=max(1, self.byte_length))
        self.offsets_shm = None
        try:
            self.offsets_shm = SharedMemory(create=True, size=len(offsets) * offsets.itemsize)
            if packed:
                self.data_shm.buf[:self.byte_length] = packed
            self.offsets_shm.buf[:len(offsets) * offsets.itemsize] = offsets.tobytes()
            self._offsets = self.offsets_shm.buf.cast("Q")
        except Exception:
            if self.offsets_shm is not None:
                self.offsets_shm.close()
                self.offsets_shm.unlink()
            self.data_shm.close()
            self.data_shm.unlink()
            raise

    @classmethod
    def attach(cls, data_name, offsets_name, word_count, byte_length, encoding="utf-8"):
        """Attach to a buffer created in another process."""
        instance = cls.__new__(cls)
        instance.word_count = int(word_count)
        instance.byte_length = int(byte_length)
        instance.encoding = str(encoding)
        instance._owner = False
        instance._closed = False
        instance.data_shm = SharedMemory(name=data_name, create=False)
        try:
            instance.offsets_shm = SharedMemory(name=offsets_name, create=False)
        except Exception:
            instance.data_shm.close()
            raise
        instance._offsets = instance.offsets_shm.buf.cast("Q")
        # SharedWordList may reserve a larger index up front using the source
        # line count; only the first word_count + 1 offsets are populated.
        if len(instance._offsets) < instance.word_count + 1:
            instance.close()
            raise ValueError("Shared word offset table is shorter than the word count")
        return instance

    @property
    def data_name(self):
        return self.data_shm.name

    @property
    def offsets_name(self):
        return self.offsets_shm.name

    def __len__(self):
        return self.word_count

    def _decode_word(self, index):
        if index < 0:
            index += self.word_count
        if index < 0 or index >= self.word_count:
            raise IndexError("packed word index out of range")
        start = self._offsets[index]
        end = self._offsets[index + 1]
        return self.data_shm.buf[start:end].tobytes().decode(self.encoding)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return PackedWordRange(self, range(*index.indices(self.word_count)))
        return self._decode_word(index)

    def range_view(self, start, stop):
        start = max(0, min(self.word_count, int(start)))
        stop = max(start, min(self.word_count, int(stop)))
        return PackedWordRange(self, range(start, stop))

    def close(self):
        """Release local handles and unlink segments when called by the owner."""
        if self._closed:
            return
        self._closed = True
        offsets = self._offsets
        self._offsets = None
        if offsets is not None:
            offsets.release()
        for shared in (self.offsets_shm, self.data_shm):
            try:
                shared.close()
            finally:
                if self._owner:
                    try:
                        shared.unlink()
                    except FileNotFoundError:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


_P0_WORD_BUFFER = None
_P0_WORKER_CONFIG = None


def _worker_init_p0(membership_spec, data_name, offsets_name, word_count, byte_length,
                    word_encoding, max_depth, min_len, max_amb, max_pre, max_suf):
    """Attach workers to the shared membership index and packed target words."""
    global _P0_WORD_BUFFER, _P0_WORKER_CONFIG
    if isinstance(membership_spec, tuple) and membership_spec:
        kind = membership_spec[0]
        if kind == "flat64":
            _membership = FlatHashSet64.attach(*membership_spec[1:])
        elif kind == "bloom":
            _membership = BloomFilter.attach(*membership_spec[1:])
        else:
            raise ValueError(f"Unknown Stage-0 membership descriptor: {kind!r}")
    else:
        # Compatibility for direct/internal callers; the normal CLI path always
        # supplies a compact shared-memory descriptor.
        _membership = membership_spec
    worker_state.p0_worker_base_set = _membership
    _P0_WORD_BUFFER = PackedWordBuffer.attach(
        data_name, offsets_name, word_count, byte_length, encoding=word_encoding
    )
    _P0_WORKER_CONFIG = (max_depth, min_len, max_amb, max_pre, max_suf)
    # Shared memory must be closed in every spawned worker.  The owner process
    # alone unlinks the segments after the Pool has shut down.
    atexit.register(_worker_close_p0)


def _worker_close_p0():
    """Close a worker's local shared-memory handles without unlinking owners."""
    global _P0_WORD_BUFFER, _P0_WORKER_CONFIG
    if _P0_WORD_BUFFER is not None:
        _P0_WORD_BUFFER.close()
        _P0_WORD_BUFFER = None
    _P0_WORKER_CONFIG = None
    membership = worker_state.p0_worker_base_set
    if hasattr(membership, "close"):
        try:
            membership.close()
        except Exception:
            pass
    worker_state.p0_worker_base_set = set()


def _process_one_word_p0(word, base_set, max_depth, min_len, max_amb, max_pre, max_suf):
    found = set()
    if not word or len(word) > MAX_WORD_LEN:
        return found
    nd = sum(1 for c in word if c.isdigit())
    na = sum(1 for c in word if c.isalpha())
    if nd > na:
        found |= _extract_digit_mode(word, base_set, max_depth, min_len, max_pre, max_suf)
        if na >= min_len:
            found |= _extract_letter_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
    else:
        found |= _extract_letter_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
        if nd > 0:
            found |= _extract_digit_mode(word, base_set, max_depth, min_len, max_pre, max_suf)
    if max_depth >= 2:
        found |= _extract_reverse_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
        found |= _extract_delete_edge_mode(word, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
    if len(word) >= 2 * min_len:
        found |= _extract_duplicate_mode(word, base_set, max_depth, min_len)
    if max_depth >= 2:
        found |= _extract_insert_mode(word, base_set, max_depth, min_len)
    return found


def _process_range_p0(word_range):
    """Process one word-index range and return its own completion metadata."""
    start, stop = word_range
    if _P0_WORD_BUFFER is None or _P0_WORKER_CONFIG is None:
        raise RuntimeError("Stage-0 worker was not initialized")
    max_depth, min_len, max_amb, max_pre, max_suf = _P0_WORKER_CONFIG
    base_set = worker_state.p0_worker_base_set
    found = set()
    for word in _P0_WORD_BUFFER.range_view(start, stop):
        found.update(_process_one_word_p0(
            word, base_set, max_depth, min_len, max_amb, max_pre, max_suf
        ))
    return start, stop, found


def _iter_word_ranges(n_words, chunk_size):
    """Yield lightweight index-only tasks without materializing task lists."""
    for start in range(0, n_words, chunk_size):
        yield start, min(start + chunk_size, n_words)

def _generate_toggle_chain_seeds(max_depth):
    leet_ops = ['sa@','se3','so0','si1','sl1','ss5','ss$','st7','sa4','si!']
    double_leet = [('se3','si1'),('se3','sl1'),('sa@','so0'),('ss5','so0'),
                   ('si1','so0'),('se3','so0'),('ss$','se3'),('sa4','sl1')]
    seeds = set()
    def _add(ops):
        if not ops or len(ops)>max_depth: return
        chain = ' '.join(ops)
        if HashcatRuleValidator.validate_rule_for_gpu(chain): seeds.add(chain)
    for n in range(min(10,max_depth)):
        togg = [f'T{i}' for i in range(n+1)]
        _add(togg)
        for l in leet_ops:
            _add(togg+[l])
            if n>=1: _add([l]+togg)
        for l1,l2 in double_leet:
            _add(togg+[l1,l2]); _add([l1]+togg+[l2])
    for n in range(1,min(5,max_depth)):
        togg = [f'T{i*2}' for i in range(n+1)]
        for l in leet_ops: _add(togg+[l]); _add([l]+togg)
        for l1,l2 in double_leet: _add(togg+[l1,l2])
    for n in range(1,min(5,max_depth)):
        togg = [f'T{i*2+1}' for i in range(n+1)]
        for l in leet_ops: _add(togg+[l]); _add([l]+togg)
        for l1,l2 in double_leet: _add(togg+[l1,l2])
    for l in leet_ops: _add(['T0',l]); _add([l,'T0'])
    return sorted(seeds)

def extract_token_strip_rules(target_words, base_set, max_depth=0, min_stem_len=4,
                              max_prefix_len=4, max_suffix_len=4, max_leet_ambiguity=3,
                              workers=0, chunk_size=0, bloom_mb=0, start_method=None):
    """Extract Stage-0 rules using shared-memory word and membership buffers.

    Target tasks contain only ``(start, stop)`` word-index ranges.  When the
    loader supplied a :class:`SharedWordList`, workers attach directly to its
    existing UTF-8/Latin-1 byte and uint64-offset segments without repacking it.
    The base-word membership table is also shared; no Python ``set[str]`` is
    pickled into each Windows ``spawn`` worker.
    """
    if max_depth <= 0:
        max_depth = MAX_HASHCAT_CHAIN
    n_workers = safe_worker_count(workers or mp.cpu_count())
    n_words = len(target_words)
    if n_words == 0:
        return []
    if chunk_size <= 0:
        chunk_size = max(500, n_words // (n_workers * 4) + 1)
    chunk_size = max(1, int(chunk_size))

    # Reuse the loader's already-built shared base index whenever available.
    membership = getattr(base_set, "membership_index", None) if bloom_mb == 0 else None
    membership_from_base = membership is not None
    membership_owned = False
    if membership is None and isinstance(base_set, FlatHashSet64) and bloom_mb == 0:
        membership = base_set
    if membership is None:
        membership, mode_str = _build_membership(base_set, bloom_mb=bloom_mb)
        membership_owned = True
        log_info(f"[S0]    Membership : {mode_str}")
    else:
        if isinstance(membership, FlatHashSet64):
            mode_str = f"shared exact hash table ({len(membership):,} words, {membership.size_bytes / 1024 / 1024:.1f} MiB)"
        else:
            mode_str = f"shared Bloom filter ({membership.size_bytes / 1024 / 1024:.1f} MiB)"
        log_info(f"[S0]    Membership : {mode_str}")

    worker_state.p0_worker_base_set = membership
    found = set()
    if start_method is None:
        start_method = "fork" if hasattr(os, "fork") else "spawn"
    ctx = mp.get_context(start_method)
    n_tasks = (n_words + chunk_size - 1) // chunk_size
    log_info(
        f"[S0]    Workers: {bold(str(n_workers))}  |  "
        f"ranges: {bold(f'{n_tasks:,}')} × up to {chunk_size:,} words"
    )
    ncols = shutil.get_terminal_size((80, 24)).columns

    packed_words_owner = None
    word_encoding = "utf-8"
    if hasattr(target_words, "shared_descriptor"):
        data_name, offsets_name, word_count, byte_length = target_words.shared_descriptor()
        word_encoding = getattr(target_words, "word_encoding", "utf-8")
    else:
        packed_words_owner = PackedWordBuffer(target_words)
        data_name = packed_words_owner.data_name
        offsets_name = packed_words_owner.offsets_name
        word_count = packed_words_owner.word_count
        byte_length = packed_words_owner.byte_length

    if isinstance(membership, FlatHashSet64):
        membership_spec = ("flat64", *membership.descriptor())
    elif isinstance(membership, BloomFilter):
        membership_spec = membership.descriptor()
    else:
        # Compatibility only; the normal shared loader/CLI never takes this path.
        membership_spec = membership

    try:
        initargs = (
            membership_spec,
            data_name,
            offsets_name,
            word_count,
            byte_length,
            word_encoding,
            max_depth,
            min_stem_len,
            max_leet_ambiguity,
            max_prefix_len,
            max_suffix_len,
        )
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init_p0,
            initargs=initargs,
        ) as pool:
            with tqdm(
                total=n_words,
                desc=green("  STAGE 0 "),
                unit="word",
                ncols=ncols,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            ) as pbar:
                t0 = time.time()
                words_done = 0
                for start, stop, result in pool.imap_unordered(
                    _process_range_p0,
                    _iter_word_ranges(n_words, chunk_size),
                    chunksize=1,
                ):
                    if _kb.poll():
                        log_warn("[S0] Early exit — keeping rules found so far")
                        pool.terminate()
                        break
                    found.update(result)
                    completed_words = stop - start
                    words_done += completed_words
                    elapsed = time.time() - t0
                    speed = _fmt_speed(words_done / elapsed if elapsed > 0 else 0, "words")
                    pbar.set_postfix(
                        {"rules": cyan(str(len(found))), "spd": green(speed)},
                        refresh=False,
                    )
                    pbar.update(completed_words)
    finally:
        if packed_words_owner is not None:
            packed_words_owner.close()
        if (membership_owned or membership_from_base) and hasattr(membership, "close"):
            membership.close(unlink=True)
            if membership_from_base:
                # Stage 0 is the only consumer of the base exact-membership
                # table.  Release it now instead of keeping hundreds of MiB
                # resident through GPU stages 1-3.
                base_set.membership_index = None
        worker_state.p0_worker_base_set = set()

    return sorted(found)

def _log_token_strip_stats(n_words, rules, inject_sbd):
    if not rules:
        log_info(f"[S0]    {yellow('0')} rules extracted ({n_words:,} words)")
        return
    depth = defaultdict(int); mode = defaultdict(int)
    for r in rules:
        toks = r.split()
        depth[len(toks)] += 1
        first = toks[0] if toks else ''
        if first == 'r': mode['reverse']+=1
        elif first == 'd': mode['dup']+=1
        elif first == 'f': mode['fold']+=1
        elif first in ('[',']'): mode['del-edge']+=1
        elif first.startswith('T') and len(first)==2: mode['toggle']+=1
        elif first.startswith('i') and len(first)==3: mode['insert']+=1
        elif all(c.isdigit() or c in ('^','$',' ') for c in r) and any(c.isdigit() for c in r): mode['digit-bnd']+=1
        else: mode['letter']+=1
    depth_sum = '  '.join(f"d{d}:{depth[d]:,}" for d in sorted(depth))
    inj = green('→ STAGE S sbd → STAGE 2') if inject_sbd else dim('STAGE S inactive')
    mod_str = '  '.join(f"{k}:{v}" for k,v in sorted(mode.items()) if v)
    log_info(f"[S0]    {bold(green(str(len(rules))))} rules extracted  ({depth_sum})  {inj}")
    if mod_str: log_info(f"[S0]    Mode breakdown : {dim(mod_str)}")
