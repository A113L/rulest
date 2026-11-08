"""Unit tests for rulest.io_utils.load_wordlist."""
import os
import sys
import multiprocessing as mp

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest.io_utils import load_wordlist
from rulest.common import MAX_WORD_LEN


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def _spawn_read_shared_wordlist(words):
    return len(words), words[0], words[-1], list(words[1:3])


class TestLoadWordlist:

    def test_loads_unique_words(self, tmp_path):
        path = _write(tmp_path, 'words.txt', b"alpha\nbeta\nalpha\ngamma\n")
        words = load_wordlist(path)
        assert sorted(words) == ['alpha', 'beta', 'gamma']

    def test_strips_whitespace_and_crlf(self, tmp_path):
        path = _write(tmp_path, 'words.txt', b"alpha \r\n beta\r\n")
        words = load_wordlist(path)
        assert sorted(words) == ['alpha', 'beta']

    def test_skips_blank_lines(self, tmp_path):
        path = _write(tmp_path, 'words.txt', b"alpha\n\n\nbeta\n")
        words = load_wordlist(path)
        assert sorted(words) == ['alpha', 'beta']

    def test_skips_words_longer_than_max(self, tmp_path):
        too_long = b'a' * (MAX_WORD_LEN + 1)
        ok = b'shortword'
        path = _write(tmp_path, 'words.txt', too_long + b'\n' + ok + b'\n')
        words = load_wordlist(path)
        assert words == ['shortword']

    def test_keeps_words_at_exactly_max_len(self, tmp_path):
        exact = b'a' * MAX_WORD_LEN
        path = _write(tmp_path, 'words.txt', exact + b'\n')
        words = load_wordlist(path)
        assert words == [exact.decode('latin-1')]

    def test_missing_file_exits(self, tmp_path):
        missing = str(tmp_path / 'does_not_exist.txt')
        with pytest.raises(SystemExit):
            load_wordlist(missing)

    def test_non_utf8_bytes_are_decoded_leniently(self, tmp_path):
        # latin-1 decoding with errors='ignore' should not raise
        path = _write(tmp_path, 'words.txt', b"\xff\xfeword\n")
        words = load_wordlist(path)
        assert len(words) == 1

    def test_returns_shared_memory_sequence(self, tmp_path):
        from rulest.io_utils import SharedWordList
        path = _write(tmp_path, 'words.txt', b"a\nb\n")
        words = load_wordlist(path)
        try:
            assert isinstance(words, SharedWordList)
            assert list(words) == ["a", "b"]
            assert words._data_shm.name
            assert words._offsets_shm.name
        finally:
            words.close()


    def test_numpy_flatten_uses_packed_ram_and_lazy_slices(self, tmp_path):
        path = _write(tmp_path, 'words.txt', b"alpha\nbeta\ngamma\n")
        words = load_wordlist(path)
        try:
            flat, offsets, lengths = words.numpy_flatten()
            assert bytes(flat) == b"alphabetagamma"
            assert offsets.tolist() == [0, 5, 9]
            assert lengths.tolist() == [5, 4, 5]

            flat2, offsets2, lengths2 = words[1:].numpy_flatten()
            assert bytes(flat2) == b"betagamma"
            assert offsets2.tolist() == [0, 4]
            assert lengths2.tolist() == [4, 5]
            del flat, offsets, lengths, flat2, offsets2, lengths2
        finally:
            words.close()


    def test_wordlist_pickle_attaches_shared_memory_in_spawn(self, tmp_path):
        path = _write(tmp_path, 'words.txt', b"alpha\nbeta\ngamma\n")
        words = load_wordlist(path)
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(1) as pool:
                result = pool.apply(_spawn_read_shared_wordlist, (words,))
            assert result == (3, "alpha", "gamma", ["beta", "gamma"])
        finally:
            words.close()

    def test_retains_compact_membership_index_for_stage0(self, tmp_path):
        from rulest.tokenstrip_hashset import FlatHashSet64
        path = _write(tmp_path, 'words.txt', b"alpha\nbeta\nalpha\n")
        words = load_wordlist(path, retain_membership_index=True)
        try:
            assert isinstance(words.membership_index, FlatHashSet64)
            assert "alpha" in words.membership_index
            assert "beta" in words.membership_index
            assert "missing" not in words.membership_index
            assert words.membership_index.shared_name
        finally:
            words.close()


if __name__ == "__main__":
    import pytest
    print(f"Running {__file__} directly — use `pytest {__file__}` for full output.\n")
    raise SystemExit(pytest.main([__file__, "-v"]))
