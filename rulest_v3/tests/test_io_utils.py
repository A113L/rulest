"""Unit tests for rulest.io_utils.load_wordlist."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest.io_utils import load_wordlist
from rulest.common import MAX_WORD_LEN


def _write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


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

    def test_returns_list_type(self, tmp_path):
        path = _write(tmp_path, 'words.txt', b"a\nb\n")
        assert isinstance(load_wordlist(path), list)


if __name__ == "__main__":
    import pytest
    print(f"Running {__file__} directly — use `pytest {__file__}` for full output.\n")
    raise SystemExit(pytest.main([__file__, "-v"]))
