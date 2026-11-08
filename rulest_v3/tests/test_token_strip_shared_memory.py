import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rulest.token_strip as token_strip


def test_packed_word_buffer_round_trips_utf8_and_lazy_ranges():
    words = ["plain", "pāsswörd", "", "nul\x00inside", "終"]
    with token_strip.PackedWordBuffer(words) as packed:
        assert len(packed) == len(words)
        assert packed._offsets.format == "Q"
        assert packed._offsets[-1] == len("".join(words).encode("utf-8"))
        assert list(packed[:]) == words
        assert list(packed.range_view(1, 4)) == words[1:4]
        assert list(packed[1:5:2]) == ["pāsswörd", "nul\x00inside"]
        assert packed[-1] == "終"


def test_word_range_tasks_are_lazy_integer_ranges():
    tasks = token_strip._iter_word_ranges(5, 2)
    assert iter(tasks) is tasks
    assert list(tasks) == [(0, 2), (2, 4), (4, 5)]


def test_unordered_pool_progress_uses_completed_range(monkeypatch):
    progress = {}

    class RecordingProgress:
        def __init__(self, total, **_kwargs):
            progress["total"] = total
            progress["updates"] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def set_postfix(self, *_args, **_kwargs):
            pass

        def update(self, amount):
            progress["updates"].append(amount)

    class FakePool:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            token_strip._worker_close_p0()
            return False

        def imap_unordered(self, function, tasks, chunksize=1):
            assert chunksize == 1
            # Deliberately complete the short final range first.
            for task in reversed(list(tasks)):
                yield function(task)

        def terminate(self):
            pass

    class FakeContext:
        Pool = FakePool

    monkeypatch.setattr(token_strip.mp, "get_context", lambda _method: FakeContext())
    monkeypatch.setattr(token_strip, "tqdm", RecordingProgress)
    monkeypatch.setattr(token_strip._kb, "poll", lambda: False)

    result = token_strip.extract_token_strip_rules(
        ["Hello", "hello", "HELLO", "hello", "Hello"],
        {"hello"},
        max_depth=2,
        min_stem_len=4,
        workers=1,
        chunk_size=2,
    )

    assert {"c", "E", "u"}.issubset(set(result))
    assert progress["total"] == 5
    assert progress["updates"] == [1, 2, 2]
    assert sum(progress["updates"]) == progress["total"]


def test_real_pool_matches_single_and_multi_worker_results():
    targets = ["Hello", "hello", "HELLO", "hElLo"]
    base_words = {"hello"}
    single = token_strip.extract_token_strip_rules(
        targets, base_words, max_depth=2, min_stem_len=4,
        workers=1, chunk_size=1,
    )
    parallel = token_strip.extract_token_strip_rules(
        targets, base_words, max_depth=2, min_stem_len=4,
        workers=2, chunk_size=1,
    )
    assert parallel == single


def test_shared_wordlist_stage0_uses_spawn_without_serializing_words(tmp_path):
    from rulest.io_utils import load_wordlist

    target_file = tmp_path / "targets.txt"
    base_file = tmp_path / "base.txt"
    # Duplicate source rows make the shared offset allocation larger than
    # the unique word count; worker attachment must use only the populated prefix.
    target_file.write_bytes(b"Hello\nhello\nHELLO\nhello\n")
    base_file.write_bytes(b"hello\n")
    targets = load_wordlist(target_file)
    base = load_wordlist(base_file, retain_membership_index=True)
    try:
        result = token_strip.extract_token_strip_rules(
            targets,
            base,
            max_depth=2,
            min_stem_len=4,
            workers=2,
            chunk_size=1,
            start_method="spawn",
        )
        assert {"c", "E", "u"}.issubset(set(result))
        assert base.membership_index is None  # Released after Stage 0.
    finally:
        targets.close()
        base.close()


def test_spawn_stage0_preserves_latin1_words(tmp_path):
    from rulest.io_utils import load_wordlist

    target_file = tmp_path / "latin1_targets.txt"
    base_file = tmp_path / "latin1_base.txt"
    target_file.write_bytes(b"caf\xe9!\n")
    base_file.write_bytes(b"caf\xe9\n")
    targets = load_wordlist(target_file)
    base = load_wordlist(base_file, retain_membership_index=True)
    try:
        assert targets[0] == "café!"
        # The punctuation is a suffix in the actual file; exercise the shared
        # worker attachment path with the same latin-1 encoding as the loader.
        result = token_strip.extract_token_strip_rules(
            targets,
            base,
            max_depth=1,
            min_stem_len=4,
            workers=1,
            chunk_size=1,
            start_method="spawn",
        )
        assert "$!" in result
    finally:
        targets.close()
        base.close()
