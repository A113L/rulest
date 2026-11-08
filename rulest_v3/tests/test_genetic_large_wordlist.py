"""Regression tests for Stage 3 memory/progress behaviour on large wordlists."""
from types import SimpleNamespace

import numpy as np

from rulest.genetic import GeneticRuleEvolver, _ga_uses_full_sweep, _hit_count
from rulest.gpu_engine import GPUEngine


def test_large_wordlist_does_not_force_generation_zero_or_periodic_full_sweeps():
    # Large lists run bounded samples throughout GA; the exact candidate pass
    # later restores counts against the full wordlist.
    assert _ga_uses_full_sweep(0, True, False, 4) is False
    assert _ga_uses_full_sweep(4, True, False, 4) is False
    assert _ga_uses_full_sweep(0, True, True, 4) is False
    assert _ga_uses_full_sweep(4, True, True, 4) is True
    assert _ga_uses_full_sweep(1, False, False, 4) is True


def test_hit_bitmap_population_count_matches_set_count():
    packed = np.zeros(3, dtype=np.uint8)
    packed[0] = (1 << 0) | (1 << 3) | (1 << 7)
    packed[1] = (1 << 2) | (1 << 5)
    assert _hit_count(packed) == 5
    assert _hit_count({0, 3, 7, 10, 13}) == 5


def test_eval_on_engine_replaces_empty_bucket_with_compact_bitmap():
    class FakeEngine:
        def evaluate_chains_for_ga(self, chains):
            bitmap = np.zeros(2, dtype=np.uint8)
            bitmap[0] = 0b00000101
            bitmap[1] = 0b00000010
            return {"a b": bitmap}

    evolver = GeneticRuleEvolver.__new__(GeneticRuleEvolver)
    hit_words = {"a b": set(), "c d": set()}
    evolver._eval_on_engine(FakeEngine(), ["a b", "c d"], hit_words)

    assert isinstance(hit_words["a b"], np.ndarray)
    assert _hit_count(hit_words["a b"]) == 3
    assert hit_words["c d"] == set()


def test_gpu_ga_evaluator_builds_bitmaps_not_python_integer_sets():
    # Exercise the GPU-engine aggregation without calling OpenCL.  The real
    # kernel's small (word-index, chain-string) result rows are converted into
    # one bit per word as batches complete.
    fake = SimpleNamespace(
        params={"CHAINS_PER_BATCH": 10, "WORD_SUB_BATCH": 4},
        _base_words_precomputed=(None, np.zeros(8, dtype=np.int32), None),
    )
    fake._compute_chain_seqs = lambda chains: ([], [])

    def get_batch(start, end):
        return {"num_words": end - start, "start": start}

    fake.get_word_batch_data = get_batch

    def run_kernel(batch, chains, _precomputed, return_word_idx, _profile):
        assert return_word_idx is True
        if batch["start"] == 0:
            return [(0, "a b"), (1, "a b"), (3, "c d")]
        return [(0, "c d"), (3, "a b")]

    fake._run_chain_kernel = run_kernel
    result = GPUEngine.evaluate_chains_for_ga(fake, ["a b", "c d"])

    assert isinstance(result["a b"], np.ndarray)
    assert isinstance(result["c d"], np.ndarray)
    assert _hit_count(result["a b"]) == 3  # local indices 0, 1, 7
    assert _hit_count(result["c d"]) == 2  # local indices 3, 4


def test_gpu_ga_count_only_verification_does_not_allocate_hit_bitmaps():
    # The post-GA exact correction only needs the number of matches per chain.
    # It must not allocate one dense word bitmap per candidate chain.
    fake = SimpleNamespace(
        params={"CHAINS_PER_BATCH": 10, "WORD_SUB_BATCH": 4},
        _base_words_precomputed=(None, np.zeros(8, dtype=np.int32), None),
    )
    fake._compute_chain_seqs = lambda chains: ([], [])

    def get_batch(start, end):
        return {"num_words": end - start, "start": start}

    fake.get_word_batch_data = get_batch

    def run_kernel(batch, chains, _precomputed, return_word_idx, _profile):
        assert return_word_idx is True
        if batch["start"] == 0:
            return [(0, "a b"), (1, "a b"), (3, "c d")]
        return [(0, "c d"), (3, "a b")]

    fake._run_chain_kernel = run_kernel
    result = GPUEngine.evaluate_chains_for_ga(fake, ["a b", "c d"], return_counts=True)

    assert result == {"a b": 3, "c d": 2}
    assert all(not isinstance(value, np.ndarray) for value in result.values())


def test_gpu_engine_pins_shared_wordlist_until_cached_view_is_released(tmp_path):
    from rulest.io_utils import load_wordlist

    path = tmp_path / "shared_base.txt"
    path.write_bytes(b"alpha\nbeta\ngamma\n")
    words = load_wordlist(path)
    data_mmap = words._data_shm._mmap
    engine = GPUEngine({})

    engine.set_base_words(words)
    assert engine._base_words_owner is words
    assert engine._base_words_precomputed is not None

    del words
    assert not data_mmap.closed
    engine.release_word_cache()
    assert engine._base_words_precomputed is None
    assert engine._base_words_owner is None
    assert data_mmap.closed


def test_release_run_references_drops_target_capturing_bloom_callback(tmp_path):
    import gc
    from rulest.io_utils import load_wordlist

    path = tmp_path / "shared_target.txt"
    path.write_bytes(b"target-one\ntarget-two\n")
    words = load_wordlist(path)
    data_mmap = words._data_shm._mmap
    engine = GPUEngine({})
    engine.set_base_words(words)
    engine._bloom_recovery_fn = lambda words=words: words
    del words

    engine.release_run_references()
    gc.collect()

    assert engine._base_words_precomputed is None
    assert engine._base_words_owner is None
    assert engine._bloom_recovery_fn is None
    assert data_mmap.closed
