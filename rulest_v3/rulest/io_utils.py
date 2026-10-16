"""Wordlist loading."""
import os
import sys

from .common import log_info, log_error, bold, cyan, MAX_WORD_LEN

def load_wordlist(filename):
    """Load a newline-delimited wordlist into a list of unique words.

    Streams the file line by line instead of reading it whole. The
    previous implementation did ``data = f.read()`` (the entire file as one
    ``bytes`` object) followed by ``data.split(b'\\n')`` (a SECOND
    full-size copy, as a list of per-line ``bytes`` objects) before any
    word was even decoded — meaning peak memory during loading alone ran
    to roughly 3-4x the file's size, on top of whatever the final word
    list/set ends up costing in steady state. base_words/target_words stay
    fully materialized in memory for the rest of the pipeline either way
    (bloom-filter construction, GPU buffer flattening, multi-GPU sharding
    and the genetic algorithm's subsampling in extractor.py/gpu_engine.py/
    genetic.py/multi_gpu.py all need random access to the complete
    corpus — this is a structurally different situation from the
    self-contained coverage-matrix streaming in selection.py, and isn't
    something a wordlist loader alone can stream away), so this function's
    return type and contract are unchanged (still one fully materialized
    ``list[str]``). What changes is only how it gets there: iterating the
    file object directly (``for line in f``) reads it incrementally through
    a small internal buffer rather than ever holding the whole file content
    twice at once, which is the actual peak-memory win for large wordlist
    files. Binary-mode file iteration splits on b'\\n' exactly like
    ``bytes.split(b'\\n')`` did (no universal-newline translation in binary
    mode), so parsing behavior — including CRLF handling via the
    ``.strip()`` below — is unchanged.
    """
    words = set()
    try:
        with open(filename, 'rb') as f:
            for line in f:
                w = line.decode('latin-1', errors='ignore').strip()
                if w and len(w) <= MAX_WORD_LEN:
                    words.add(w)
    except FileNotFoundError:
        log_error(f"File not found: {filename}"); sys.exit(1)
    res = list(words)
    log_info(f"[LOAD] {bold(os.path.basename(filename))}: {bold(cyan(f'{len(res):,}'))} unique words")
    return res
