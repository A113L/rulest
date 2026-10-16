#!/usr/bin/env python3
"""Entry point: `python3 run_rulest.py <base_wordlist> <target_wordlist> [options]`"""
import multiprocessing as mp

from rulest.cli import main

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except Exception:
        pass
    main()
