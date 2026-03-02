#!/usr/bin/env python3
"""
GPU-OPTIMIZED HASHCAT RULES ENGINE - MITM ENHANCED with OPTIMIZATIONS
Supports higher depths and Meet-in-the-Middle algorithm with performance optimizations
"""

import os
import sys
import numpy as np
import pyopencl as cl
import argparse
import string
import itertools
from collections import defaultdict, deque, Counter
from tqdm import tqdm
import mmap
import time
import hashlib
import signal
import math
import zlib 
import random
import json
import pickle
from typing import List, Dict, Set, Tuple, Optional
import multiprocessing as mp
from functools import lru_cache
import gc
import psutil
import heapq

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def bold(text):
    return f"{Colors.BOLD}{text}{Colors.END}"

def blue(text):
    return f"{Colors.BLUE}{text}{Colors.END}"

def cyan(text):
    return f"{Colors.CYAN}{text}{Colors.END}"

def green(text):
    return f"{Colors.GREEN}{text}{Colors.END}"

def yellow(text):
    return f"{Colors.YELLOW}{text}{Colors.END}"

def red(text):
    return f"{Colors.RED}{text}{Colors.END}"

# Suppress compiler warnings
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

# ====================================================================
# --- HASHCAT RULE VALIDATION (GPU COMPATIBILITY) ---
# ====================================================================

class HashcatRuleValidator:
    """Validates rules according to Hashcat's official GPU compatibility"""
    
    @staticmethod
    def is_digit(c):
        return '0' <= c <= '9'
    
    @staticmethod
    def is_hex_digit(c):
        return ('0' <= c <= '9') or ('A' <= c <= 'F') or ('a' <= c <= 'f')
    
    @staticmethod
    def validate_rule_for_gpu(rule_str):
        """Validate rule for GPU compatibility"""
        line_len = len(rule_str)
        pos = 0
        cnt = 0
        
        MAX_GPU_RULES = 10000
        
        while pos < line_len:
            c = rule_str[pos]
            
            if c == ' ':
                pos += 1
                continue
            
            # Hashcat rule commands with GPU compatibility check
            if c == ':': pos += 1
            elif c == 'l': pos += 1
            elif c == 'u': pos += 1
            elif c == 'c': pos += 1
            elif c == 'C': pos += 1
            elif c == 't': pos += 1
            elif c == 'T':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'r': pos += 1
            elif c == 'd': pos += 1
            elif c == 'p':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'f': pos += 1
            elif c == '{':
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == '}':
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == '$':
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '^':
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '[':
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == ']':
                pos += 1
                if pos < line_len and HashcatRuleValidator.is_digit(rule_str[pos]):
                    pos += 1
            elif c == 'D':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'x':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'i':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == 'o':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '\'':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 's':
                pos += 1
                if pos >= line_len: return False
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == '@':
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c == 'a': pos += 1
            elif c == 'z':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'Z':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'q': pos += 1
            elif c == 'y':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'Y':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'k': pos += 1
            elif c == 'K': pos += 1
            elif c == '*':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'L':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'R':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == '+':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == '-':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == '.':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == ',':
                pos += 1
                if pos >= line_len: return False
                if not HashcatRuleValidator.is_hex_digit(rule_str[pos]): return False
                pos += 1
            elif c == 'E': pos += 1
            elif c == 'e':
                pos += 1
                if pos >= line_len: return False
                pos += 1
            elif c in 'X46M<>!/(=)%Qv3?':
                return False
            else:
                return False
            
            cnt += 1
            if cnt > MAX_GPU_RULES:
                return False
        
        return True
    
    @staticmethod
    def validate_rules_for_gpu(rules_list):
        """Validate a list of rules for GPU compatibility"""
        valid_rules = []
        for rule in rules_list:
            if isinstance(rule, str) and HashcatRuleValidator.validate_rule_for_gpu(rule):
                valid_rules.append(rule)
        return valid_rules

# ====================================================================
# --- OPTIMIZED MEET-IN-THE-MIDDLE CHAIN FINDER ---
# ====================================================================

class OptimizedMITMChainFinder:
    """
    Optimized Meet-in-the-Middle algorithm for finding rule chains
    Uses:
    - Bloom filters for fast rejection
    - Sampling strategies
    - Parallel processing
    - Heuristic pruning
    """
    
    def __init__(self, working_rules: List[str], validator, num_workers=None):
        self.rules = working_rules
        self.validator = validator
        self.rule_to_idx = {r: i for i, r in enumerate(working_rules)}
        self.num_workers = num_workers or mp.cpu_count()
        
    def generate_half_chains_smart(self, depth: int, max_chains: int, 
                                   base_words_sample: List[str], 
                                   target_words_sample: Set[str]) -> Dict[str, List[str]]:
        """Generate half-chains with smart filtering using word samples"""
        print(f"    {cyan('->')} Generating half-chains of depth {depth} (smart mode)...")
        
        half_chains = {}
        
        if depth == 1:
            # Single rules - but filter by effectiveness
            effective_rules = self._filter_effective_rules(base_words_sample, target_words_sample)
            for rule in effective_rules[:max_chains]:
                half_chains[rule] = [rule]
            return half_chains
        
        # Generate chains with heuristic scoring
        candidates = []
        
        # Use rule categories for better chain construction
        categories = self._categorize_rules()
        
        # Generate candidates using beam search
        beam_width = min(100, max_chains)
        
        for _ in range(min(max_chains * 2, 100000)):
            # Build chain from categories
            chain_parts = []
            for _ in range(depth):
                cat = random.choice(list(categories.keys()))
                if categories[cat]:
                    chain_parts.append(random.choice(categories[cat]))
                else:
                    break
            
            if len(chain_parts) == depth:
                chain = ' '.join(chain_parts)
                if all(self.validator.validate_rule_for_gpu(p) for p in chain_parts):
                    # Score the chain
                    score = self._score_chain(chain_parts, base_words_sample, target_words_sample)
                    candidates.append((score, chain, chain_parts))
        
        # Keep best chains
        candidates.sort(reverse=True)
        for _, chain, parts in candidates[:max_chains]:
            half_chains[chain] = parts
        
        return half_chains
    
    def _categorize_rules(self) -> Dict[str, List[str]]:
        """Categorize rules for better chain generation"""
        categories = defaultdict(list)
        for rule in self.rules:
            if len(rule) == 1:
                if rule in 'lucr':
                    categories['core'].append(rule)
                elif rule in 'd f':
                    categories['duplicate'].append(rule)
                elif rule in '{ } [ ]':
                    categories['rotate'].append(rule)
                else:
                    categories['basic'].append(rule)
            elif rule[0] in '^$':
                categories['affix'].append(rule)
            elif rule[0] == 's':
                categories['substitute'].append(rule)
            elif rule[0] in 'TD':
                categories['position'].append(rule)
            elif rule[0] in 'io':
                categories['insert'].append(rule)
            elif rule[0] in 'x*K':
                categories['swap'].append(rule)
            else:
                categories['other'].append(rule)
        return categories
    
    def _filter_effective_rules(self, base_words: List[str], target_words: Set[str]) -> List[str]:
        """Filter rules that actually produce target-like transformations"""
        effective = []
        sample_size = min(100, len(base_words))
        base_sample = random.sample(base_words, sample_size)
        
        for rule in tqdm(self.rules, desc="Filtering rules", leave=False):
            hits = 0
            for word in base_sample:
                try:
                    transformed = self.apply_rule_simple(word, rule)
                    if transformed in target_words:
                        hits += 1
                        if hits >= 3:  # Early stop
                            break
                except:
                    continue
            if hits >= 3:
                effective.append(rule)
        
        return effective
    
    def _score_chain(self, chain_parts: List[str], base_words: List[str], 
                     target_words: Set[str]) -> float:
        """Score a chain based on its effectiveness"""
        sample_size = min(50, len(base_words))
        base_sample = random.sample(base_words, sample_size)
        
        hits = 0
        for word in base_sample:
            try:
                result = self.apply_chain(word, chain_parts)
                if result in target_words:
                    hits += 1
            except:
                continue
        
        return hits / sample_size
    
    def find_combinations_optimized(self, left_chains: Dict[str, List[str]], 
                                    right_chains: Dict[str, List[str]],
                                    base_words: List[str], target_words: Set[str],
                                    max_candidates: int = 100000) -> List[str]:
        """
        Optimized MITM that scales to large chain sets
        """
        print(f"    {cyan('->')} Running OPTIMIZED MITM on {len(left_chains)}×{len(right_chains)} combinations...")
        
        # Use bloom filter for fast rejection
        target_bloom = self._create_bloom_filter(target_words)
        
        # Sample base words for testing
        sample_size = min(200, len(base_words))
        base_sample = random.sample(base_words, sample_size)
        
        # Build signature map for right chains
        print(f"    {cyan('->')} Building signature map...")
        right_signatures = defaultdict(list)
        
        # Process right chains in parallel
        with mp.Pool(self.num_workers) as pool:
            args = [(chain, parts, base_sample[:50]) for chain, parts in right_chains.items()]
            results = pool.starmap(self._compute_chain_signatures, args)
            
            for chain, signatures in results:
                for sig in signatures:
                    right_signatures[sig].append(chain)
        
        # Find matches
        print(f"    {cyan('->')} Finding matches...")
        candidates = []
        
        for left_chain, left_parts in tqdm(left_chains.items(), desc="Matching", leave=False):
            # Sample transformations
            for word in base_sample:
                try:
                    intermediate = self.apply_chain(word, left_parts)
                    # Quick bloom check
                    if self._bloom_check(target_bloom, intermediate):
                        # Look for matches
                        if intermediate in right_signatures:
                            for right_chain in right_signatures[intermediate]:
                                full_chain = f"{left_chain} {right_chain}"
                                # Quick verification
                                if self._quick_verify(full_chain, base_sample, target_words):
                                    candidates.append(full_chain)
                                    if len(candidates) >= max_candidates:
                                        return candidates
                except:
                    continue
        
        return candidates
    
    def _compute_chain_signatures(self, chain: str, parts: List[str], 
                                   words: List[str]) -> Tuple[str, List[str]]:
        """Compute signatures for a chain (for parallel processing)"""
        signatures = []
        for word in words:
            try:
                result = self.apply_chain(word, parts)
                signatures.append(result)
            except:
                continue
        return chain, list(set(signatures))
    
    def _create_bloom_filter(self, words: Set[str], size: int = 1000000) -> np.ndarray:
        """Create a simple bloom filter for fast rejection"""
        bloom = np.zeros(size // 8, dtype=np.uint8)
        for word in words:
            h = hash(word) % size
            byte_idx = h // 8
            bit_idx = h % 8
            bloom[byte_idx] |= (1 << bit_idx)
        return bloom
    
    def _bloom_check(self, bloom: np.ndarray, word: str) -> bool:
        """Check if word might be in bloom filter"""
        h = hash(word) % (len(bloom) * 8)
        byte_idx = h // 8
        bit_idx = h % 8
        return bool(bloom[byte_idx] & (1 << bit_idx))
    
    def _quick_verify(self, chain: str, base_sample: List[str], 
                      target_words: Set[str]) -> bool:
        """Quick verification of a chain"""
        chain_parts = chain.split()
        hits = 0
        for word in base_sample[:10]:
            try:
                result = self.apply_chain(word, chain_parts)
                if result in target_words:
                    hits += 1
                    if hits >= 2:  # Found enough hits
                        return True
            except:
                continue
        return hits >= 2
    
    def apply_chain(self, word: str, chain_parts: List[str]) -> str:
        """Apply a chain of rules to a word"""
        result = word
        for rule in chain_parts:
            result = self.apply_rule_simple(result, rule)
        return result
    
    def apply_rule_simple(self, word: str, rule: str) -> str:
        """Simple rule application for MITM (non-GPU version)"""
        try:
            if rule == 'l': return word.lower()
            if rule == 'u': return word.upper()
            if rule == 'c': return word.capitalize() if word else word
            if rule == 'r': return word[::-1]
            if rule.startswith('^') and len(rule) == 2:
                return rule[1] + word
            if rule.startswith('$') and len(rule) == 2:
                return word + rule[1]
            if rule.startswith('s') and len(rule) == 3:
                return word.replace(rule[1], rule[2])
            if rule.startswith('T') and len(rule) == 2 and rule[1].isdigit():
                pos = int(rule[1])
                if pos < len(word):
                    return word[:pos] + word[pos].swapcase() + word[pos+1:]
            if rule.startswith('D') and len(rule) == 2 and rule[1].isdigit():
                pos = int(rule[1])
                if pos < len(word):
                    return word[:pos] + word[pos+1:]
            if rule.startswith('i') and len(rule) == 3 and rule[1].isdigit():
                pos = int(rule[1])
                if pos <= len(word):
                    return word[:pos] + rule[2] + word[pos:]
            if rule.startswith('{') and len(rule) == 2 and rule[1].isdigit():
                n = int(rule[1]) % len(word) if word else 0
                if word and n:
                    return word[n:] + word[:n]
            if rule.startswith('}') and len(rule) == 2 and rule[1].isdigit():
                n = int(rule[1]) % len(word) if word else 0
                if word and n:
                    return word[-n:] + word[:-n]
        except:
            pass
        return word

# ====================================================================
# --- ADVANCED RULE GENERATION WITH DEPTH CONTROL ---
# ====================================================================

class AdvancedRuleGenerator:
    """Generate comprehensive GPU-compatible Hashcat rules with depth control"""
    
    def __init__(self, slow_mode=False):
        self.slow_mode = slow_mode
        self.validator = HashcatRuleValidator()
    
    def generate_comprehensive_rules(self):
        """Generate comprehensive GPU-compatible rules"""
        rules = set()
        
        print(f"{blue('[SETUP]')} {bold('Generating comprehensive GPU-compatible Hashcat rules...')}")
        
        # ===== CATEGORY 1: SIMPLE RULES =====
        print(f"  {cyan('[*]')} Simple rules...")
        simple_rules = [
            'l', 'u', 'c', 'C', 't', 'r', 'd', 'f', 'p', 'z', 'Z', 'q', 'E',
            '{', '}', '[', ']', 'k', 'K', ':', 'a'
        ]
        rules.update(simple_rules)
        
        # ===== CATEGORY 2: POSITION-BASED RULES =====
        print(f"  {cyan('[*]')} Position-based rules...")
        digits = '0123456789'
        
        position_cmds = ['T', 'D', 'L', 'R', '+', '-', '.', ',', '\'']
        for cmd in position_cmds:
            for pos in digits:
                rules.add(f'{cmd}{pos}')
        
        # Two position rules
        for cmd in ['T', 'x', '*', 'K']:
            for pos1 in digits:
                for pos2 in digits:
                    if self.slow_mode or (int(pos1) < 5 and int(pos2) < 5):
                        rules.add(f'{cmd}{pos1}{pos2}')
        
        # ===== CATEGORY 3: PREFIX/SUFFIX =====
        print(f"  {cyan('[*]')} Prefix/suffix rules...")
        common_chars = string.ascii_letters + string.digits + '!@#$%^&*()_+-='
        for char in common_chars:
            rules.add(f'^{char}')
            rules.add(f'${char}')
        
        # ===== CATEGORY 4: SUBSTITUTIONS =====
        print(f"  {cyan('[*]')} Substitution rules...")
        leet_subs = [
            ('a', '@'), ('a', '4'), ('e', '3'), ('i', '1'), ('o', '0'),
            ('s', '$'), ('s', '5'), ('t', '7'), ('l', '1'), ('g', '9'),
            ('b', '8'), ('z', '2'), ('c', '('), ('c', '{'), ('c', '<'),
            ('d', '6'), ('p', '?'), ('f', 'v'), ('m', 'w'), ('n', '~')
        ]
        
        for orig, sub in leet_subs:
            rules.add(f's{orig}{sub}')
        
        for char in string.punctuation:
            rules.add(f'@{char}')
            rules.add(f'p{char}')
        
        # ===== CATEGORY 5: INSERTION/OVERWRITE =====
        print(f"  {cyan('[*]')} Insertion/overwrite rules...")
        insert_chars = string.ascii_letters + string.digits + '!@#$%^&*'
        positions = digits if self.slow_mode else digits[:5]
        
        for pos in positions:
            for char in insert_chars[:20] if not self.slow_mode else insert_chars[:50]:
                rules.add(f'i{pos}{char}')
                rules.add(f'o{pos}{char}')
        
        # ===== CATEGORY 6: DUPLICATION =====
        print(f"  {cyan('[*]')} Duplication rules...")
        for n in range(1, 10 if self.slow_mode else 6):
            rules.add(f'p{n}')
            rules.add(f'y{n}')
            rules.add(f'Y{n}')
            rules.add(f'z{n}')
            rules.add(f'Z{n}')
        
        # ===== CATEGORY 7: TITLE CASE =====
        print(f"  {cyan('[*]')} Title case rules...")
        separators = [' ', '-', '_', '.', ',', ';', ':', '|', '/', '\\', '+']
        for sep in separators:
            rules.add(f'e{sep}')
        
        # Convert to list and validate
        rules_list = list(rules)
        valid_rules = []
        
        print(f"  {cyan('[*]')} Validating rules for GPU compatibility...")
        for rule in tqdm(rules_list, desc="Validating", leave=False):
            if self.validator.validate_rule_for_gpu(rule):
                if 1 <= len(rule) <= 16:  # MAX_RULE_LEN
                    valid_rules.append(rule)
        
        print(f"{green('[OK]')} {bold('Generated:')} {cyan(f'{len(valid_rules):,}')} {bold('GPU-compatible Hashcat rules')}")
        
        return valid_rules
    
    def generate_intelligent_chains(self, working_rules, max_depth, limits):
        """Generate intelligent chains from working rules for extraction"""
        print(f"  {cyan('->')} Generating intelligent chains up to depth {max_depth}...")
        
        valid_rules = [r for r in working_rules if self.validator.validate_rule_for_gpu(r)]
        
        if not valid_rules:
            print(f"  {yellow('[WARN]')} No valid rules found")
            return []
        
        # Categorize rules
        categories = defaultdict(list)
        for rule in valid_rules:
            if len(rule) == 1:
                categories['basic'].append(rule)
            elif rule[0] in '^$':
                categories['affix'].append(rule)
            elif rule[0] == 's':
                categories['substitute'].append(rule)
            elif rule[0] in 'TD':
                categories['position'].append(rule)
            elif rule[0] in 'io':
                categories['insert'].append(rule)
            elif rule[0] in 'x*K':
                categories['swap'].append(rule)
            elif rule[0] in 'yYzZ':
                categories['duplicate'].append(rule)
            elif rule[0] in '{}[]':
                categories['transform'].append(rule)
            else:
                categories['other'].append(rule)
        
        chains = set()
        
        # Generate chains for each depth
        for depth in range(2, max_depth + 1):
            print(f"    {cyan('->')} Depth {depth} chains...")
            
            # Get limit for this depth
            limit_key = f'CHAIN_GEN_LIMIT_{depth}'
            target = limits.get(limit_key, 50000 // (depth - 1))
            
            chains_added = 0
            attempts = 0
            max_attempts = target * (20 if depth <= 3 else 50)
            
            # Smart patterns based on depth
            if depth == 2:
                patterns = [
                    ('affix', 'basic'),
                    ('basic', 'affix'),
                    ('substitute', 'basic'),
                    ('basic', 'substitute'),
                    ('position', 'basic'),
                    ('insert', 'basic'),
                    ('duplicate', 'affix'),
                    ('affix', 'substitute'),
                ]
            elif depth == 3:
                patterns = [
                    ('affix', 'substitute', 'basic'),
                    ('affix', 'basic', 'affix'),
                    ('basic', 'substitute', 'affix'),
                    ('affix', 'position', 'basic'),
                    ('insert', 'basic', 'affix'),
                    ('substitute', 'substitute', 'affix'),
                    ('position', 'substitute', 'basic'),
                    ('affix', 'duplicate', 'basic'),
                    ('basic', 'position', 'affix'),
                ]
            elif depth == 4:
                patterns = [
                    ('affix', 'basic', 'substitute', 'affix'),
                    ('affix', 'substitute', 'basic', 'affix'),
                    ('basic', 'affix', 'substitute', 'basic'),
                    ('affix', 'position', 'basic', 'affix'),
                    ('insert', 'basic', 'substitute', 'affix'),
                ]
            elif depth == 5:
                patterns = [
                    ('affix', 'basic', 'substitute', 'basic', 'affix'),
                    ('affix', 'substitute', 'basic', 'substitute', 'affix'),
                    ('basic', 'affix', 'substitute', 'basic', 'affix'),
                ]
            elif depth == 6:
                patterns = [
                    ('affix', 'basic', 'substitute', 'basic', 'substitute', 'affix'),
                    ('affix', 'substitute', 'basic', 'substitute', 'basic', 'affix'),
                    ('basic', 'affix', 'substitute', 'basic', 'substitute', 'affix'),
                    ('affix', 'position', 'basic', 'substitute', 'basic', 'affix'),
                    ('insert', 'basic', 'substitute', 'basic', 'substitute', 'affix'),
                ]
            else:  # depth 5+ fallback
                patterns = [
                    ('affix', 'basic', 'substitute', 'basic', 'affix'),
                    ('affix', 'substitute', 'basic', 'substitute', 'affix'),
                    ('basic', 'affix', 'substitute', 'basic', 'affix'),
                ]
            
            while chains_added < target and attempts < max_attempts:
                attempts += 1
                
                # Select random pattern
                if patterns:
                    pattern = random.choice(patterns)
                else:
                    # Fallback to random selection
                    pattern = random.choices(list(categories.keys()), k=depth)
                
                # Build chain
                chain_parts = []
                valid = True
                
                for cat in pattern:
                    if categories[cat]:
                        chain_parts.append(random.choice(categories[cat]))
                    else:
                        valid = False
                        break
                
                if not valid or len(chain_parts) != depth:
                    continue
                
                chain = ' '.join(chain_parts)
                
                if all(self.validator.validate_rule_for_gpu(p) for p in chain_parts):
                    if chain not in chains:
                        chains.add(chain)
                        chains_added += 1
        
        chains_list = list(chains)
        print(f"    {cyan('[*]')} Generated {len(chains_list):,} intelligent chains")
        
        return chains_list

# ====================================================================
# --- GPU ENGINE WITH MITM SUPPORT ---
# ====================================================================

# Constants for GPU kernel
MAX_WORD_LEN = 256
MAX_RULE_LEN = 16
MAX_OUTPUT_LEN = 512
MAX_CHAIN_STRING_LEN = 512

class GPUEngine:
    """GPU-accelerated engine with MITM support"""
    
    def __init__(self, params):
        self.params = params
        self.context = None
        self.queue = None
        self.device = None
        self.program = None
        self.max_work_group_size = 256
        self.local_work_size = params.get('LOCAL_WORK_SIZE', 256)
        
    def initialize_gpu(self):
        """Initialize OpenCL with dynamic parameters"""
        try:
            platform = cl.get_platforms()[0]
            self.device = platform.get_devices(cl.device_type.GPU)[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            
            global_mem = self.device.global_mem_size
            self.max_work_group_size = self.device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            max_compute_units = self.device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            
            print(f"{green('[GPU]')} {bold('GPU:')} {cyan(self.device.name.strip())}")
            print(f"{blue('[INFO]')} {bold('Global Memory:')} {cyan(f'{global_mem // (1024**3)}GB')}")
            print(f"{blue('[INFO]')} {bold('Max Work Group Size:')} {cyan(self.max_work_group_size)}")
            print(f"{blue('[INFO]')} {bold('Compute Units:')} {cyan(max_compute_units)}")
            
            self.local_work_size = min(self.local_work_size, self.max_work_group_size)
            while self.max_work_group_size % self.local_work_size != 0 and self.local_work_size > 32:
                self.local_work_size //= 2
            
            print(f"{blue('[INFO]')} {bold('Using Work Group Size:')} {cyan(self.local_work_size)}")
            
            return True
        except Exception as e:
            print(f"{red('[ERROR]')} {bold('GPU initialization failed:')} {e}")
            return False
    
    def compile_kernel(self):
        """Compile the GPU kernel"""
        try:
            print(f"{blue('[SETUP]')} {bold('Compiling GPU-compatible kernel...')}")
            
            kernel_source = GPU_COMPATIBLE_KERNEL
            
            kernel_source = kernel_source.replace(
                "BLOOM_FILTER_SIZE 8388608",
                f"BLOOM_FILTER_SIZE {self.params['BLOOM_FILTER_SIZE']}"
            )
            kernel_source = kernel_source.replace(
                "MAX_CHAINS_TO_FIND 10000000",
                f"MAX_CHAINS_TO_FIND {self.params['MAX_CHAINS_TO_FIND']}"
            )
            kernel_source = kernel_source.replace(
                "MAX_CHAIN_DEPTH 6",
                f"MAX_CHAIN_DEPTH {self.params.get('MAX_CHAIN_DEPTH', 6)}"
            )
            
            self.program = cl.Program(self.context, kernel_source).build()
            print(f"{green('[OK]')} {bold('Kernel compiled successfully')}")
            return self.program
        except Exception as e:
            print(f"{red('[ERROR]')} {bold('Kernel compilation failed:')}")
            print(f"  {str(e)}")
            return None
    
    def generate_bloom_filter(self, target_words):
        """Generate Bloom filter with dynamic size"""
        print(f"{blue('[SETUP]')} {bold('Generating comprehensive Bloom filter...')}")
        
        bloom_size_bytes = self.params['BLOOM_FILTER_SIZE'] // 8
        bloom_filter = np.zeros(bloom_size_bytes, dtype=np.uint8)
        
        print(f"  {cyan('[*]')} Bloom filter size: {bloom_size_bytes / 1024:.1f} KB")
        print(f"  {cyan('[*]')} Hashing ALL target words: {len(target_words):,}")
        
        for word in tqdm(target_words, desc="Building bloom filter", leave=False):
            word_bytes = word.encode('latin-1')
            
            hash_funcs = [hashlib.md5, hashlib.sha1, hashlib.sha256]
            
            for hash_func in hash_funcs:
                hash_val = int(hash_func(word_bytes).hexdigest()[:8], 16)
                idx = hash_val % self.params['BLOOM_FILTER_SIZE']
                byte_idx = idx // 8
                bit_idx = idx % 8
                bloom_filter[byte_idx] |= (1 << bit_idx)
        
        bits_set = np.sum(np.unpackbits(bloom_filter))
        fill_ratio = bits_set / self.params['BLOOM_FILTER_SIZE']
        
        print(f"  {cyan('[*]')} Bloom filter fill ratio: {fill_ratio:.3%}")
        print(f"  {cyan('[*]')} False positive rate: {(fill_ratio**3):.6%}")
        
        return bloom_filter
    
    def prepare_batch_data(self, words, rules):
        """Prepare data for a single batch"""
        words_flat = []
        word_offsets = []
        word_lengths = []
        current_offset = 0
        
        for word in words:
            word_bytes = word.encode('latin-1')
            words_flat.extend(word_bytes)
            word_offsets.append(current_offset)
            word_lengths.append(len(word_bytes))
            current_offset += len(word_bytes)
        
        rules_flat = []
        rule_offsets = []
        rule_lengths = []
        current_offset = 0
        
        for rule in rules:
            rule_bytes = rule.encode('latin-1')
            rules_flat.extend(rule_bytes)
            rule_offsets.append(current_offset)
            rule_lengths.append(len(rule_bytes))
            current_offset += len(rule_bytes)
        
        return {
            'words_flat': np.array(words_flat, dtype=np.uint8),
            'word_offsets': np.array(word_offsets, dtype=np.int32),
            'word_lengths': np.array(word_lengths, dtype=np.int32),
            'rules_flat': np.array(rules_flat, dtype=np.uint8),
            'rule_offsets': np.array(rule_offsets, dtype=np.int32),
            'rule_lengths': np.array(rule_lengths, dtype=np.int32),
            'num_words': len(words),
            'num_rules': len(rules)
        }
    
    def process_all_words_single_rule(self, base_words, target_words, rules):
        """Process ALL base words with single rules"""
        print(f"{blue('[GPU]')} {bold('Processing ALL words with single rules...')}")
        
        bloom_filter = self.generate_bloom_filter(target_words)
        
        if not self.compile_kernel():
            return []
        
        gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(rules)
        
        print(f"{blue('[INFO]')} {bold('GPU-compatible rules:')} {len(gpu_rules):,}")
        
        all_found_rules = []
        
        batch_size = min(10000, len(base_words))
        num_batches = (len(base_words) + batch_size - 1) // batch_size
        
        print(f"{blue('[INFO]')} {bold('Processing ALL')} {len(base_words):,} {bold('words in')} {num_batches} {bold('batches')}")
        print(f"{blue('[INFO]')} {bold('Batch size:')} {batch_size:,} words")
        
        with tqdm(total=num_batches, desc="Processing single rules", unit="batch") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(base_words))
                batch_words = base_words[start_idx:end_idx]
                
                if not batch_words:
                    pbar.update(1)
                    continue
                
                batch_data = self.prepare_batch_data(batch_words, gpu_rules)
                batch_found = self.process_batch_single(batch_data, bloom_filter, gpu_rules)
                
                if batch_found:
                    all_found_rules.extend(batch_found)
                
                pbar.set_postfix({
                    'found': len(all_found_rules),
                    'progress': f"{end_idx:,}/{len(base_words):,}"
                })
                pbar.update(1)
        
        all_found_rules = list(set(all_found_rules))
        print(f"\n{green('[OK]')} {bold('Total unique single rules found:')} {cyan(len(all_found_rules))}")
        
        return all_found_rules
    
    def process_batch_single(self, batch_data, bloom_filter, rules):
        """Process a single batch on GPU"""
        mf = cl.mem_flags
        
        try:
            base_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['words_flat'])
            base_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_offsets'])
            base_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_lengths'])
            
            rules_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['rules_flat'])
            rule_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_offsets'])
            rule_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_lengths'])
            
            bloom_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=bloom_filter)
            
            max_output_size = min(self.params['MAX_CHAINS_TO_FIND'], 
                                 batch_data['num_words'] * batch_data['num_rules'])
            
            found_rules_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                      max_output_size * MAX_CHAIN_STRING_LEN)
            found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)
            
            zero_count = np.array([0], dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count_buf, zero_count)
            
            kernel = self.program.find_single_rules_gpu
            
            total_combinations = batch_data['num_words'] * batch_data['num_rules']
            global_size = ((total_combinations + self.local_work_size - 1) // 
                          self.local_work_size) * self.local_work_size
            
            kernel.set_args(
                base_buf, base_offsets_buf, base_lengths_buf,
                rules_buf, rule_offsets_buf, rule_lengths_buf,
                bloom_buf,
                np.int32(batch_data['num_words']),
                np.int32(batch_data['num_rules']),
                found_rules_buf,
                found_count_buf
            )
            
            cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (self.local_work_size,))
            self.queue.finish()
            
            found_count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count, found_count_buf)
            
            num_found = min(found_count[0], max_output_size)
            
            batch_found = []
            if num_found > 0:
                found_data = np.zeros(num_found * MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, found_data, found_rules_buf)
                
                for i in range(num_found):
                    start = i * MAX_CHAIN_STRING_LEN
                    rule_bytes = bytes(found_data[start:start + MAX_CHAIN_STRING_LEN])
                    rule_str = rule_bytes.split(b'\0')[0].decode('latin-1', errors='ignore')
                    if rule_str:
                        batch_found.append(rule_str)
            
            return batch_found
            
        except Exception as e:
            print(f"{yellow('[WARN]')} GPU processing failed: {e}")
            return []
            
        finally:
            try:
                base_buf.release()
                base_offsets_buf.release()
                base_lengths_buf.release()
                rules_buf.release()
                rule_offsets_buf.release()
                rule_lengths_buf.release()
                bloom_buf.release()
                found_rules_buf.release()
                found_count_buf.release()
            except:
                pass
    
    def process_all_words_chain_rules(self, base_words, target_words, rules, chains):
        """Process ALL base words with rule chains"""
        print(f"{blue('[GPU]')} {bold('Processing ALL words with rule chains...')}")
        
        bloom_filter = self.generate_bloom_filter(target_words)
        
        if not self.compile_kernel():
            return []
        
        gpu_rules = HashcatRuleValidator.validate_rules_for_gpu(rules)
        
        print(f"{blue('[INFO]')} {bold('Testing')} {len(chains):,} {bold('chains...')}")
        
        all_found_chains = []
        
        chain_batch_size = min(5000, len(chains))
        num_chain_batches = (len(chains) + chain_batch_size - 1) // chain_batch_size
        
        print(f"{blue('[INFO]')} {bold('Processing in')} {num_chain_batches} {bold('batches')}")
        
        with tqdm(total=num_chain_batches, desc="Processing chains", unit="batch") as pbar:
            for chain_batch_idx in range(0, len(chains), chain_batch_size):
                chain_end = min(chain_batch_idx + chain_batch_size, len(chains))
                chain_batch = chains[chain_batch_idx:chain_end]
                
                batch_chains = self._process_chain_batch(
                    base_words, gpu_rules, chain_batch, bloom_filter
                )
                
                if batch_chains:
                    all_found_chains.extend(batch_chains)
                
                pbar.update(1)
                pbar.set_postfix({
                    'found': len(all_found_chains),
                    'progress': f"{chain_end}/{len(chains)}"
                })
        
        all_found_chains = list(set(all_found_chains))
        print(f"\n{green('[OK]')} {bold('Total unique chains found:')} {cyan(len(all_found_chains))}")
        
        return all_found_chains
    
    def _process_chain_batch(self, words, rules, chains, bloom_filter):
        """Process a single chain batch"""
        
        max_depth = self.params.get('MAX_CHAIN_DEPTH', 6)
        
        # Prepare chain sequences
        chain_sequences = []
        chain_depths = []
        
        for chain in chains:
            chain_rules = chain.split()
            depth = len(chain_rules)
            chain_depths.append(depth)
            
            rule_indices = []
            for rule in chain_rules:
                try:
                    idx = rules.index(rule)
                    rule_indices.append(idx)
                except ValueError:
                    rule_indices.append(-1)
            
            while len(rule_indices) < max_depth:
                rule_indices.append(-1)
            
            chain_sequences.extend(rule_indices)
        
        batch_data = self.prepare_batch_data(words, rules)
        
        mf = cl.mem_flags
        
        try:
            base_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['words_flat'])
            base_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_offsets'])
            base_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['word_lengths'])
            
            rules_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=batch_data['rules_flat'])
            rule_offsets_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_offsets'])
            rule_lengths_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                       hostbuf=batch_data['rule_lengths'])
            
            chain_seq_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=np.array(chain_sequences, dtype=np.int32))
            chain_depth_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                      hostbuf=np.array(chain_depths, dtype=np.int32))
            
            bloom_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=bloom_filter)
            
            max_output_size = min(self.params['MAX_CHAINS_TO_FIND'], 
                                 len(words) * len(chains))
            
            found_chains_buf = cl.Buffer(self.context, mf.WRITE_ONLY,
                                       max_output_size * MAX_CHAIN_STRING_LEN)
            found_count_buf = cl.Buffer(self.context, mf.READ_WRITE, 4)
            
            zero_count = np.array([0], dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count_buf, zero_count)
            
            kernel = self.program.find_rule_chains_gpu
            
            total_items = len(words) * len(chains)
            
            global_size = ((total_items + self.local_work_size - 1) // 
                          self.local_work_size) * self.local_work_size
            
            kernel.set_args(
                base_buf, base_offsets_buf, base_lengths_buf,
                rules_buf, rule_offsets_buf, rule_lengths_buf,
                chain_seq_buf, chain_depth_buf,
                bloom_buf,
                np.int32(len(words)),
                np.int32(len(chains)),
                np.int32(max_depth),
                found_chains_buf,
                found_count_buf
            )
            
            cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (self.local_work_size,))
            self.queue.finish()
            
            found_count = np.zeros(1, dtype=np.int32)
            cl.enqueue_copy(self.queue, found_count, found_count_buf)
            
            num_found = min(found_count[0], max_output_size)
            
            batch_chains = []
            if num_found > 0:
                found_data = np.zeros(num_found * MAX_CHAIN_STRING_LEN, dtype=np.uint8)
                cl.enqueue_copy(self.queue, found_data, found_chains_buf)
                
                for i in range(num_found):
                    start = i * MAX_CHAIN_STRING_LEN
                    chain_bytes = bytes(found_data[start:start + MAX_CHAIN_STRING_LEN])
                    chain_str = chain_bytes.split(b'\0')[0].decode('latin-1', errors='ignore')
                    if chain_str:
                        batch_chains.append(chain_str)
            
            return batch_chains
            
        except Exception as e:
            print(f"{yellow('[WARN]')} GPU chain processing failed: {e}")
            return []
            
        finally:
            try:
                base_buf.release()
                base_offsets_buf.release()
                base_lengths_buf.release()
                rules_buf.release()
                rule_offsets_buf.release()
                rule_lengths_buf.release()
                chain_seq_buf.release()
                chain_depth_buf.release()
                bloom_buf.release()
                found_chains_buf.release()
                found_count_buf.release()
            except:
                pass

# ====================================================================
# --- GPU EXTRACTOR WITH OPTIMIZED MITM ---
# ====================================================================

class GPUExtractor:
    """GPU-optimized extractor with optimized MITM"""
    
    def __init__(self, base_count, target_count, device=None, slow_mode=False, target_hours=1.0):
        self.base_count = base_count
        self.target_count = target_count
        self.slow_mode = slow_mode
        
        self.params = calculate_dynamic_parameters(base_count, target_count, device, target_hours)
        
        # Increase chain depth for MITM
        self.params['MAX_CHAIN_DEPTH'] = 6
        
        # Adjust chain generation limits for higher depths
        if base_count > 1000000:
            self.params['CHAIN_GEN_LIMIT_2'] = 200000
            self.params['CHAIN_GEN_LIMIT_3'] = 100000
            self.params['CHAIN_GEN_LIMIT_4'] = 50000
            self.params['CHAIN_GEN_LIMIT_5'] = 25000
            self.params['CHAIN_GEN_LIMIT_6'] = 10000
        elif base_count > 100000:
            self.params['CHAIN_GEN_LIMIT_2'] = 150000
            self.params['CHAIN_GEN_LIMIT_3'] = 75000
            self.params['CHAIN_GEN_LIMIT_4'] = 35000
            self.params['CHAIN_GEN_LIMIT_5'] = 15000
            self.params['CHAIN_GEN_LIMIT_6'] = 5000
        else:
            self.params['CHAIN_GEN_LIMIT_2'] = 75000
            self.params['CHAIN_GEN_LIMIT_3'] = 35000
            self.params['CHAIN_GEN_LIMIT_4'] = 15000
            self.params['CHAIN_GEN_LIMIT_5'] = 5000
            self.params['CHAIN_GEN_LIMIT_6'] = 2000
        
        # MITM-specific limits
        self.params['MITM_MAX_CANDIDATES'] = 200000  # Max candidates before GPU testing
        self.params['MITM_BEAM_WIDTH'] = 5000  # Beam width for chain generation
        
        print(f"{blue('[CONFIG]')} {bold('GPU-Optimized Configuration:')}")
        print(f"  {cyan('[*]')} Bloom filter size: {self.params['BLOOM_FILTER_SIZE']:,} bits")
        print(f"  {cyan('[*]')} Max chains to find: {self.params['MAX_CHAINS_TO_FIND']:,}")
        print(f"  {cyan('[*]')} Max chain depth: {self.params['MAX_CHAIN_DEPTH']}")
        print(f"  {cyan('[*]')} MITM max candidates: {self.params['MITM_MAX_CANDIDATES']:,}")
        print(f"  {cyan('[*]')} Chain gen limits: D2={self.params['CHAIN_GEN_LIMIT_2']:,}, "
              f"D3={self.params['CHAIN_GEN_LIMIT_3']:,}, D4={self.params['CHAIN_GEN_LIMIT_4']:,}, "
              f"D5={self.params['CHAIN_GEN_LIMIT_5']:,}, D6={self.params['CHAIN_GEN_LIMIT_6']:,}")
        
        self.rule_generator = AdvancedRuleGenerator(slow_mode)
        self.gpu_engine = GPUEngine(self.params)
        self.validator = HashcatRuleValidator()
    
    def extract_rules(self, base_words, target_words, max_depth=6):
        """Extract GPU-compatible rules using optimized MITM"""
        print(f"{blue('[MAIN]')} {bold('Starting GPU-optimized rule extraction with OPTIMIZED MITM...')}")
        
        all_chains = []
        
        # Generate comprehensive GPU-compatible rules
        rules = self.rule_generator.generate_comprehensive_rules()
        
        # Initialize GPU
        if not self.gpu_engine.initialize_gpu():
            print(f"{yellow('[WARN]')} {bold('GPU not available')}")
            return []
        
        # Phase 1: Single rules
        print(f"\n{blue('=' * 60)}")
        print(f"{bold('PHASE 1: SINGLE RULE EXTRACTION')}")
        print(f"{blue('=' * 60)}")
        
        single_chains = self.gpu_engine.process_all_words_single_rule(
            base_words, target_words, rules
        )
        all_chains.extend(single_chains)
        
        print(f"{green('[OK]')} {bold('Working single rules found:')} {cyan(len(single_chains))}")
        
        # Phase 2: Generate and test chains up to depth 3 (standard approach)
        if max_depth >= 2 and single_chains:
            print(f"\n{blue('=' * 60)}")
            print(f"{bold('PHASE 2: STANDARD CHAIN GENERATION (DEPTH 2-3)')}")
            print(f"{blue('=' * 60)}")
            
            chains_to_test = self.rule_generator.generate_intelligent_chains(
                single_chains, min(3, max_depth), self.params
            )
            
            if chains_to_test:
                print(f"{blue('[INFO]')} {bold('Testing')} {len(chains_to_test):,} {bold('chains...')}")
                
                chain_results = self.gpu_engine.process_all_words_chain_rules(
                    base_words, target_words, rules, chains_to_test
                )
                
                if chain_results:
                    all_chains.extend(chain_results)
                    print(f"{green('[OK]')} {bold('Working chains found:')} {cyan(len(chain_results))}")
        
        # Phase 3: OPTIMIZED MITM for higher depths (4-6)
        if max_depth >= 4 and single_chains:
            print(f"\n{blue('=' * 60)}")
            print(f"{bold('PHASE 3: OPTIMIZED MEET-IN-THE-MIDDLE (DEPTH 4-6)')}")
            print(f"{blue('=' * 60)}")
            
            # Create optimized MITM finder
            mitm_finder = OptimizedMITMChainFinder(single_chains, self.validator)
            
            # Sample words for smart generation
            base_sample = random.sample(base_words, min(500, len(base_words)))
            target_sample = set(random.sample(target_words, min(5000, len(target_words))))
            
            for depth in [4, 5, 6]:
                if depth > max_depth:
                    continue
                    
                print(f"\n{blue('---')} {bold(f'Depth {depth} OPTIMIZED MITM')} {blue('---')}")
                
                half_depth = depth // 2
                other_half = depth - half_depth
                
                # Generate half chains with smart filtering
                left_chains = mitm_finder.generate_half_chains_smart(
                    half_depth, 
                    self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 10000) // 2,
                    base_sample,
                    target_sample
                )
                
                right_chains = mitm_finder.generate_half_chains_smart(
                    other_half,
                    self.params.get(f'CHAIN_GEN_LIMIT_{depth}', 10000) // 2,
                    base_sample,
                    target_sample
                )
                
                # Find combinations using optimized MITM
                mitm_results = mitm_finder.find_combinations_optimized(
                    left_chains, right_chains,
                    base_words, target_sample,
                    max_candidates=self.params['MITM_MAX_CANDIDATES']
                )
                
                if mitm_results:
                    print(f"{blue('[INFO]')} Found {len(mitm_results):,} potential depth {depth} chains")
                    
                    # Test found chains on GPU
                    chain_results = self.gpu_engine.process_all_words_chain_rules(
                        base_words, target_words, rules, mitm_results
                    )
                    
                    if chain_results:
                        all_chains.extend(chain_results)
                        print(f"{green('[OK]')} {bold(f'Working depth {depth} chains found:')} {cyan(len(chain_results))}")
        
        # Final cleanup
        print(f"\n{blue('=' * 60)}")
        print(f"{bold('FINAL CLEANUP')}")
        print(f"{blue('=' * 60)}")
        
        final_chains = self.validator.validate_rules_for_gpu(all_chains)
        
        return final_chains

# ====================================================================
# --- DYNAMIC PARAMETERS CALCULATION ---
# ====================================================================

def calculate_dynamic_parameters(base_count, target_count, device=None, target_hours=1.0):
    """Calculate dynamic parameters with support for higher depths"""
    
    BASE_BLOOM_SIZE = 1024 * 1024 * 8
    
    bloom_scale = max(1.0, math.log10(base_count + target_count) / 2.0)
    
    BLOOM_FILTER_SIZE = int(BASE_BLOOM_SIZE * bloom_scale)
    BLOOM_FILTER_SIZE = 1 << (BLOOM_FILTER_SIZE.bit_length() - 1)
    
    if device:
        try:
            max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
            max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
            global_mem = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
            
            possible_sizes = [32, 64, 128, 256, 512, 1024]
            LOCAL_WORK_SIZE = max([s for s in possible_sizes if s <= max_work_group_size])
            
            EST_COMBOS_PER_SEC = 50000000
            
            print(f"{blue('[GPU]')} {bold('Work Group Limits:')}")
            print(f"  {cyan('[*]')} Max work group size: {max_work_group_size}")
            print(f"  {cyan('[*]')} Compute units: {max_compute_units}")
            print(f"  {cyan('[*]')} Global memory: {global_mem // (1024**3)}GB")
            print(f"  {cyan('[*]')} Using work group size: {LOCAL_WORK_SIZE}")
            
        except:
            LOCAL_WORK_SIZE = 256
            EST_COMBOS_PER_SEC = 30000000
    else:
        LOCAL_WORK_SIZE = 256
        EST_COMBOS_PER_SEC = 30000000
    
    target_seconds = target_hours * 3600
    max_combinations_time_limit = int(EST_COMBOS_PER_SEC * target_seconds * 0.8)
    
    MAX_CHAINS_TO_FIND = min(2000000, max_combinations_time_limit // 1000)
    
    # Chain generation limits for depths 2-6
    if base_count > 1000000:
        CHAIN_GEN_LIMIT_2 = 200000
        CHAIN_GEN_LIMIT_3 = 100000
        CHAIN_GEN_LIMIT_4 = 50000
        CHAIN_GEN_LIMIT_5 = 25000
        CHAIN_GEN_LIMIT_6 = 10000
    elif base_count > 100000:
        CHAIN_GEN_LIMIT_2 = 150000
        CHAIN_GEN_LIMIT_3 = 75000
        CHAIN_GEN_LIMIT_4 = 35000
        CHAIN_GEN_LIMIT_5 = 15000
        CHAIN_GEN_LIMIT_6 = 5000
    else:
        CHAIN_GEN_LIMIT_2 = 75000
        CHAIN_GEN_LIMIT_3 = 35000
        CHAIN_GEN_LIMIT_4 = 15000
        CHAIN_GEN_LIMIT_5 = 5000
        CHAIN_GEN_LIMIT_6 = 2000
    
    print(f"\n{blue('[TIME]')} {bold(f'Target completion: {target_hours} hours')}")
    print(f"{blue('[PERF]')} {bold('Estimated processing speed:')} {cyan(f'{EST_COMBOS_PER_SEC:,}')} combos/sec")
    print(f"{blue('[PERF]')} {bold('Max combinations in time:')} {cyan(f'{max_combinations_time_limit:,}')}")
    
    return {
        'BLOOM_FILTER_SIZE': BLOOM_FILTER_SIZE,
        'MAX_CHAINS_TO_FIND': MAX_CHAINS_TO_FIND,
        'LOCAL_WORK_SIZE': LOCAL_WORK_SIZE,
        'CHAIN_GEN_LIMIT_2': CHAIN_GEN_LIMIT_2,
        'CHAIN_GEN_LIMIT_3': CHAIN_GEN_LIMIT_3,
        'CHAIN_GEN_LIMIT_4': CHAIN_GEN_LIMIT_4,
        'CHAIN_GEN_LIMIT_5': CHAIN_GEN_LIMIT_5,
        'CHAIN_GEN_LIMIT_6': CHAIN_GEN_LIMIT_6,
        'MITM_MAX_CANDIDATES': 200000,  # Limit MITM candidates
        'MITM_BEAM_WIDTH': 5000,
        'EST_COMBOS_PER_SEC': EST_COMBOS_PER_SEC,
        'TARGET_SECONDS': target_seconds,
        'MAX_CHAIN_DEPTH': 6,
    }

# ====================================================================
# --- GPU-COMPATIBLE KERNEL (with higher depth support) ---
# ====================================================================

GPU_COMPATIBLE_KERNEL = """
#define MAX_WORD_LEN 256
#define MAX_RULE_LEN 16
#define MAX_OUTPUT_LEN 512
#define MAX_CHAIN_STRING_LEN 512
#define MAX_CHAINS_TO_FIND 10000000
#define MAX_CHAIN_DEPTH 6
#define BLOOM_FILTER_SIZE 8388608
#define BLOOM_HASH_FUNCTIONS 3

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

inline int is_lower(unsigned char c) {
    return (c >= 'a' && c <= 'z');
}

inline int is_upper(unsigned char c) {
    return (c >= 'A' && c <= 'Z');
}

inline int is_digit(unsigned char c) {
    return (c >= '0' && c <= '9');
}

inline unsigned char to_lower(unsigned char c) {
    if (is_upper(c)) return c + 32;
    return c;
}

inline unsigned char to_upper(unsigned char c) {
    if (is_lower(c)) return c - 32;
    return c;
}

inline unsigned char toggle_case(unsigned char c) {
    if (is_lower(c)) return c - 32;
    if (is_upper(c)) return c + 32;
    return c;
}

// ============================================================================
// BLOOM FILTER FUNCTIONS
// ============================================================================

uint bloom_hash(const unsigned char *str, int len, uint seed) {
    uint hash = seed;
    for (int i = 0; i < len; i++) {
        hash = (hash * 31) + str[i];
    }
    return hash;
}

int bloom_check(__global const uchar *bloom_filter, const unsigned char *word, int len) {
    uint h1 = bloom_hash(word, len, 0xDEADBEEF);
    uint h2 = bloom_hash(word, len, 0xCAFEBABE);
    
    for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {
        uint hash_val = (h1 + i * h2) % BLOOM_FILTER_SIZE;
        uint byte_idx = hash_val / 8;
        uint bit_idx = hash_val % 8;
        
        if (!(bloom_filter[byte_idx] & (1 << bit_idx))) {
            return 0;
        }
    }
    return 1;
}

// ============================================================================
// GPU-COMPATIBLE RULE APPLICATION
// ============================================================================

int apply_gpu_rule(
    const unsigned char *rule_str, int rule_len,
    const unsigned char *input_word, int input_len,
    unsigned char *output_word, int *output_len
) {
    *output_len = input_len;
    for (int i = 0; i < input_len; i++) {
        output_word[i] = input_word[i];
    }
    output_word[input_len] = '\\0';
    
    if (rule_len == 0 || input_len == 0) return 1;
    
    unsigned char cmd = rule_str[0];
    int changed = 0;
    
    // SINGLE CHARACTER RULES
    if (rule_len == 1) {
        switch (cmd) {
            case 'l': for (int i = 0; i < *output_len; i++) output_word[i] = to_lower(output_word[i]); changed = 1; break;
            case 'u': for (int i = 0; i < *output_len; i++) output_word[i] = to_upper(output_word[i]); changed = 1; break;
            case 'c': if (*output_len > 0) { output_word[0] = to_upper(output_word[0]); for (int i = 1; i < *output_len; i++) output_word[i] = to_lower(output_word[i]); } changed = 1; break;
            case 'C': if (*output_len > 0) { output_word[0] = to_lower(output_word[0]); for (int i = 1; i < *output_len; i++) output_word[i] = to_upper(output_word[i]); } changed = 1; break;
            case 't': for (int i = 0; i < *output_len; i++) output_word[i] = toggle_case(output_word[i]); changed = 1; break;
            case 'r': for (int i = 0; i < *output_len / 2; i++) { unsigned char temp = output_word[i]; output_word[i] = output_word[*output_len - 1 - i]; output_word[*output_len - 1 - i] = temp; } changed = 1; break;
            case 'd': if (*output_len * 2 <= MAX_OUTPUT_LEN) { for (int i = 0; i < *output_len; i++) output_word[*output_len + i] = output_word[i]; *output_len *= 2; } changed = 1; break;
            case 'f': if (*output_len * 2 <= MAX_OUTPUT_LEN) { for (int i = 0; i < *output_len; i++) output_word[*output_len + i] = output_word[*output_len - 1 - i]; *output_len *= 2; } changed = 1; break;
            case '{': if (*output_len > 1) { unsigned char first = output_word[0]; for (int i = 0; i < *output_len - 1; i++) output_word[i] = output_word[i + 1]; output_word[*output_len - 1] = first; } changed = 1; break;
            case '}': if (*output_len > 1) { unsigned char last = output_word[*output_len - 1]; for (int i = *output_len - 1; i > 0; i--) output_word[i] = output_word[i - 1]; output_word[0] = last; } changed = 1; break;
            case '[': if (*output_len > 0) { for (int i = 0; i < *output_len - 1; i++) output_word[i] = output_word[i + 1]; (*output_len)--; } changed = 1; break;
            case ']': if (*output_len > 0) { (*output_len)--; } changed = 1; break;
            case 'k': if (*output_len >= 2) { unsigned char temp = output_word[0]; output_word[0] = output_word[1]; output_word[1] = temp; } changed = 1; break;
            case 'K': if (*output_len >= 2) { unsigned char temp = output_word[*output_len - 2]; output_word[*output_len - 2] = output_word[*output_len - 1]; output_word[*output_len - 1] = temp; } changed = 1; break;
            case ':': changed = 0; break;
            case 'q': if (*output_len * 2 <= MAX_OUTPUT_LEN) { unsigned char temp[MAX_OUTPUT_LEN]; for (int i = 0; i < *output_len; i++) temp[i] = output_word[i]; int idx = 0; for (int i = 0; i < *output_len; i++) { output_word[idx++] = temp[i]; output_word[idx++] = temp[i]; } *output_len *= 2; } changed = 1; break;
            case 'E': if (*output_len > 0) { int capitalize = 1; for (int i = 0; i < *output_len; i++) { if (capitalize && is_lower(output_word[i])) { output_word[i] = to_upper(output_word[i]); capitalize = 0; } if (output_word[i] == ' ' || output_word[i] == '-' || output_word[i] == '_') capitalize = 1; } } changed = 1; break;
        }
    }
    
    // TWO CHARACTER RULES
    else if (rule_len == 2) {
        unsigned char param = rule_str[1];
        
        if (cmd == '^' && *output_len + 1 <= MAX_OUTPUT_LEN) {
            for (int i = *output_len; i > 0; i--) output_word[i] = output_word[i - 1];
            output_word[0] = param; (*output_len)++; changed = 1;
        }
        else if (cmd == '$' && *output_len + 1 <= MAX_OUTPUT_LEN) {
            output_word[*output_len] = param; (*output_len)++; changed = 1;
        }
        else if (cmd == '@') {
            int new_len = 0;
            for (int i = 0; i < *output_len; i++) {
                if (output_word[i] != param) output_word[new_len++] = output_word[i];
                else changed = 1;
            }
            *output_len = new_len;
        }
        else if (cmd == 'T' && is_digit(param)) {
            int pos = param - '0';
            if (pos < *output_len) { output_word[pos] = toggle_case(output_word[pos]); changed = 1; }
        }
        else if (cmd == 'D' && is_digit(param)) {
            int pos = param - '0';
            if (pos < *output_len) {
                for (int i = pos; i < *output_len - 1; i++) output_word[i] = output_word[i + 1];
                (*output_len)--; changed = 1;
            }
        }
        else if (cmd == 'L' && is_digit(param)) {
            int pos = param - '0';
            if (pos < *output_len) {
                int new_len = 0;
                for (int i = pos; i < *output_len; i++) output_word[new_len++] = output_word[i];
                *output_len = new_len; changed = 1;
            }
        }
        else if (cmd == 'R' && is_digit(param)) {
            int pos = param - '0';
            if (pos < *output_len) { *output_len = pos + 1; changed = 1; }
        }
        else if (cmd == '+' && is_digit(param)) {
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] < 255) { output_word[pos]++; changed = 1; }
        }
        else if (cmd == '-' && is_digit(param)) {
            int pos = param - '0';
            if (pos < *output_len && output_word[pos] > 0) { output_word[pos]--; changed = 1; }
        }
        else if (cmd == 'z' && is_digit(param)) {
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {
                unsigned char first = output_word[0];
                for (int i = *output_len + n - 1; i >= n; i--) output_word[i] = output_word[i - n];
                for (int i = 0; i < n; i++) output_word[i] = first;
                *output_len += n; changed = 1;
            }
        }
        else if (cmd == 'Z' && is_digit(param)) {
            int n = param - '0';
            if (*output_len + n <= MAX_OUTPUT_LEN) {
                unsigned char last = output_word[*output_len - 1];
                for (int i = 0; i < n; i++) output_word[*output_len + i] = last;
                *output_len += n; changed = 1;
            }
        }
    }
    
    // THREE CHARACTER RULES
    else if (rule_len == 3) {
        unsigned char param1 = rule_str[1];
        unsigned char param2 = rule_str[2];
        
        if (cmd == 's') {
            for (int i = 0; i < *output_len; i++) {
                if (output_word[i] == param1) { output_word[i] = param2; changed = 1; }
            }
        }
        else if (cmd == 'i' && is_digit(param1)) {
            int pos = param1 - '0';
            if (pos <= *output_len && *output_len + 1 <= MAX_OUTPUT_LEN) {
                for (int i = *output_len; i > pos; i--) output_word[i] = output_word[i - 1];
                output_word[pos] = param2; (*output_len)++; changed = 1;
            }
        }
        else if (cmd == 'o' && is_digit(param1)) {
            int pos = param1 - '0';
            if (pos < *output_len) { output_word[pos] = param2; changed = 1; }
        }
        else if (cmd == 'x' && is_digit(param1) && is_digit(param2)) {
            int n = param1 - '0', m = param2 - '0';
            if (n > m) { int temp = n; n = m; m = temp; }
            if (n < *output_len) {
                int new_len = 0;
                for (int i = n; i <= m && i < *output_len; i++) output_word[new_len++] = output_word[i];
                *output_len = new_len; changed = 1;
            }
        }
        else if (cmd == '*' && is_digit(param1) && is_digit(param2)) {
            int n = param1 - '0', m = param2 - '0';
            if (n < *output_len && m < *output_len && n != m) {
                unsigned char temp = output_word[n];
                output_word[n] = output_word[m];
                output_word[m] = temp; changed = 1;
            }
        }
    }
    
    output_word[*output_len] = '\\0';
    return changed ? 1 : 0;
}

// KERNEL: FIND SINGLE RULES
__kernel void find_single_rules_gpu(
    __global const unsigned char *base_words,
    __global const int *base_offsets,
    __global const int *base_lengths,
    __global const unsigned char *rules,
    __global const int *rule_offsets,
    __global const int *rule_lengths,
    __global const uchar *bloom_filter,
    const int num_base_words,
    const int num_rules,
    __global char *found_rules,
    __global volatile int *found_count
) {
    int gid = get_global_id(0);
    int total_items = num_base_words * num_rules;
    
    if (gid >= total_items) return;
    
    int word_idx = gid / num_rules;
    int rule_idx = gid % num_rules;
    
    __global const unsigned char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];
    
    unsigned char input_word[MAX_WORD_LEN];
    for (int i = 0; i < word_len; i++) input_word[i] = word_ptr[i];
    input_word[word_len] = '\\0';
    
    __global const unsigned char *rule_ptr = rules + rule_offsets[rule_idx];
    int rule_len = rule_lengths[rule_idx];
    
    unsigned char rule_str[MAX_RULE_LEN];
    for (int i = 0; i < rule_len; i++) rule_str[i] = rule_ptr[i];
    rule_str[rule_len] = '\\0';
    
    unsigned char output_word[MAX_OUTPUT_LEN];
    int output_len;
    int result = apply_gpu_rule(rule_str, rule_len, input_word, word_len, output_word, &output_len);
    
    if (result > 0 && output_len > 0) {
        if (bloom_check(bloom_filter, output_word, output_len)) {
            int idx = atomic_inc(found_count);
            
            if (idx < MAX_CHAINS_TO_FIND) {
                __global char *output_ptr = found_rules + idx * MAX_CHAIN_STRING_LEN;
                for (int i = 0; i < rule_len && i < MAX_CHAIN_STRING_LEN - 1; i++) {
                    output_ptr[i] = rule_str[i];
                }
                output_ptr[rule_len] = '\\0';
            }
        }
    }
}

// KERNEL: FIND RULE CHAINS (supports up to depth 6)
__kernel void find_rule_chains_gpu(
    __global const unsigned char *base_words,
    __global const int *base_offsets,
    __global const int *base_lengths,
    __global const unsigned char *rules,
    __global const int *rule_offsets,
    __global const int *rule_lengths,
    __global const int *chain_sequences,
    __global const int *chain_depths,
    __global const uchar *bloom_filter,
    const int num_base_words,
    const int num_chains,
    const int max_chain_depth,
    __global char *found_chains,
    __global volatile int *found_count
) {
    int gid = get_global_id(0);
    int total_items = num_base_words * num_chains;
    
    if (gid >= total_items) return;
    
    int word_idx = gid / num_chains;
    int chain_idx = gid % num_chains;
    
    __global const unsigned char *word_ptr = base_words + base_offsets[word_idx];
    int word_len = base_lengths[word_idx];
    
    unsigned char current_word[MAX_OUTPUT_LEN];
    for (int i = 0; i < word_len; i++) current_word[i] = word_ptr[i];
    current_word[word_len] = '\\0';
    int current_len = word_len;
    
    int depth = chain_depths[chain_idx];
    if (depth < 1 || depth > max_chain_depth) return;
    
    unsigned char temp_word[MAX_OUTPUT_LEN];
    char chain_buffer[MAX_CHAIN_STRING_LEN];
    int chain_pos = 0;
    
    for (int d = 0; d < depth; d++) {
        int rule_idx = chain_sequences[chain_idx * max_chain_depth + d];
        if (rule_idx < 0) break;
        
        __global const unsigned char *rule_ptr = rules + rule_offsets[rule_idx];
        int rule_len = rule_lengths[rule_idx];
        
        unsigned char rule_str[MAX_RULE_LEN];
        for (int i = 0; i < rule_len; i++) rule_str[i] = rule_ptr[i];
        rule_str[rule_len] = '\\0';
        
        for (int i = 0; i < rule_len && chain_pos < MAX_CHAIN_STRING_LEN - 2; i++) {
            chain_buffer[chain_pos++] = rule_str[i];
        }
        if (d < depth - 1 && chain_pos < MAX_CHAIN_STRING_LEN - 1) {
            chain_buffer[chain_pos++] = ' ';
        }
        
        int new_len;
        int result = apply_gpu_rule(rule_str, rule_len, current_word, current_len, temp_word, &new_len);
        
        if (result <= 0 || new_len == 0) return;
        
        for (int i = 0; i < new_len; i++) current_word[i] = temp_word[i];
        current_word[new_len] = '\\0';
        current_len = new_len;
    }
    
    chain_buffer[chain_pos] = '\\0';
    
    if (bloom_check(bloom_filter, current_word, current_len)) {
        int idx = atomic_inc(found_count);
        
        if (idx < MAX_CHAINS_TO_FIND) {
            __global char *output_ptr = found_chains + idx * MAX_CHAIN_STRING_LEN;
            for (int i = 0; i < chain_pos && i < MAX_CHAIN_STRING_LEN - 1; i++) {
                output_ptr[i] = chain_buffer[i];
            }
            output_ptr[chain_pos] = '\\0';
        }
    }
}
"""

# ====================================================================
# --- UTILITY FUNCTIONS ---
# ====================================================================

def load_wordlist_fast(filename):
    """Fast wordlist loading with statistics"""
    words = set()
    print(f"{blue('[LOAD]')} {bold('Loading:')} {filename}")
    
    try:
        with open(filename, 'r', encoding='latin-1', errors='ignore') as f:
            for line in tqdm(f, desc="Loading words"):
                word = line.strip()
                if word and len(word) <= MAX_WORD_LEN:
                    words.add(word)
    except FileNotFoundError:
        print(f"{red('[ERROR]')} {bold('FATAL ERROR:')} Wordlist not found: {filename}")
        sys.exit(1)
    
    words_list = list(words)
    
    avg_len = sum(len(w) for w in words_list) / len(words_list) if words_list else 0
    max_len = max(len(w) for w in words_list) if words_list else 0
    
    print(f"{green('[OK]')} {bold('Loaded:')} {cyan(f'{len(words_list):,}')} {bold('words')}")
    print(f"{blue('[INFO]')} {bold('Average length:')} {cyan(f'{avg_len:.1f}')}")
    print(f"{blue('[INFO]')} {bold('Max length:')} {cyan(f'{max_len}')}")
    
    return words_list

# ====================================================================
# --- MAIN EXECUTION ---
# ====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"{bold('GPU-COMPATIBLE Hashcat Rules Engine with OPTIMIZED MITM Support')}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('base_wordlist', help='Base wordlist path')
    parser.add_argument('target_wordlist', help='Target wordlist path')
    parser.add_argument('-d', '--depth', type=int, default=6, 
                       choices=[1, 2, 3, 4, 5, 6],
                       help='Max chain depth (1-6, default: 6)')
    parser.add_argument('-o', '--output', type=str, default='found_chains.txt', 
                       help='Output file (default: found_chains.txt)')
    parser.add_argument('--slow', action='store_true', 
                       help='Enable more comprehensive search (may take longer)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify all chains work before saving')
    parser.add_argument('--max-chains', type=int, default=None,
                       help='Maximum chains to generate (overrides automatic limits)')
    parser.add_argument('--target-hours', type=float, default=1.0,
                       help='Target completion time in hours (default: 1.0)')
    parser.add_argument('--mitm', action='store_true', default=True,
                       help='Enable Meet-in-the-Middle for depths 4-6 (default: True)')
    parser.add_argument('--mitm-candidates', type=int, default=200000,
                       help='Maximum MITM candidates before GPU testing (default: 200000)')
    
    args = parser.parse_args()

    print(f"\n{bold(green('=' * 80))}")
    print(f"{bold('GPU-COMPATIBLE HASHCAT RULES ENGINE (OPTIMIZED MITM)')}")
    print(f"{bold(green('=' * 80))}{Colors.END}\n")
    
    # Load data
    print(f"{blue('[INIT]')} {bold('Loading data...')}")
    
    base_words = load_wordlist_fast(args.base_wordlist)
    target_words = load_wordlist_fast(args.target_wordlist)
    
    print(f"\n{blue('[ANALYSIS]')} {bold('Dataset Analysis:')}")
    print(f"  {cyan('[*]')} Base words: {len(base_words):,}")
    print(f"  {cyan('[*]')} Target words: {len(target_words):,}")
    print(f"  {cyan('[*]')} Target completion: {args.target_hours} hours")
    print(f"  {cyan('[*]')} Max depth: {args.depth}")
    print(f"  {cyan('[*]')} MITM: {'Enabled' if args.mitm else 'Disabled'}")
    print(f"  {cyan('[*]')} MITM candidates: {args.mitm_candidates:,}")
    
    start_time = time.time()
    
    # Get GPU device
    try:
        platform = cl.get_platforms()[0]
        device = platform.get_devices(cl.device_type.GPU)[0]
    except:
        device = None
    
    # Initialize GPU extractor
    extractor = GPUExtractor(len(base_words), len(target_words), device, args.slow, args.target_hours)
    
    if args.max_chains:
        extractor.params['MAX_CHAINS_TO_FIND'] = args.max_chains
        print(f"{blue('[OVERRIDE]')} {bold('Max chains set to:')} {cyan(args.max_chains)}")
    
    if args.mitm_candidates:
        extractor.params['MITM_MAX_CANDIDATES'] = args.mitm_candidates
    
    # Run extraction
    print(f"\n{blue('=' * 60)}")
    print(f"{bold('STARTING GPU-COMPATIBLE RULE EXTRACTION')}")
    print(f"{blue('=' * 60)}")
    
    chains = extractor.extract_rules(base_words, target_words, args.depth)
    
    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    
    # Save results
    print(f"\n{blue('[SAVE]')} {bold('Saving results...')}")
    
    final_chains = HashcatRuleValidator.validate_rules_for_gpu(chains)
    
    with open(args.output, 'w', encoding='latin-1') as f:
        for chain in final_chains:
            f.write(f"{chain}\n")
    
    # Print summary
    print(f"\n{bold(green('=' * 80))}")
    print(f"{bold('FINAL RESULTS')}")
    print(f"{bold(green('=' * 80))}")
    print(f"{blue('[INFO]')} {bold('Base words:')} {cyan(f'{len(base_words):,}')}")
    print(f"{blue('[INFO]')} {bold('Target words:')} {cyan(f'{len(target_words):,}')}")
    print(f"{blue('[INFO]')} {bold('Max depth:')} {cyan(f'{args.depth}')}")
    print(f"{blue('[INFO]')} {bold('Total time:')} {cyan(f'{elapsed_hours:.2f} hours ({end_time - start_time:.2f}s)')}")
    print(f"{green('[RESULT]')} {bold('GPU-validated chains found:')} {cyan(f'{len(final_chains):,}')}")
    
    if final_chains:
        print(f"{blue('[SAMPLE]')} {bold('Sample chains (max 20):')}")
        chains_by_depth = defaultdict(list)
        for chain in final_chains[:100]:
            depth = len(chain.split())
            chains_by_depth[depth].append(chain)
        
        shown = 0
        for depth in sorted(chains_by_depth.keys()):
            for chain in chains_by_depth[depth][:5]:
                print(f"  {cyan(f'{shown+1:2d}.')} [d{depth}] {chain}")
                shown += 1
                if shown >= 20:
                    break
            if shown >= 20:
                break
    
    print(f"{blue('[OUTPUT]')} {bold('Output saved to:')} {bold(args.output)}")
    print(f"{bold(green('=' * 80))}{Colors.END}")

