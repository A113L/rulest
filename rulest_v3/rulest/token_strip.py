"""
Stage 0 (--token-strip): cheap CPU pre-pass that tries to explain each
target word as a base-wordlist word plus a short, exact-match transform
(case toggling, leet substitution, reversal, duplication, edge deletion,
insertion, prefix/suffix boundary stripping) before any GPU work happens.
Runs across a multiprocessing pool for throughput on large target lists.
"""
import os
import time
import shutil
import itertools
import multiprocessing as mp
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

def _extract_insert_mode(word, base_set, max_depth, min_len, max_amb, base_by_len):
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

def _worker_init_p0(base_set=None, base_by_len=None):
    if base_set is not None: worker_state.p0_worker_base_set = base_set
    if base_by_len is not None: worker_state.p0_worker_base_by_len = base_by_len

def _process_chunk_p0(args):
    words, max_depth, min_len, max_amb, max_pre, max_suf = args
    base_set = worker_state.p0_worker_base_set  # read-only, no copy needed
    base_by_len = worker_state.p0_worker_base_by_len
    found = set()
    for w in words:
        if not w or len(w)>MAX_WORD_LEN: continue
        nd = sum(1 for c in w if c.isdigit())
        na = sum(1 for c in w if c.isalpha())
        if nd > na:
            found |= _extract_digit_mode(w, base_set, max_depth, min_len, max_pre, max_suf)
            if na >= min_len:
                found |= _extract_letter_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
        else:
            found |= _extract_letter_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
            if nd>0:
                found |= _extract_digit_mode(w, base_set, max_depth, min_len, max_pre, max_suf)
        if max_depth >= 2:
            found |= _extract_reverse_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
            found |= _extract_delete_edge_mode(w, base_set, max_depth, min_len, max_pre, max_suf, max_amb)
        if len(w) >= 2*min_len:
            found |= _extract_duplicate_mode(w, base_set, max_depth, min_len)
        if max_depth >= 2:
            found |= _extract_insert_mode(w, base_set, max_depth, min_len, max_amb, base_by_len)
    return found

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
                              workers=0, chunk_size=0):
    if max_depth<=0: max_depth=MAX_HASHCAT_CHAIN
    n_workers = safe_worker_count(workers or mp.cpu_count())
    n_words = len(target_words)
    if chunk_size<=0: chunk_size = max(500, n_words//(n_workers*4)+1)
    base_by_len = defaultdict(set)
    for w in base_set: base_by_len[len(w)].add(w)
    base_by_len = dict(base_by_len)
    total_bytes = sum(len(w) for w in target_words)
    DISK_THRESHOLD_BYTES = 150 * 1024 * 1024
    if total_bytes > DISK_THRESHOLD_BYTES:
        log_warn(f"[S0] Target wordlist size {total_bytes/1e6:.0f}MB > {DISK_THRESHOLD_BYTES/1e6:.0f}MB — using single worker & disk buffer")
        n_workers = 1
    worker_state.p0_worker_base_set = base_set
    worker_state.p0_worker_base_by_len = base_by_len
    chunks = [target_words[i:i+chunk_size] for i in range(0,n_words,chunk_size)]
    task_args = [(ch, max_depth, min_stem_len, max_leet_ambiguity, max_prefix_len, max_suffix_len) for ch in chunks]
    log_info(f"[S0]    Workers: {bold(str(n_workers))}  |  chunks: {bold(str(len(chunks)))} × ~{chunk_size}")
    found = set()
    use_fork = hasattr(os,'fork')
    ctx = mp.get_context('fork' if use_fork else 'spawn')
    if use_fork: pool_kw = dict(processes=n_workers, initializer=_worker_init_p0)
    else: pool_kw = dict(processes=n_workers, initializer=_worker_init_p0, initargs=(base_set, base_by_len))
    ncols = shutil.get_terminal_size((80,24)).columns
    with ctx.Pool(**pool_kw) as pool:
        with tqdm(total=n_words, desc=green("  STAGE 0 "), unit="word", ncols=ncols,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            t0 = time.time()
            words_done = 0
            for task, res in zip(task_args, pool.imap_unordered(_process_chunk_p0, task_args)):
                if _kb.poll():
                    log_warn("[S0] Early exit — keeping rules found so far")
                    pool.terminate()
                    break
                found |= res
                words_done += len(task[0])
                elapsed = time.time() - t0
                spd = _fmt_speed(words_done / elapsed if elapsed>0 else 0, "words")
                pbar.set_postfix({"rules": cyan(str(len(found))), "spd": green(spd)}, refresh=False)
                pbar.update(len(task[0]))
    worker_state.p0_worker_base_set.clear()
    worker_state.p0_worker_base_by_len.clear()
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
