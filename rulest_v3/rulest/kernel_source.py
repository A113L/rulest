"""
The full space of atomic GPU-compatible hashcat rules
(GPUCompatibleRulesGenerator), and the OpenCL C kernel template
(GPU_KERNEL_TEMPLATE) that implements the same rule DSL on-device: single
bloom-filter membership test, single-rule sweep, and rule-chain search
kernels. Kept as one module since the generator's rule-space directly
mirrors what the kernel below is able to execute.
"""
import string

from .common import log_info, HashcatRuleValidator, POSITION_CHARS, MAX_RULE_LEN

class GPUCompatibleRulesGenerator:
    def __init__(self): self.validator = HashcatRuleValidator()
    def generate_gpu_compatible_rules(self):
        rules = set()
        digits = '0123456789'
        rules.update(['l','u','c','C','t','r','d','f','q','E','{','}','[',']','k','K',':'])
        for cmd in ('T','D','L','R','+','-','.',',',"'",'z','Z','y','Y'):
            for p in POSITION_CHARS: rules.add(f'{cmd}{p}')
        for p in POSITION_CHARS: rules.add(f'p{p}')
        for cmd in ('x','*','O'):
            for p1 in digits:
                for p2 in digits: rules.add(f'{cmd}{p1}{p2}')
        for i in range(33,127):
            ch = chr(i); rules.add(f'^{ch}'); rules.add(f'${ch}'); rules.add(f'@{ch}')
        for orig in string.ascii_lowercase+string.ascii_uppercase:
            for sub in string.digits+string.punctuation:
                if orig != sub: rules.add(f's{orig}{sub}')
        chars = string.ascii_letters+string.digits+'!@#$%^&*()_+-=[]{}|;:,.<>?/~'
        for p in digits:
            for ch in chars: rules.add(f'i{p}{ch}'); rules.add(f'o{p}{ch}')
        for n in range(1,10):
            for ch in ('p','y','Y','z','Z'): rules.add(f'{ch}{n}')
        for sep in '-_.,;:|/\\+*&^%$#@!~`': rules.add(f'e{sep}')
        chars2 = string.ascii_letters+string.digits+'!@#$%^&*()_+-=[]{}|;:,.<>?/~'
        for n in digits:
            for sep in chars2: rules.add(f'3{n}{sep}')
        valid = [r for r in rules if self.validator.validate_rule_for_gpu(r) and 1<=len(r)<=MAX_RULE_LEN]
        log_info(f"Generated {len(valid):,} atomic rules")
        return valid

GPU_KERNEL_TEMPLATE = r"""
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define MAX_WORD_LEN         {MAX_WORD_LEN}
#define MAX_RULE_LEN         {MAX_RULE_LEN}
#define MAX_OUTPUT_LEN       {MAX_OUTPUT_LEN}
#define MAX_CHAIN_STRING_LEN {MAX_CHAIN_STRING_LEN}
#define MAX_CHAINS_TO_FIND   {MAX_SAFE_RESULTS_PER_BATCH}
#define MAX_CHAIN_DEPTH      {MAX_CHAIN_DEPTH}
#define BLOOM_HASH_FUNCTIONS {BLOOM_HASH_FUNCTIONS}
#define BLOOM_NUM_SHARDS     {BLOOM_NUM_SHARDS}
#define BLOOM_SHARD_BITS     {BLOOM_SHARD_BITS}
#define BLOOM_SHARD_BYTES    {BLOOM_SHARD_BYTES}

inline uint bloom_shard_idx(uint h1, uint h2, uint i, uint shard_bits) {{
    return (h1 + i * h2) & (shard_bits - 1);
}}

int bloom(__global const uchar *bf, const unsigned char *w, int len) {{
    for (int s = 0; s < BLOOM_NUM_SHARDS; s++) {{
        uint seed1 = 0xDEADBEEFu + (uint)s;
        uint seed2 = 0xCAFEBABEu + (uint)s;
        uint h1 = seed1 ^ 2166136261U;
        uint h2 = seed2 ^ 2166136261U;
        for (int i = 0; i < len; i++) {{
            h1 ^= w[i]; h1 *= 16777619U;
            h2 ^= w[i]; h2 *= 16777619U;
        }}
        for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {{
            uint idx = bloom_shard_idx(h1, h2, i, BLOOM_SHARD_BITS);
            uint byte_off = s * BLOOM_SHARD_BYTES + (idx >> 3);
            if (!(bf[byte_off] & (1 << (idx & 7)))) return 0;
        }}
    }}
    return 1;
}}

// Decodes hashcat's position/count encoding: '0'-'9' -> 0-9, 'A'-'Z' -> 10-35, else -1.
inline int dpos(unsigned char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return -1;
}}

int apply(const unsigned char *rs, int rl,
          const unsigned char *in, int il,
          unsigned char *out, int *ol) {{
    *ol = il; for (int i = 0; i < il; i++) out[i] = in[i]; out[il] = '\0';
    if (!rl || !il) return 1;
    unsigned char cmd = rs[0]; int changed = 0;
    if (rl == 1) {{
        switch (cmd) {{
        case 'l': for (int i = 0; i < *ol; i++) out[i] = (out[i] >= 'A' && out[i] <= 'Z') ? out[i] + 32 : out[i]; changed = 1; break;
        case 'u': for (int i = 0; i < *ol; i++) out[i] = (out[i] >= 'a' && out[i] <= 'z') ? out[i] - 32 : out[i]; changed = 1; break;
        case 'c': if (*ol > 0) {{
            out[0] = (out[0] >= 'a' && out[0] <= 'z') ? out[0] - 32 : out[0];
            for (int i = 1; i < *ol; i++) out[i] = (out[i] >= 'A' && out[i] <= 'Z') ? out[i] + 32 : out[i];
        }} changed = 1; break;
        case 'C': if (*ol > 0) {{
            out[0] = (out[0] >= 'A' && out[0] <= 'Z') ? out[0] + 32 : out[0];
            for (int i = 1; i < *ol; i++) out[i] = (out[i] >= 'a' && out[i] <= 'z') ? out[i] - 32 : out[i];
        }} changed = 1; break;
        case 't': for (int i = 0; i < *ol; i++) {{
            if (out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
            else if (out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
        }} changed = 1; break;
        case 'r': for (int i = 0; i < *ol/2; i++) {{
            unsigned char t = out[i]; out[i] = out[*ol-1-i]; out[*ol-1-i] = t;
        }} changed = 1; break;
        case 'd': if (*ol*2 <= MAX_OUTPUT_LEN) {{
            for (int i = 0; i < *ol; i++) out[*ol+i] = out[i]; *ol *= 2;
        }} changed = 1; break;
        case 'f': if (*ol*2 <= MAX_OUTPUT_LEN) {{
            for (int i = 0; i < *ol; i++) out[*ol+i] = out[*ol-1-i]; *ol *= 2;
        }} changed = 1; break;
        case '{{': if (*ol > 1) {{
            unsigned char f = out[0];
            for (int i = 0; i < *ol-1; i++) out[i] = out[i+1];
            out[*ol-1] = f;
        }} changed = 1; break;
        case '}}': if (*ol > 1) {{
            unsigned char l = out[*ol-1];
            for (int i = *ol-1; i > 0; i--) out[i] = out[i-1];
            out[0] = l;
        }} changed = 1; break;
        case '[': if (*ol > 0) {{
            for (int i = 0; i < *ol-1; i++) out[i] = out[i+1]; (*ol)--;
        }} changed = 1; break;
        case ']': if (*ol > 0) (*ol)--; changed = 1; break;
        case 'k': if (*ol >= 2) {{
            unsigned char t = out[0]; out[0] = out[1]; out[1] = t;
        }} changed = 1; break;
        case 'K': if (*ol >= 2) {{
            unsigned char t = out[*ol-2]; out[*ol-2] = out[*ol-1]; out[*ol-1] = t;
        }} changed = 1; break;
        case ':': break;
        case 'q': if (*ol*2 <= MAX_OUTPUT_LEN) {{
            unsigned char tmp[MAX_OUTPUT_LEN];
            for (int i = 0; i < *ol; i++) tmp[i] = out[i];
            int idx = 0;
            for (int i = 0; i < *ol; i++) {{
                out[idx++] = tmp[i]; out[idx++] = tmp[i];
            }} *ol *= 2;
        }} changed = 1; break;
        case 'E': {{
            int cap = 1;
            for (int i = 0; i < *ol; i++) {{
                if (cap && out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                else if (!cap && out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                cap = (out[i] == ' ');
            }} changed = 1;
        }} break;
        }}
    }} else if (rl == 2) {{
        unsigned char p = rs[1];
        int pv = dpos(p);
        if (cmd == '^') {{
            if (*ol+1 <= MAX_OUTPUT_LEN) {{
                for (int i = *ol; i > 0; i--) out[i] = out[i-1];
                out[0] = p; (*ol)++;
            }} changed = 1;
        }} else if (cmd == '$') {{
            if (*ol+1 <= MAX_OUTPUT_LEN) {{
                out[*ol] = p; (*ol)++;
            }} changed = 1;
        }} else if (cmd == '@') {{
            int nl = 0;
            for (int i = 0; i < *ol; i++) if (out[i] != p) out[nl++] = out[i];
            *ol = nl; changed = 1;
        }} else if (cmd == 'p' && pv >= 0) {{
            int n = pv;
            if (n > 0 && *ol*(n+1) <= MAX_OUTPUT_LEN) {{
                int o = *ol;
                for (int r = 0; r < n; r++) {{
                    for (int i = 0; i < o; i++) out[*ol+i] = out[i];
                    *ol += o;
                }} changed = 1;
            }}
        }} else if (cmd == 'T' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{
                if (out[pos] >= 'a' && out[pos] <= 'z') out[pos] -= 32;
                else if (out[pos] >= 'A' && out[pos] <= 'Z') out[pos] += 32;
                changed = 1;
            }}
        }} else if (cmd == 'D' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{
                for (int i = pos; i < *ol-1; i++) out[i] = out[i+1];
                (*ol)--; changed = 1;
            }}
        }} else if (cmd == 'L' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{ out[pos] <<= 1; changed = 1; }}
        }} else if (cmd == 'R' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{ out[pos] >>= 1; changed = 1; }}
        }} else if (cmd == '+' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol && out[pos] < 255) {{ out[pos]++; changed = 1; }}
        }} else if (cmd == '-' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol && out[pos] > 0) {{ out[pos]--; changed = 1; }}
        }} else if ((cmd == '.' || cmd == ',') && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{
                out[pos] += (cmd == '.') ? 1 : -1; changed = 1;
            }}
        }} else if (cmd == '\'' && pv >= 0) {{
            int pos = pv;
            if (pos < *ol) {{ *ol = pos; changed = 1; }}
        }} else if (cmd == 'z' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                unsigned char f = out[0];
                for (int i = *ol+n-1; i >= n; i--) out[i] = out[i-n];
                for (int i = 0; i < n; i++) out[i] = f;
                *ol += n; changed = 1;
            }}
        }} else if (cmd == 'Z' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                unsigned char l = out[*ol-1];
                for (int i = 0; i < n; i++) out[*ol+i] = l;
                *ol += n; changed = 1;
            }}
        }} else if (cmd == 'y' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                for (int i = 0; i < n; i++) out[*ol+i] = out[i];
                *ol += n; changed = 1;
            }}
        }} else if (cmd == 'Y' && pv >= 0) {{
            int n = pv;
            if (n>0 && *ol+n <= MAX_OUTPUT_LEN) {{
                for (int i = 0; i < n; i++) out[*ol+i] = out[*ol-n+i];
                *ol += n; changed = 1;
            }}
        }}
    }} else if (rl == 3) {{
        unsigned char p1 = rs[1], p2 = rs[2];
        if (cmd == 's') {{
            for (int i = 0; i < *ol; i++) if (out[i] == p1) {{ out[i] = p2; changed = 1; }}
        }} else if (cmd == 'i' && p1>='0' && p1<='9') {{
            int pos = p1-'0';
            if (pos <= *ol && *ol+1 <= MAX_OUTPUT_LEN) {{
                for (int i = *ol; i > pos; i--) out[i] = out[i-1];
                out[pos] = p2; (*ol)++; changed = 1;
            }}
        }} else if (cmd == 'o' && p1>='0' && p1<='9') {{
            int pos = p1-'0';
            if (pos < *ol) {{ out[pos] = p2; changed = 1; }}
        }} else if (cmd == 'e') {{
            int cap = 1;
            for (int i = 0; i < *ol; i++) {{
                if (cap && out[i] >= 'a' && out[i] <= 'z') out[i] -= 32;
                else if (!cap && out[i] >= 'A' && out[i] <= 'Z') out[i] += 32;
                cap = (out[i] == p1);
            }} changed = 1;
        }} else if (cmd == 'x' && p1>='0' && p1<='9' && p2>='0' && p2<='9') {{
            int a = p1-'0', b = p2-'0';
            if (a >= 0 && b >= 0 && a < *ol) {{
                int nl = 0;
                int end = a + b;
                for (int i = a; i < end && i < *ol; i++) out[nl++] = out[i];
                *ol = nl; changed = 1;
            }}
        }} else if (cmd == 'O' && p1>='0' && p1<='9' && p2>='0' && p2<='9') {{
            int n = p1-'0', m = p2-'0';
            if (n < *ol && m > 0) {{
                int e = n+m; if (e > *ol) e = *ol;
                int sh = e - n;
                for (int i = e; i < *ol; i++) out[i-sh] = out[i];
                *ol -= sh; changed = 1;
            }}
        }} else if (cmd == '*' && p1>='0' && p1<='9' && p2>='0' && p2<='9') {{
            int a = p1-'0', b = p2-'0';
            if (a < *ol && b < *ol && a != b) {{
                unsigned char t = out[a]; out[a] = out[b]; out[b] = t; changed = 1;
            }}
        }} else if (cmd == '3' && p1>='0' && p1<='9') {{
            int n = p1-'0', cnt = 0, found = -1;
            for (int i = 0; i < *ol; i++) if (out[i] == p2 && ++cnt == n + 1) {{ found = i; break; }}
            if (found != -1 && found+1 < *ol) {{
                if (out[found+1] >= 'a' && out[found+1] <= 'z') out[found+1] -= 32;
                else if (out[found+1] >= 'A' && out[found+1] <= 'Z') out[found+1] += 32;
                changed = 1;
            }}
        }}
    }}
    out[*ol] = '\0';
    return changed ? 1 : 0;
}}

__kernel void find_single_rules_gpu(
    __global const unsigned char *bw, __global const int *bo, __global const int *bl,
    __global const unsigned char *rs, __global const int *ro, __global const int *rl,
    __global const uchar *bf, const int nw, const int nr,
    __global char *found, __global volatile int *cnt)
{{
    int gid = get_global_id(0);
    if (gid >= nw * nr) return;
    int wi = gid / nr, ri = gid % nr;
    unsigned char iw[MAX_WORD_LEN], ow[MAX_OUTPUT_LEN], rr[MAX_RULE_LEN];
    int wl = bl[wi]; for (int i = 0; i < wl; i++) iw[i] = bw[bo[wi] + i]; iw[wl] = '\0';
    int rlen = rl[ri]; for (int i = 0; i < rlen; i++) rr[i] = rs[ro[ri] + i]; rr[rlen] = '\0';
    int ol; apply(rr, rlen, iw, wl, ow, &ol);
    int same = (ol == wl); for (int i = 0; i < wl && same; i++) if (ow[i] != iw[i]) same = 0;
    if (!same && ol > 0 && bloom(bf, ow, ol)) {{
        int idx = atomic_inc(cnt);
        if (idx < MAX_CHAINS_TO_FIND) {{
            __global char *p = found + idx * MAX_CHAIN_STRING_LEN;
            for (int i = 0; i < rlen && i < MAX_CHAIN_STRING_LEN-1; i++) p[i] = rr[i];
            p[rlen] = '\0';
        }}
    }}
}}

__kernel void find_rule_chains_gpu(
    __global const unsigned char *bw, __global const int *bo, __global const int *bl,
    __global const unsigned char *rs, __global const int *ro, __global const int *rl,
    __global const int *cseq, __global const int *cdep,
    __global const uchar *bf, const int nw, const int nc, const int mcd,
    __global char *found, __global volatile int *cnt, __global int *foundw)
{{
    int gid = get_global_id(0);
    if (gid >= nw * nc) return;
    int wi = gid / nc, ci = gid % nc;
    unsigned char cur[MAX_OUTPUT_LEN], tmp[MAX_OUTPUT_LEN], rr[MAX_RULE_LEN];
    char cb[MAX_CHAIN_STRING_LEN]; int cp = 0;
    int wl = bl[wi]; for (int i = 0; i < wl; i++) cur[i] = bw[bo[wi] + i]; cur[wl] = '\0';
    int cl_ = wl, dep = cdep[ci]; if (dep < 1 || dep > mcd) return;
    __global const unsigned char *wp = bw + bo[wi];
    for (int d = 0; d < dep; d++) {{
        int ri = cseq[ci * mcd + d]; if (ri < 0) break;
        int rlen = rl[ri]; for (int i = 0; i < rlen; i++) rr[i] = rs[ro[ri] + i]; rr[rlen] = '\0';
        for (int i = 0; i < rlen && cp < MAX_CHAIN_STRING_LEN-2; i++) cb[cp++] = rr[i];
        if (d < dep-1 && cp < MAX_CHAIN_STRING_LEN-1) cb[cp++] = ' ';
        int nl; apply(rr, rlen, cur, cl_, tmp, &nl);
        if (nl == 0) return;
        for (int i = 0; i < nl; i++) cur[i] = tmp[i]; cur[nl] = '\0'; cl_ = nl;
    }}
    cb[cp] = '\0';
    int same = (cl_ == wl); for (int i = 0; i < wl && same; i++) if (cur[i] != wp[i]) same = 0;
    if (!same && bloom(bf, cur, cl_)) {{
        int idx = atomic_inc(cnt);
        if (idx < MAX_CHAINS_TO_FIND) {{
            __global char *p = found + idx * MAX_CHAIN_STRING_LEN;
            for (int i = 0; i < cp && i < MAX_CHAIN_STRING_LEN-1; i++) p[i] = cb[i];
            p[cp] = '\0';
            foundw[idx] = wi;
        }}
    }}
}}

__kernel void build_bloom_filter_gpu(
    __global const unsigned char *bw, __global const int *bo, __global const int *bl,
    const int nw, volatile __global int *bf)
{{
    int gid = get_global_id(0);
    if (gid >= nw) return;
    int wlen = bl[gid];
    __global const unsigned char *word = bw + bo[gid];
    for (int s = 0; s < BLOOM_NUM_SHARDS; s++) {{
        uint seed1 = 0xDEADBEEFu + (uint)s;
        uint seed2 = 0xCAFEBABEu + (uint)s;
        uint h1 = seed1 ^ 2166136261U;
        uint h2 = seed2 ^ 2166136261U;
        for (int i = 0; i < wlen; i++) {{
            h1 ^= word[i]; h1 *= 16777619U;
            h2 ^= word[i]; h2 *= 16777619U;
        }}
        for (int i = 0; i < BLOOM_HASH_FUNCTIONS; i++) {{
            uint idx = (h1 + i * h2) & (BLOOM_SHARD_BITS - 1);
            uint widx = (s * (BLOOM_SHARD_BYTES / 4)) + (idx >> 5);
            int mask = 1 << (idx & 31);
            atomic_or(bf + widx, mask);
        }}
    }}
}}

// --exact-recovery verification kernel. Applies the FULL rule chain
// cseq[pair_chain_idx[gid]] to word pair_word_idx[gid] using the SAME
// per-token apply() loop as find_rule_chains_gpu above (unmodified — no new
// rule semantics here, just chain application), and unconditionally writes
// the resulting string + length to host-visible buffers. No bloom test, no
// atomic counter: meant to be called only on the small (chain, word) pairs
// already flagged as bloom hits during selection, so every gid is a real
// result the host wants to see, not a candidate that might not match.
__kernel void verify_pairs_gpu(
    __global const unsigned char *bw, __global const int *bo, __global const int *bl,
    __global const unsigned char *rs, __global const int *ro, __global const int *rl,
    __global const int *cseq, __global const int *cdep,
    __global const int *pair_word_idx, __global const int *pair_chain_idx,
    const int npairs, const int mcd,
    __global unsigned char *out, __global int *out_len)
{{
    int gid = get_global_id(0);
    if (gid >= npairs) return;
    int wi = pair_word_idx[gid], ci = pair_chain_idx[gid];
    unsigned char cur[MAX_OUTPUT_LEN], tmp[MAX_OUTPUT_LEN], rr[MAX_RULE_LEN];
    int wl = bl[wi]; for (int i = 0; i < wl; i++) cur[i] = bw[bo[wi] + i]; cur[wl] = '\0';
    int cl_ = wl, dep = cdep[ci];
    for (int d = 0; d < dep; d++) {{
        int ri = cseq[ci * mcd + d]; if (ri < 0) break;
        int rlen = rl[ri]; for (int i = 0; i < rlen; i++) rr[i] = rs[ro[ri] + i]; rr[rlen] = '\0';
        int nl; apply(rr, rlen, cur, cl_, tmp, &nl);
        for (int i = 0; i < nl; i++) cur[i] = tmp[i]; cur[nl] = '\0'; cl_ = nl;
    }}
    if (cl_ > MAX_OUTPUT_LEN) cl_ = MAX_OUTPUT_LEN;
    out_len[gid] = cl_;
    __global unsigned char *p = out + (long)gid * MAX_OUTPUT_LEN;
    for (int i = 0; i < cl_; i++) p[i] = cur[i];
}}
inline ulong celf_fnv1a64(__private const uchar *data, int len) {{
    ulong h = (ulong)14695981039346656037UL;
    for (int i = 0; i < len; i++) {{ h ^= (ulong)data[i]; h *= (ulong)1099511628211UL; }}
    return h;
}}

inline ulong celf_mix64(ulong x) {{
    x += (ulong)0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * (ulong)0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * (ulong)0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}}

// RULEST_CELF_PACKED_TARGETS_V10_1: byte payload uses per-target offsets.
inline int celf_find_target_slot(
    __global const ulong *target_hashes,
    __global const int *slot_target_ids,
    __global const uint *occupied,
    __global const uchar *target_bytes, __global const ulong *target_offsets, __global const int *target_lengths,
    __private const uchar *candidate,
    int candidate_len,
    uint table_mask)
{{
    ulong key = celf_fnv1a64(candidate, candidate_len);
    uint slot = (uint)(celf_mix64(key) & (ulong)table_mask);
    for (uint probe = 0; probe <= table_mask; probe++) {{
        uint bit_word = slot >> 5;
        uint bit_mask = 1U << (slot & 31U);
        if ((occupied[bit_word] & bit_mask) == 0U) return -1;
        if (target_hashes[slot] == key) {{
            int target_id = slot_target_ids[slot];
            if (target_id >= 0 && target_lengths[target_id] == candidate_len) {{
                int same = 1;
                for (int j = 0; j < candidate_len; j++) {{
                    if (target_bytes[target_offsets[target_id] + (ulong)j] != candidate[j]) {{ same = 0; break; }}
                }}
                if (same) return (int)slot;
            }}
        }}
        slot = (slot + 1U) & table_mask;
    }}
    return -1;
}}

// Every (base word, candidate chain) work item applies the same chain DSL as
// the main search kernel, then performs exact target-table lookup on the GPU.
// Every scoring mode deduplicates exact target IDs with a compact per-candidate
// bitmap. Batched scores are exact current marginal gains; stale heap values
// remain admissible upper bounds after the active target set shrinks.
__kernel void celf_score_chains_gpu(
    __global const uchar *bw, __global const int *bo, __global const int *bl,
    __global const uchar *rs, __global const int *ro, __global const int *rl,
    __global const int *cseq, __global const int *cdep,
    __global const int *candidate_indices,
    __global const ulong *target_hashes, __global const int *slot_target_ids,
    __global const uint *occupied, __global const uint *active,
    __global const uchar *target_bytes, __global const ulong *target_offsets, __global const int *target_lengths,
    __global uint *seen, __global int *gains,
    __global int *hit_slots, __global int *hit_count,
    const int nw, const int nc, const int word_start, const int mcd,
    const uint table_mask, const int bitmap_words,
    const int record_hits)
{{
    int gid = (int)get_global_id(0);
    int total = nw * nc;
    if (gid >= total || nw <= 0) return;
    int ci = gid / nw;
    int candidate_id = candidate_indices[ci];
    int wi = gid % nw + word_start;
    int wl = bl[wi];
    if (wl <= 0 || wl > MAX_OUTPUT_LEN) return;

    uchar cur[MAX_OUTPUT_LEN], tmp[MAX_OUTPUT_LEN], rr[MAX_RULE_LEN];
    for (int i = 0; i < wl; i++) cur[i] = bw[bo[wi] + i];
    cur[wl] = '\0';
    int cl = wl;
    int dep = cdep[candidate_id];
    if (dep < 1 || dep > mcd) return;
    for (int d = 0; d < dep; d++) {{
        int ri = cseq[candidate_id * mcd + d];
        if (ri < 0) return;
        int rlen = rl[ri];
        if (rlen <= 0 || rlen >= MAX_RULE_LEN) return;
        for (int i = 0; i < rlen; i++) rr[i] = rs[ro[ri] + i];
        rr[rlen] = '\0';
        int nl = 0;
        apply(rr, rlen, cur, cl, tmp, &nl);
        if (nl <= 0 || nl > MAX_OUTPUT_LEN) return;
        for (int i = 0; i < nl; i++) cur[i] = tmp[i];
        cl = nl;
        cur[cl] = '\0';
    }}

    // Match the existing Rulest search contract: no-op chains do not count.
    int same = (cl == wl);
    for (int i = 0; i < wl && same; i++) if (cur[i] != bw[bo[wi] + i]) same = 0;
    if (same) return;

    int slot = celf_find_target_slot(target_hashes, slot_target_ids, occupied,
                                     target_bytes, target_offsets, target_lengths, cur, cl, table_mask);
    if (slot < 0) return;
    uint slot_u = (uint)slot;
    uint slot_word = slot_u >> 5;
    uint slot_mask = 1U << (slot_u & 31U);
    if ((active[slot_word] & slot_mask) == 0U) return;
    int target_id = slot_target_ids[slot];
    if (target_id < 0) return;

    // Fast batch mode is an admissible upper bound: count every matching
    // base-word output. Duplicate outputs can overcount but cannot undercount
    // unique active targets. This avoids per-hit bitmap atomics in large batches.
    if (!record_hits) {{
        atomic_inc((volatile __global int *)(gains + ci));
        return;
    }}

    // Exact singleton mode deduplicates target IDs and records one slot per
    // distinct active target in the same pass.
    __global uint *seen_rule = seen + ci * bitmap_words;
    uint seen_word = ((uint)target_id) >> 5;
    uint seen_mask = 1U << (((uint)target_id) & 31U);
    uint previous = (uint)atomic_or((volatile __global int *)(seen_rule + seen_word), (int)seen_mask);
    if ((previous & seen_mask) != 0U) return;
    atomic_inc((volatile __global int *)(gains + ci));
    int out_idx = atomic_inc((volatile __global int *)hit_count);
    if (out_idx >= 0) hit_slots[out_idx] = slot;
}}

__kernel void celf_clear_active_slots(
    __global uint *active, __global const int *slots, const int n_slots)
{{
    int gid = (int)get_global_id(0);
    if (gid >= n_slots) return;
    uint slot = (uint)slots[gid];
    uint mask = 1U << (slot & 31U);
    atomic_and((volatile __global int *)(active + (slot >> 5)), (int)(~mask));
}}
"""
