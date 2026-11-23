// Senja Site v6 — flat-architecture rewrite (see BUILD-v6.md).
// Leaf functions below are carried over verbatim from v5 (our own code,
// already verified byte-identical); the architecture above them — flat
// JSON parse, no JSON tree, no Val tree, constant-folded template
// slots, AVX2 scans — is new.
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <cpuid.h>
#include <xmmintrin.h>
#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3 _mm_shuffle_epi8 (x86-64-v2)
#include <smmintrin.h>  // SSE4.1 (kept for parity with v5's helpers)
#include <immintrin.h>  // AVX2
#include <node_api.h>
#include "unicode_tables.h"

// Runtime CPU dispatch (set once in senja_init).
static int have_avx2 = 0;

// (JParser is defined after the Slice/Arena types, just before j_hex4,
// because it references both.)



// ---- carried from v5: types/arena/buf (v5 lines 26-122) ----

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

// ------------------------------------------------------------------ slices

typedef struct { const u8 *p; size_t n; } Slice;
static inline Slice sl(const u8 *p, size_t n) { Slice s; s.p = p; s.n = n; return s; }
static inline Slice slc(const char *s) { return sl((const u8 *)s, strlen(s)); }
static inline int sl_eq(Slice a, Slice b) { return a.n == b.n && (a.n == 0 || memcmp(a.p, b.p, a.n) == 0); }
static inline int sl_eq_c(Slice a, const char *b) { size_t n = strlen(b); return a.n == n && (n == 0 || memcmp(a.p, b, n) == 0); }
static inline int sl_ends(Slice a, const char *suf) {
  size_t n = strlen(suf);
  return a.n >= n && (n == 0 || memcmp(a.p + a.n - n, suf, n) == 0);
}
static inline int sl_starts(Slice a, const char *pre) {
  size_t n = strlen(pre);
  return a.n >= n && (n == 0 || memcmp(a.p, pre, n) == 0);
}

// ------------------------------------------------------------------ arena

typedef struct AChunk { struct AChunk *next; size_t used, cap; u8 data[]; } AChunk;
typedef struct { AChunk *head, *cur; } Arena;
#define ARENA_CHUNK (1u << 22) // 4 MiB chunks

static void arena_init(Arena *a) { a->head = a->cur = NULL; }
static void *arena_alloc(Arena *a, size_t n) {
  n = (n + 7) & ~(size_t)7;
  if (n == 0) n = 8;
  AChunk *c = a->cur;
  if (!c || c->used + n > c->cap) {
    size_t cap = ARENA_CHUNK;
    if (n > cap) cap = n;
    c = (AChunk *)malloc(sizeof(AChunk) + cap);
    if (!c) abort();
    c->next = NULL; c->used = 0; c->cap = cap;
    if (a->cur) a->cur->next = c; else a->head = c;
    a->cur = c;
  }
  void *p = c->data + c->used;
  c->used += n;
  return p;
}
static void arena_reset(Arena *a) {
  for (AChunk *c = a->head; c; c = c->next) c->used = 0;
  a->cur = a->head;
}
static void arena_free(Arena *a) {
  AChunk *c = a->head;
  while (c) { AChunk *nx = c->next; free(c); c = nx; }
  a->head = a->cur = NULL;
}
static Slice arena_slice(Arena *a, const u8 *p, size_t n) {
  u8 *q = (u8 *)arena_alloc(a, n ? n : 1);
  if (n) memcpy(q, p, n);
  return sl(q, n);
}
static Slice arena_cat2(Arena *a, Slice x, Slice y) {
  u8 *q = (u8 *)arena_alloc(a, x.n + y.n ? x.n + y.n : 1);
  if (x.n) memcpy(q, x.p, x.n);
  if (y.n) memcpy(q + x.n, y.p, y.n);
  return sl(q, x.n + y.n);
}
static Slice arena_cat3(Arena *a, Slice x, Slice y, Slice z) {
  u8 *q = (u8 *)arena_alloc(a, x.n + y.n + z.n ? x.n + y.n + z.n : 1);
  size_t o = 0;
  if (x.n) { memcpy(q + o, x.p, x.n); o += x.n; }
  if (y.n) { memcpy(q + o, y.p, y.n); o += y.n; }
  if (z.n) { memcpy(q + o, z.p, z.n); o += z.n; }
  return sl(q, x.n + y.n + z.n);
}

// ------------------------------------------------------------------ growable byte buffer (malloc'd)

typedef struct { u8 *p; size_t len, cap; } Buf;
static void buf_reserve(Buf *b, size_t extra) {
  if (b->len + extra > b->cap) {
    size_t nc = b->cap ? b->cap : 256;
    while (nc < b->len + extra) nc *= 2;
    b->p = (u8 *)realloc(b->p, nc);
    if (!b->p) abort();
    b->cap = nc;
  }
}
static inline void buf_put(Buf *b, const void *s, size_t n) {
  if (n == 0) return;
  buf_reserve(b, n);
  memcpy(b->p + b->len, s, n);
  b->len += n;
}
static inline void buf_sl(Buf *b, Slice s) { buf_put(b, s.p, s.n); }
static inline void buf_c(Buf *b, u8 c) { buf_reserve(b, 1); b->p[b->len++] = c; }
static inline void buf_cs(Buf *b, const char *s) { buf_put(b, s, strlen(s)); }
static inline void buf_u64(Buf *b, u64 v) { // decimal, no allocation
  char tmp[24]; int i = 23;
  if (v == 0) { buf_c(b, '0'); return; }
  while (v > 0) { tmp[--i] = (char)('0' + (v % 10)); v /= 10; }
  buf_put(b, tmp + i, 24 - (size_t)i - 1);
}



// ---- carried from v5: utf8/u16/cmp/trim/find_byte (v5 lines 128-247) ----

static inline u32 utf8_decode(const u8 *b, size_t len, size_t i, size_t *used) {
  u8 b0 = b[i];
  if (b0 < 0x80) { *used = 1; return b0; }
  if (b0 >= 0xC2 && b0 < 0xE0 && i + 1 < len && (b[i + 1] & 0xC0) == 0x80) {
    *used = 2; return ((u32)(b0 & 0x1F) << 6) | (u32)(b[i + 1] & 0x3F);
  }
  if (b0 >= 0xE0 && b0 < 0xF0 && i + 2 < len && (b[i + 1] & 0xC0) == 0x80 && (b[i + 2] & 0xC0) == 0x80) {
    u32 cp = ((u32)(b0 & 0x0F) << 12) | ((u32)(b[i + 1] & 0x3F) << 6) | (u32)(b[i + 2] & 0x3F);
    if (cp >= 0x800) { *used = 3; return cp; }
    *used = 1; return 0xFFFD;
  }
  if (b0 >= 0xF0 && b0 < 0xF5 && i + 3 < len
      && (b[i + 1] & 0xC0) == 0x80 && (b[i + 2] & 0xC0) == 0x80 && (b[i + 3] & 0xC0) == 0x80) {
    u32 cp = ((u32)(b0 & 0x07) << 18) | ((u32)(b[i + 1] & 0x3F) << 12)
           | ((u32)(b[i + 2] & 0x3F) << 6) | (u32)(b[i + 3] & 0x3F);
    if (cp >= 0x10000 && cp <= 0x10FFFF) { *used = 4; return cp; }
    *used = 1; return 0xFFFD;
  }
  *used = 1; return 0xFFFD;
}

static inline void utf8_encode(u32 cp, Buf *out) {
  if (cp < 0x80) { buf_c(out, (u8)cp); }
  else if (cp < 0x800) {
    buf_c(out, (u8)(0xC0 | (cp >> 6))); buf_c(out, (u8)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    buf_c(out, (u8)(0xE0 | (cp >> 12))); buf_c(out, (u8)(0x80 | ((cp >> 6) & 0x3F))); buf_c(out, (u8)(0x80 | (cp & 0x3F)));
  } else {
    buf_c(out, (u8)(0xF0 | (cp >> 18))); buf_c(out, (u8)(0x80 | ((cp >> 12) & 0x3F)));
    buf_c(out, (u8)(0x80 | ((cp >> 6) & 0x3F))); buf_c(out, (u8)(0x80 | (cp & 0x3F)));
  }
}

// UTF-16 code units of a WTF-8 string, one at a time.
typedef struct { const u8 *b; size_t len, i; int has_pending; u32 pending; } U16Iter;
static inline void u16_init(U16Iter *it, Slice s) { it->b = s.p; it->len = s.n; it->i = 0; it->has_pending = 0; it->pending = 0; }
// Returns 1 and sets *u, or 0 at the end.
static int u16_next(U16Iter *it, u32 *u) {
  if (it->has_pending) { it->has_pending = 0; *u = it->pending; return 1; }
  if (it->i >= it->len) return 0;
  size_t used; u32 cp = utf8_decode(it->b, it->len, it->i, &used);
  it->i += used;
  if (cp >= 0x10000) {
    u32 v = cp - 0x10000;
    it->pending = 0xDC00 + (v & 0x3FF); it->has_pending = 1;
    *u = 0xD800 + (v >> 10);
  } else *u = cp;
  return 1;
}

// Compare two WTF-8 strings in UTF-16 code-unit order (JS `<`).
static int cmp_u16(Slice a, Slice b) {
  U16Iter ia, ib; u16_init(&ia, a); u16_init(&ib, b);
  for (;;) {
    u32 x, y; int ha = u16_next(&ia, &x), hb = u16_next(&ib, &y);
    if (!ha && !hb) return 0;
    if (!ha) return -1;
    if (!hb) return 1;
    if (x != y) return x < y ? -1 : 1;
  }
}
static size_t u16_len(Slice s) {
  U16Iter it; u16_init(&it, s); u32 u; size_t n = 0;
  while (u16_next(&it, &u)) n++;
  return n;
}

// JavaScript \s: White_Space plus line terminators.
static inline int is_js_space(u32 cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0xA0 || cp == 0x1680
      || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029
      || cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0xFEFF;
}

// JavaScript String.prototype.trim.
static Slice js_trim(Slice b) {
  size_t a = 0, z = b.n;
  while (a < z) {
    size_t used; u32 cp = utf8_decode(b.p, b.n, a, &used);
    if (!is_js_space(cp)) break;
    a += used;
  }
  while (z > a) {
    size_t s = z - 1;
    while (s > a && (b.p[s] & 0xC0) == 0x80) s--;
    size_t used; u32 cp = utf8_decode(b.p, b.n, s, &used);
    if (!is_js_space(cp) || s + used != z) break;
    z = s;
  }
  return sl(b.p + a, z - a);
}

static inline int is_blank_sp_tab(Slice line) {
  for (size_t i = 0; i < line.n; i++) if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  return 1;
}

// ------------------------------------------------------------------ SSE2 scans

// First index >= from holding byte b, or n.
static size_t find_byte(const u8 *s, size_t n, size_t from, u8 b) {
  size_t i = from;
  __m128i needle = _mm_set1_epi8((char)b);
  while (i + 16 <= n) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + i));
    unsigned mask = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(chunk, needle));
    if (mask) return i + (size_t)__builtin_ctz(mask);
    i += 16;
  }
  if (i < n && n >= 16) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + n - 16));
    unsigned mask = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(chunk, needle));
    size_t skip = i - (n - 16);
    mask &= ~((1u << skip) - 1u);
    if (mask) return (n - 16) + (size_t)__builtin_ctz(mask);
    return n;
  }
  while (i < n) { if (s[i] == b) return i; i++; }
  return n;
}



// ---- carried from v5: next_json_esc (v5 lines 252-285) ----

static size_t next_json_esc_avx2(const u8 *s, size_t n, size_t from);
static size_t next_json_esc(const u8 *s, size_t n, size_t from) {
  if (have_avx2) return next_json_esc_avx2(s, n, from);
  size_t i = from;
  __m128i quote = _mm_set1_epi8('"');
  __m128i backslash = _mm_set1_epi8('\\');
  __m128i lead_ed = _mm_set1_epi8((char)0xED);
  __m128i lo31 = _mm_set1_epi8(0x1F);
  __m128i zero = _mm_setzero_si128();
  while (i + 16 <= n) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + i));
    __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, quote), _mm_cmpeq_epi8(chunk, backslash)),
                              _mm_or_si128(_mm_cmpeq_epi8(chunk, lead_ed),
                                           _mm_cmpeq_epi8(_mm_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (mask) return i + (size_t)__builtin_ctz(mask);
    i += 16;
  }
  if (i < n && n >= 16) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + n - 16));
    __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, quote), _mm_cmpeq_epi8(chunk, backslash)),
                              _mm_or_si128(_mm_cmpeq_epi8(chunk, lead_ed),
                                           _mm_cmpeq_epi8(_mm_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    size_t skip = i - (n - 16);
    mask &= ~((1u << skip) - 1u);
    if (mask) return (n - 16) + (size_t)__builtin_ctz(mask);
    return n;
  }
  while (i < n) {
    u8 c = s[i];
    if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) return i;
    i++;
  }
  return n;
}



// ---- carried from v5: fxhash/strmap (v5 lines 289-350) ----

static inline u64 fx_mix(u64 h, u64 v) { return ((((h << 5) | (h >> 59)) ^ v) * 0x517cc1b727220a95ULL); }
static u64 fx_hash_bytes(const u8 *p, size_t n) {
  u64 h = 0;
  size_t i = 0;
  // 8 bytes per mix (the hash is internal only — bucket order is never
  // observable: the one place that reads map keys sorts them after).
  while (i + 8 <= n) {
    u64 v;
    memcpy(&v, p + i, 8);
    h = fx_mix(h, v);
    i += 8;
  }
  while (i < n) h = fx_mix(h, p[i++]);
  h = fx_mix(h, (u64)n); // Rust Hash for [u8] writes the length last
  return h;
}

// Open-addressing map from byte-string keys to void* values. Keys are
// copied into the arena on insert, so callers may pass scratch slices.
typedef struct { u64 hash; const u8 *key; size_t klen; void *val; } StrSlot;
typedef struct { StrSlot *slots; size_t cap, len; Arena *arena; } StrMap;

static void strmap_init(StrMap *m, Arena *a) { m->slots = NULL; m->cap = 0; m->len = 0; m->arena = a; }
static void strmap_grow(StrMap *m) {
  size_t nc = m->cap ? m->cap * 2 : 32;
  StrSlot *ns = (StrSlot *)arena_alloc(m->arena, nc * sizeof(StrSlot));
  memset(ns, 0, nc * sizeof(StrSlot));
  for (size_t i = 0; i < m->cap; i++) {
    if (m->slots[i].key) {
      size_t j = m->slots[i].hash & (nc - 1);
      while (ns[j].key) j = (j + 1) & (nc - 1);
      ns[j] = m->slots[i];
    }
  }
  m->slots = ns; m->cap = nc;
}
static void *strmap_get(StrMap *m, Slice key) {
  if (!m->cap) return NULL;
  u64 h = fx_hash_bytes(key.p, key.n);
  size_t j = h & (m->cap - 1);
  while (m->slots[j].key) {
    if (m->slots[j].hash == h && m->slots[j].klen == key.n && memcmp(m->slots[j].key, key.p, key.n) == 0)
      return m->slots[j].val;
    j = (j + 1) & (m->cap - 1);
  }
  return NULL;
}
static int strmap_has(StrMap *m, Slice key) { return strmap_get(m, key) != NULL; }
static void strmap_put(StrMap *m, Slice key, void *val) {
  if ((m->len + 1) * 10 > m->cap * 7) strmap_grow(m);
  u64 h = fx_hash_bytes(key.p, key.n);
  size_t j = h & (m->cap - 1);
  while (m->slots[j].key) {
    if (m->slots[j].hash == h && m->slots[j].klen == key.n && memcmp(m->slots[j].key, key.p, key.n) == 0) {
      m->slots[j].val = val; return;
    }
    j = (j + 1) & (m->cap - 1);
  }
  Slice owned = arena_slice(m->arena, key.p, key.n);
  m->slots[j].hash = h; m->slots[j].key = owned.p; m->slots[j].klen = owned.n; m->slots[j].val = val;
  m->len++;
}



// ---- carried from v5: j_hex4 (v5 lines 376-389) ----

// Flat JSON parser cursor (same layout as v5's JParser; the extracted
// j_hex4 uses it, and the v6 flat parser is built on it).
typedef struct { const u8 *b; size_t len, pos; Arena *arena; } JParser;

static int j_hex4(JParser *p, u32 *out) {
  if (p->pos + 4 > p->len) return 0;
  u32 v = 0;
  for (int i = 0; i < 4; i++) {
    u8 c = p->b[p->pos];
    u32 d;
    if (c >= '0' && c <= '9') d = (u32)(c - '0');
    else if (c >= 'a' && c <= 'f') d = (u32)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') d = (u32)(c - 'A' + 10);
    else return 0;
    v = (v << 4) | d; p->pos++;
  }
  *out = v; return 1;
}



// ---- carried from v5: json_write/js_num (v5 lines 601-740) ----

static size_t emit_json_escape(Buf *out, const u8 *p, size_t n, size_t j) {
  static const char HEX[] = "0123456789abcdef";
  u8 c = p[j];
  switch (c) {
    case '"': buf_cs(out, "\\\""); return j + 1;
    case '\\': buf_cs(out, "\\\\"); return j + 1;
    case 0x08: buf_cs(out, "\\b"); return j + 1;
    case 0x09: buf_cs(out, "\\t"); return j + 1;
    case 0x0A: buf_cs(out, "\\n"); return j + 1;
    case 0x0C: buf_cs(out, "\\f"); return j + 1;
    case 0x0D: buf_cs(out, "\\r"); return j + 1;
    default:
      if (c < 0x20) {
        buf_cs(out, "\\u00"); buf_c(out, (u8)HEX[(c >> 4) & 0xF]); buf_c(out, (u8)HEX[c & 0xF]);
        return j + 1;
      } else if (c == 0xED && j + 2 < n && p[j + 1] >= 0xA0 && p[j + 1] <= 0xBF && (p[j + 2] & 0xC0) == 0x80) {
        u32 cp = 0xD800 + (((u32)(p[j + 1] & 0x3F)) << 6) + (u32)(p[j + 2] & 0x3F);
        buf_cs(out, "\\u");
        buf_c(out, (u8)HEX[(cp >> 12) & 0xF]); buf_c(out, (u8)HEX[(cp >> 8) & 0xF]);
        buf_c(out, (u8)HEX[(cp >> 4) & 0xF]); buf_c(out, (u8)HEX[cp & 0xF]);
        return j + 3;
      } else { buf_c(out, c); return j + 1; }
  }
}

// JSON string writer. The scan runs inline (no per-escape function
// call): 16-byte chunks, one movemask per chunk, and when a chunk has
// escapes every set bit is drained before moving on (mask-draining —
// the chunk is never re-scanned for its next escape). Plain runs are
// copied in bulk across chunk boundaries.
static void json_write_str(Buf *out, Slice s) {
  buf_c(out, '"');
  const u8 *p = s.p; size_t n = s.n;
  __m128i quote = _mm_set1_epi8('"');
  __m128i backslash = _mm_set1_epi8('\\');
  __m128i lead_ed = _mm_set1_epi8((char)0xED);
  __m128i lo31 = _mm_set1_epi8(0x1F);
  __m128i zero = _mm_setzero_si128();
  size_t i = 0;     // scan position
  size_t plain = 0; // start of the not-yet-copied plain run
  while (i < n) {
    if (i + 16 <= n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(p + i));
      __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, quote), _mm_cmpeq_epi8(chunk, backslash)),
                                _mm_or_si128(_mm_cmpeq_epi8(chunk, lead_ed),
                                             _mm_cmpeq_epi8(_mm_subs_epu8(chunk, lo31), zero)));
      unsigned mask = (unsigned)_mm_movemask_epi8(m);
      if (!mask) { i += 16; continue; }
      size_t base = i;
      while (mask) {
        unsigned b = __builtin_ctz(mask);
        size_t j = base + b;
        if (j > plain) buf_put(out, p + plain, j - plain);
        plain = emit_json_escape(out, p, n, j);
        mask &= mask - 1;
      }
      i = plain;
    } else {
      size_t j = i;
      while (j < n) {
        u8 c = p[j];
        if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) break;
        j++;
      }
      if (j > plain) buf_put(out, p + plain, j - plain);
      if (j >= n) { plain = n; break; }
      plain = emit_json_escape(out, p, n, j);
      i = plain;
    }
  }
  if (plain < n) buf_put(out, p + plain, n - plain);
  buf_c(out, '"');
}

// JavaScript Number.prototype.toString for a double (ECMAScript
// Number::toString): shortest digits that round-trip, then the spec's
// plain/scientific layout. Integers with |n| < 1e21 use plain digits.
static void js_num(Buf *out, double n) {
  if (isnan(n)) { buf_cs(out, "NaN"); return; }
  if (isinf(n)) { buf_cs(out, n < 0 ? "-Infinity" : "Infinity"); return; }
  if (n == 0.0) { buf_c(out, '0'); return; } // -0 prints "0" in JS too
  if (n < 0) { buf_c(out, '-'); js_num(out, -n); return; }
  if (n == trunc(n) && n < 1e21) {
    // Exact integer digits via %.0f (glibc converts exactly).
    char tmp[32];
    int m = snprintf(tmp, sizeof(tmp), "%.0f", n);
    buf_put(out, tmp, (size_t)m);
    return;
  }
  // Shortest round-trip decimal digits: try 15, 16, 17 significant digits.
  char digs[24]; int ndig = 0, exp10 = 0;
  for (int prec = 15; prec <= 17; prec++) {
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%.*e", prec - 1, n);
    // tmp like "1.2345e+20" or "1e-07".
    char *e = strchr(tmp, 'e');
    exp10 = atoi(e + 1);
    int di = 0;
    for (char *p = tmp; p < e; p++) {
      if (*p >= '0' && *p <= '9') digs[di++] = *p;
    }
    while (di > 1 && digs[di - 1] == '0') di--; // strip trailing zeros
    digs[di] = 0;
    // Round-trip check.
    char back[40];
    snprintf(back, sizeof(back), "%se%d", digs, exp10 - (di - 1));
    if (strtod(back, NULL) == n) { ndig = di; break; }
  }
  if (ndig == 0) { // fall back to 17 digits
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%.16e", n);
    char *e = strchr(tmp, 'e');
    exp10 = atoi(e + 1);
    int di = 0;
    for (char *p = tmp; p < e; p++) if (*p >= '0' && *p <= '9') digs[di++] = *p;
    while (di > 1 && digs[di - 1] == '0') di--;
    digs[di] = 0; ndig = di;
  }
  int k = ndig, nn = exp10 + 1; // n = 0.digs * 10^nn
  if (k <= nn && nn <= 21) {
    buf_put(out, digs, (size_t)k);
    for (int i = 0; i < nn - k; i++) buf_c(out, '0');
  } else if (0 < nn && nn <= 21) {
    buf_put(out, digs, (size_t)nn);
    buf_c(out, '.');
    buf_put(out, digs + nn, (size_t)(k - nn));
  } else if (-6 < nn && nn <= 0) {
    buf_cs(out, "0.");
    for (int i = 0; i < -nn; i++) buf_c(out, '0');
    buf_put(out, digs, (size_t)k);
  } else {
    buf_c(out, (u8)digs[0]);
    if (k > 1) { buf_c(out, '.'); buf_put(out, digs + 1, (size_t)(k - 1)); }
    buf_c(out, 'e');
    buf_c(out, (u8)(nn - 1 < 0 ? '-' : '+'));
    int ex = nn - 1; if (ex < 0) ex = -ex;
    char et[8]; int m = snprintf(et, sizeof(et), "%d", ex);
    buf_put(out, et, (size_t)m);
  }
}



// ---- carried from v5: esc/slugify (v5 lines 749-921) ----

static const u8 ESC_AMP_B[16] = { '&','&','&','&','&','&','&','&','&','&','&','&','&','&','&','&' };
static const u8 ESC_LT_B[16] = { '<','<','<','<','<','<','<','<','<','<','<','<','<','<','<','<' };
static const u8 ESC_GT_B[16] = { '>','>','>','>','>','>','>','>','>','>','>','>','>','>','>','>' };
static const u8 ESC_QUOT_B[16] = { '"','"','"','"','"','"','"','"','"','"','"','"','"','"','"','"' };
static const u8 ESC_APOS_B[16] = { '\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'' };

static inline void esc_one(Buf *out, u8 c) {
  switch (c) {
    case '&': buf_cs(out, "&amp;"); break;
    case '<': buf_cs(out, "&lt;"); break;
    case '>': buf_cs(out, "&gt;"); break;
    case '"': buf_cs(out, "&quot;"); break;
    default: buf_cs(out, "&#39;"); break;
  }
}

static void esc(Buf *out, Slice s) {
  const u8 *p = s.p; size_t n = s.n;
  if (n < 16) {
    // Scalar loop with a plain-run pointer (no vector setup).
    size_t plain = 0;
    size_t i = 0;
    while (i < n) {
      u8 c = p[i];
      if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') {
        if (i > plain) buf_put(out, p + plain, i - plain);
        esc_one(out, c);
        plain = i + 1;
      }
      i++;
    }
    if (plain < n) buf_put(out, p + plain, n - plain);
    return;
  }
  __m128i amp = _mm_loadu_si128((const __m128i *)ESC_AMP_B);
  __m128i lt = _mm_loadu_si128((const __m128i *)ESC_LT_B);
  __m128i gt = _mm_loadu_si128((const __m128i *)ESC_GT_B);
  __m128i quot = _mm_loadu_si128((const __m128i *)ESC_QUOT_B);
  __m128i apos = _mm_loadu_si128((const __m128i *)ESC_APOS_B);
  size_t i = 0;     // scan position
  size_t plain = 0; // start of the not-yet-copied plain run
  while (i < n) {
    if (i + 16 <= n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(p + i));
      __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, amp), _mm_cmpeq_epi8(chunk, lt)),
                                _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, gt), _mm_cmpeq_epi8(chunk, quot)),
                                              _mm_cmpeq_epi8(chunk, apos)));
      unsigned mask = (unsigned)_mm_movemask_epi8(m);
      if (!mask) { i += 16; continue; }
      size_t base = i;
      while (mask) {
        unsigned b = __builtin_ctz(mask);
        size_t j = base + b;
        if (j > plain) buf_put(out, p + plain, j - plain);
        esc_one(out, p[j]);
        plain = j + 1;
        mask &= mask - 1;
      }
      i = plain;
    } else {
      size_t j = i;
      while (j < n) {
        u8 c = p[j];
        if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') break;
        j++;
      }
      if (j > plain) buf_put(out, p + plain, j - plain);
      if (j >= n) { plain = n; break; }
      esc_one(out, p[j]);
      plain = j + 1;
      i = plain;
    }
  }
  if (plain < n) buf_put(out, p + plain, n - plain);
}

static int is_mark(u32 cp) {
  if (cp < 0x300) return 0; // combining marks start at U+0300
  size_t lo = 0, hi = MARK_N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    u32 a = MARK_RANGES[mid][0], b = MARK_RANGES[mid][1];
    if (cp < a) hi = mid;
    else if (cp > b) lo = mid + 1;
    else return 1;
  }
  return 0;
}

// Fully-expanded NFKD sequence for cp, or NULL. Returns count via *n.
static const u32 *decomp_of(u32 cp, size_t *n) {
  size_t lo = 0, hi = DECOMP_N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    u32 k = DECOMP_KEYS[mid];
    if (cp < k) hi = mid;
    else if (cp > k) lo = mid + 1;
    else { *n = (size_t)(DECOMP_OFF[mid + 1] - DECOMP_OFF[mid]); return DECOMP_VALS + DECOMP_OFF[mid]; }
  }
  return NULL;
}

static inline void step_slug(u32 c, Buf *out, int *gap) {
  if (is_mark(c)) return;
  int keep = -1;
  if (c >= 'A' && c <= 'Z') keep = (int)(c + 32);
  else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) keep = (int)c;
  if (keep < 0) { *gap = out->len > 0; }
  else {
    if (*gap) buf_c(out, '-');
    *gap = 0;
    buf_c(out, (u8)keep);
  }
}

// The slug of `text`: NFKD, marks dropped, A-Z lowered, runs of other
// chars one '-', none at either end, cut to 64 with a trailing '-' at the
// cut dropped, 'untitled' when nothing is left.
static Slice slugify(Arena *arena, Slice text) {
  Buf out = {0};
  int gap = 0;
  // ASCII fast path: no ASCII code point has an NFKD decomposition and
  // no ASCII char is a mark, so bytes map straight through.
  int ascii = 1;
  for (size_t k = 0; k + 16 <= text.n; k += 16) {
    if (_mm_movemask_epi8(_mm_loadu_si128((const __m128i *)(text.p + k)))) { ascii = 0; break; }
  }
  if (ascii) for (size_t k = text.n & ~(size_t)15; k < text.n; k++) if (text.p[k] >= 0x80) { ascii = 0; break; }
  if (ascii) {
    for (size_t i = 0; i < text.n; i++) {
      u8 c = text.p[i];
      int keep = -1;
      if (c >= 'A' && c <= 'Z') keep = (int)(c + 32);
      else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) keep = (int)c;
      if (keep < 0) { gap = out.len > 0; }
      else {
        if (gap) buf_c(&out, '-');
        gap = 0;
        buf_c(&out, (u8)keep);
      }
    }
  } else {
    size_t i = 0;
    while (i < text.n) {
      size_t used; u32 cp = utf8_decode(text.p, text.n, i, &used);
      i += used;
      size_t n; const u32 *seq = cp < 0x80 ? NULL : decomp_of(cp, &n);
      if (seq) { for (size_t k = 0; k < n; k++) step_slug(seq[k], &out, &gap); }
      else step_slug(cp, &out, &gap);
    }
  }
  if (out.len > 64) out.len = 64;
  while (out.len > 0 && out.p[out.len - 1] == '-') out.len--;
  Slice r;
  if (out.len == 0) r = slc("untitled");
  else r = arena_slice(arena, out.p, out.len);
  free(out.p);
  return r;
}

// `slug` made unique against `taken` (and added to it).
static Slice unique_slug(Arena *arena, StrMap *taken, Slice slug) {
  if (!strmap_has(taken, slug)) { strmap_put(taken, slug, (void *)1); return slug; }
  u32 n = 2;
  for (;;) {
    char num[16]; int m = snprintf(num, sizeof(num), "%u", n);
    Buf tmp = {0};
    buf_sl(&tmp, slug); buf_c(&tmp, '-'); buf_put(&tmp, num, (size_t)m);
    Slice cand = arena_slice(arena, tmp.p, tmp.len);
    free(tmp.p);
    if (!strmap_has(taken, cand)) { strmap_put(taken, cand, (void *)1); return cand; }
    n++;
  }
}



// ---- carried from v5: long_date/words_of (v5 lines 924-1005) ----

static const char *MONTHS[12] = {
  "January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December",
};

// A YYYY-MM-DD date in words, from the string alone.
static Slice long_date(Arena *arena, Slice date) {
  u32 day = (u32)(date.p[8] - '0') * 10 + (u32)(date.p[9] - '0');
  size_t mon = (size_t)(((u32)(date.p[5] - '0') * 10 + (u32)(date.p[6] - '0')));
  char num[8]; int m = snprintf(num, sizeof(num), "%u", day);
  Buf out = {0};
  buf_put(&out, num, (size_t)m); buf_c(&out, ' ');
  buf_cs(&out, MONTHS[mon - 1]); buf_c(&out, ' ');
  buf_put(&out, date.p, 4);
  Slice r = arena_slice(arena, out.p, out.len);
  free(out.p);
  return r;
}

// The words of a body: runs of characters other than whitespace
// (the reference's /\S+/g over the JS \s set). Pure-ASCII 16-byte
// chunks are counted with an SSSE3 whitespace classifier (a word starts
// where a non-ws byte follows a ws byte); any chunk with a byte >= 0x80
// falls back to the scalar UTF-8 path for that chunk, so the JS \s set
// for non-ASCII is exactly preserved.
static size_t words_of(Slice text) {
  static const u8 WS_LO[16] = { 0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,0x01,0x01,0x01,0x00,0x00 };
  static const u8 WS_BIT[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };
  __m128i ws_lo = _mm_loadu_si128((const __m128i *)WS_LO);
  __m128i ws_bit = _mm_loadu_si128((const __m128i *)WS_BIT);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  size_t n = 0, i = 0;
  int prev_ws = 1; // as if preceded by whitespace (first word counts)
  while (i < text.n) {
    if (i + 16 <= text.n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(text.p + i));
      if (_mm_movemask_epi8(chunk)) {
        // Non-ASCII in this chunk: scalar, character by character.
        // (A character may straddle the chunk end; i then lands on the
        // next character boundary and chunking resumes from there.)
        int inw = !prev_ws;
        size_t end = i + 16;
        while (i < end) {
          u8 c = text.p[i];
          if (c < 0x80) {
            int ws = c == ' ' || (c >= 0x09 && c <= 0x0D);
            if (ws) inw = 0; else if (!inw) { inw = 1; n++; }
            i++;
          } else {
            size_t used; u32 cp = utf8_decode(text.p, text.n, i, &used);
            i += used;
            if (is_js_space(cp)) inw = 0; else if (!inw) { inw = 1; n++; }
          }
        }
        prev_ws = !inw;
        continue;
      }
      __m128i lo = _mm_and_si128(chunk, lowmask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
      __m128i cand = _mm_and_si128(_mm_shuffle_epi8(ws_lo, lo), _mm_shuffle_epi8(ws_bit, hi));
      unsigned ws = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero));
      ws = ~ws & 0xFFFFu; // bit = 1 iff the byte is whitespace
      unsigned starts = ~ws & ((ws << 1) | (prev_ws ? 1u : 0u)) & 0xFFFFu;
      n += (size_t)__builtin_popcount(starts);
      prev_ws = (int)((ws >> 15) & 1u);
      i += 16;
    } else {
      u8 c = text.p[i];
      if (c < 0x80) {
        int ws = c == ' ' || (c >= 0x09 && c <= 0x0D);
        if (ws) prev_ws = 1; else if (prev_ws) { prev_ws = 0; n++; }
        i++;
      } else {
        size_t used; u32 cp = utf8_decode(text.p, text.n, i, &used);
        i += used;
        if (is_js_space(cp)) prev_ws = 1; else if (prev_ws) { prev_ws = 0; n++; }
      }
    }
  }
  return n;
}



// ---- carried from v5: front_matter (v5 lines 1009-1250) ----

typedef struct {
  int has_title, has_date, has_tags, has_draft, has_layout, has_slug, has_summary;
  Slice title, date, layout, slug, summary;
  Slice *tags; size_t ntags; // arena array
  int draft;
} FmData;

typedef struct { FmData data; Slice body; } Fm;

static int days_in(u32 y, u32 m) {
  if (m == 2) return ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 29 : 28;
  if (m == 4 || m == 6 || m == 9 || m == 11) return 30;
  return 31;
}

static int is_digit(u8 c) { return c >= '0' && c <= '9'; }

// A string value: "..." with \" and \\ its only escapes, or bare.
// On error returns 0 and sets *err to the reason (static text).
static int fm_string_of(Slice raw, Arena *arena, Slice *out, Slice *err) {
  if (raw.n > 0 && raw.p[0] == '"') {
    if (raw.n < 2 || raw.p[raw.n - 1] != '"') { *err = slc("a quoted string is not closed"); return 0; }
    Buf b = {0};
    size_t i = 1;
    while (i < raw.n - 1) {
      u8 c = raw.p[i];
      if (c == '\\') {
        u8 nx = raw.p[i + 1];
        if ((nx != '"' && nx != '\\') || i + 1 == raw.n - 1) {
          free(b.p); *err = slc("a quoted string holds an escape other than \\\" or \\\\"); return 0;
        }
        buf_c(&b, nx); i += 2;
      } else if (c == '"') {
        free(b.p); *err = slc("a quoted string holds a stray quote"); return 0;
      } else { buf_c(&b, c); i++; }
    }
    *out = arena_slice(arena, b.p ? b.p : (const u8 *)"", b.len);
    free(b.p);
    return 1;
  }
  if (raw.n > 0 && raw.p[0] == '[') { *err = slc("a bare string opens on ["); return 0; }
  *out = raw;
  return 1;
}

// Error line: key "K": reason
static Slice fm_err_key(Arena *arena, Slice key, Slice reason) {
  Buf b = {0};
  buf_cs(&b, "key \""); buf_sl(&b, key); buf_cs(&b, "\": "); buf_sl(&b, reason);
  Slice r = arena_slice(arena, b.p, b.len);
  free(b.p);
  return r;
}

static int fm_value_of(Slice key, Slice raw, FmData *data, Arena *arena, Slice *err) {
  if (sl_eq_c(key, "title") || sl_eq_c(key, "summary")) {
    Slice s;
    if (!fm_string_of(raw, arena, &s, err)) { *err = fm_err_key(arena, key, *err); return 0; }
    if (sl_eq_c(key, "title")) { data->title = s; data->has_title = 1; }
    else { data->summary = s; data->has_summary = 1; }
    return 1;
  }
  if (sl_eq_c(key, "date")) {
    int ok = raw.n == 10 && is_digit(raw.p[0]) && is_digit(raw.p[1]) && is_digit(raw.p[2]) && is_digit(raw.p[3])
          && raw.p[4] == '-' && is_digit(raw.p[5]) && is_digit(raw.p[6]) && raw.p[7] == '-'
          && is_digit(raw.p[8]) && is_digit(raw.p[9]);
    u32 y = 0, mo = 0, d = 0;
    if (ok) {
      y = (u32)(raw.p[0] - '0') * 1000 + (u32)(raw.p[1] - '0') * 100 + (u32)(raw.p[2] - '0') * 10 + (u32)(raw.p[3] - '0');
      mo = (u32)(raw.p[5] - '0') * 10 + (u32)(raw.p[6] - '0');
      d = (u32)(raw.p[8] - '0') * 10 + (u32)(raw.p[9] - '0');
    }
    if (!ok || y < 1970 || y > 2099 || mo < 1 || mo > 12 || d < 1 || d > (u32)days_in(y, mo)) {
      *err = fm_err_key(arena, key, slc("not a date from 1970-01-01 to 2099-12-31 (YYYY-MM-DD)"));
      return 0;
    }
    data->date = raw; data->has_date = 1;
    return 1;
  }
  if (sl_eq_c(key, "tags")) {
    if (!(raw.n >= 2 && raw.p[0] == '[' && raw.p[raw.n - 1] == ']')) {
      *err = fm_err_key(arena, key, slc("not a list [a, b]")); return 0;
    }
    Slice inner = sl(raw.p + 1, raw.n - 2);
    if (js_trim(inner).n == 0) { data->tags = NULL; data->ntags = 0; data->has_tags = 1; return 1; }
    // Split inner on ','.
    size_t cap = 4, n = 0;
    Slice *tags = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
    size_t start = 0;
    for (size_t i = 0; i <= inner.n; i++) {
      if (i == inner.n || inner.p[i] == ',') {
        Slice piece = js_trim(sl(inner.p + start, i - start));
        if (piece.n == 0) { *err = fm_err_key(arena, key, slc("an empty tag, or one holding \" [ or ]")); return 0; }
        for (size_t k = 0; k < piece.n; k++) {
          u8 c = piece.p[k];
          if (c == '"' || c == '[' || c == ']') { *err = fm_err_key(arena, key, slc("an empty tag, or one holding \" [ or ]")); return 0; }
        }
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *nt = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
          memcpy(nt, tags, n * sizeof(Slice));
          tags = nt; cap = nc;
        }
        tags[n++] = piece;
        start = i + 1;
      }
    }
    data->tags = tags; data->ntags = n; data->has_tags = 1;
    return 1;
  }
  if (sl_eq_c(key, "draft")) {
    if (sl_eq_c(raw, "true")) { data->draft = 1; data->has_draft = 1; return 1; }
    if (sl_eq_c(raw, "false")) { data->draft = 0; data->has_draft = 1; return 1; }
    *err = fm_err_key(arena, key, slc("true or false")); return 0;
  }
  if (sl_eq_c(key, "layout")) {
    int ok = raw.n > 0;
    for (size_t i = 0; ok && i < raw.n; i++) {
      u8 c = raw.p[i];
      if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) ok = 0;
    }
    if (!ok) { *err = fm_err_key(arena, key, slc("not a template name")); return 0; }
    data->layout = raw; data->has_layout = 1;
    return 1;
  }
  if (sl_eq_c(key, "slug")) {
    int ok = raw.n > 0 && raw.n <= 64;
    if (ok) {
      int prev_dash = 1;
      for (size_t i = 0; i < raw.n; i++) {
        u8 c = raw.p[i];
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) prev_dash = 0;
        else if (c == '-' && !prev_dash) prev_dash = 1;
        else { ok = 0; break; }
      }
      if (prev_dash) ok = 0;
    }
    if (!ok) { *err = fm_err_key(arena, key, slc("not a slug")); return 0; }
    data->slug = raw; data->has_slug = 1;
    return 1;
  }
  // Unreachable (KEYS checked by the caller), kept for symmetry.
  Buf b = {0};
  buf_cs(&b, "unknown key \""); buf_sl(&b, key); buf_c(&b, '"');
  *err = arena_slice(arena, b.p, b.len);
  free(b.p);
  return 0;
}

static int fm_has_key(FmData *data, Slice key) {
  if (sl_eq_c(key, "title")) return data->has_title;
  if (sl_eq_c(key, "date")) return data->has_date;
  if (sl_eq_c(key, "tags")) return data->has_tags;
  if (sl_eq_c(key, "draft")) return data->has_draft;
  if (sl_eq_c(key, "layout")) return data->has_layout;
  if (sl_eq_c(key, "slug")) return data->has_slug;
  if (sl_eq_c(key, "summary")) return data->has_summary;
  return 0;
}

static const char *FM_KEYS[7] = { "title", "date", "tags", "draft", "layout", "slug", "summary" };

// Parse front matter: { data, body } or the refusal reason in *err.
// `text` has "\n" line ends. Returns 1 on success.
static int parse_front_matter(Slice text, Arena *arena, Fm *out, Slice *err) {
  if (!(sl_eq_c(text, "---") || sl_starts(text, "---\n"))) { *err = slc("no front matter"); return 0; }
  // Walk the front-matter lines only (one SSE find_byte per line). The
  // body is never scanned: the old version counted/split every line of
  // the whole source and re-scanned for the body start. Collect the FM
  // content lines (between the opening and closing "---") as slices;
  // pass 2 below is unchanged, over fm_lines[i] as line number i + 2.
  size_t cap = 8, nfm = 0;
  Slice *fm_lines = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
  size_t pos = 0, body_start = text.n;
  long close = -1, idx = 0;
  for (;;) {
    size_t nl = find_byte(text.p, text.n, pos, '\n');
    size_t lend = nl < text.n ? nl : text.n;
    if (idx > 0 && lend - pos == 3 && memcmp(text.p + pos, "---", 3) == 0) {
      close = idx;
      body_start = nl < text.n ? nl + 1 : text.n;
      break;
    }
    if (idx > 0) {
      if (nfm == cap) {
        size_t nc = cap * 2;
        Slice *nl2 = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
        memcpy(nl2, fm_lines, nfm * sizeof(Slice));
        fm_lines = nl2; cap = nc;
      }
      fm_lines[nfm++] = sl(text.p + pos, lend - pos);
    }
    if (nl >= text.n) break;
    pos = nl + 1; idx++;
  }
  if (close < 0) { *err = slc("the front matter is not closed"); return 0; }
  // Pass 2: the lines between.
  FmData data; memset(&data, 0, sizeof(data));
  for (size_t li2 = 0; li2 < nfm; li2++) {
    Slice line = fm_lines[li2];
    long line_no = (long)li2 + 2;
    if (is_blank_sp_tab(line) || (line.n > 0 && line.p[0] == '#')) continue;
    size_t ki = 0;
    while (ki < line.n && ((line.p[ki] >= 'a' && line.p[ki] <= 'z') || line.p[ki] == '_')) ki++;
    if (ki == 0 || ki >= line.n || line.p[ki] != ':') {
      char num[64]; int m = snprintf(num, sizeof(num), "line %ld is not \"key: value\"", line_no);
      Buf b = {0}; buf_put(&b, num, (size_t)m);
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    Slice key = sl(line.p, ki);
    Slice raw_full = slc("");
    if (ki + 1 < line.n && line.p[ki + 1] == ' ') raw_full = sl(line.p + ki + 2, line.n - (ki + 2));
    Slice raw = raw_full;
    while (raw.n > 0 && (raw.p[raw.n - 1] == ' ' || raw.p[raw.n - 1] == '\t')) raw.n--;
    int known = 0;
    for (int k = 0; k < 7; k++) if (sl_eq_c(key, FM_KEYS[k])) known = 1;
    if (!known) {
      Buf b = {0}; buf_cs(&b, "unknown key \""); buf_sl(&b, key); buf_c(&b, '"');
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    if (fm_has_key(&data, key)) {
      // Note: no colon here (the reference builds this message directly,
      // unlike value-parser errors which use `key "K": reason`).
      Buf b = {0}; buf_cs(&b, "key \""); buf_sl(&b, key); buf_cs(&b, "\" is given twice");
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    if (raw.n == 0) {
      Buf b = {0}; buf_cs(&b, "key \""); buf_sl(&b, key); buf_cs(&b, "\" has no value");
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    if (!fm_value_of(key, raw, &data, arena, err)) return 0;
  }
  if (!data.has_title) { *err = slc("no title"); return 0; }
  // Body: everything after the closing line's "\n" (already found above).
  out->data = data;
  out->body = sl(text.p + body_start, text.n - body_start);
  return 1;
}



// ---- carried from v5: markdown (v5 lines 1254-1755) ----

typedef struct { u32 level; Slice id, html; } TocEntry;
typedef struct { Slice html; TocEntry *toc; size_t ntoc, captoc; Slice excerpt; } MdOut;

// Link resolution context (per source).
typedef struct {
  Slice dir, src_path, base;
  // by_source: sources sorted by path; binary search.
  struct Source *sources; size_t nsources;
  StrMap *static_paths;
  StrMap *broken; // set of report lines
  Arena *arena;
} LinkCtx;

static Slice *md_split_lines(Arena *arena, Slice src, size_t *out_n) {
  size_t n = 1;
  for (size_t i = 0; i < src.n; i++) if (src.p[i] == '\n') n++;
  Slice *lines = (Slice *)arena_alloc(arena, n * sizeof(Slice));
  size_t li = 0, start = 0;
  for (size_t i = 0; i < src.n; i++) {
    if (src.p[i] == '\n') { lines[li++] = sl(src.p + start, i - start); start = i + 1; }
  }
  lines[li++] = sl(src.p + start, src.n - start);
  *out_n = li;
  return lines;
}

static inline int md_is_blank(Slice line) { return is_blank_sp_tab(line); }

static Slice md_trim_line(Slice line) {
  size_t a = 0, b = line.n;
  while (a < b && (line.p[a] == ' ' || line.p[a] == '\t')) a++;
  while (b > a && (line.p[b - 1] == ' ' || line.p[b - 1] == '\t')) b--;
  return sl(line.p + a, b - a);
}

static inline int md_is_lang_char(u8 c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '-';
}

// Fence-open language, or (int)-1 sentinel via *is_fence + lang out.
static int md_fence_lang(Slice line, Slice *lang) {
  if (line.n < 3 || line.p[0] != '`' || line.p[1] != '`' || line.p[2] != '`') return 0;
  size_t i = 3;
  while (i < line.n && md_is_lang_char(line.p[i])) i++;
  *lang = sl(line.p + 3, i - 3);
  while (i < line.n) {
    if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
    i++;
  }
  return 1;
}

static int md_is_fence_close(Slice line) {
  if (line.n < 3 || line.p[0] != '`' || line.p[1] != '`' || line.p[2] != '`') return 0;
  for (size_t i = 3; i < line.n; i++) if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  return 1;
}

// (depth, text) or 0.
static int md_heading_of(Slice line, u32 *depth, Slice *text) {
  size_t n = 0;
  while (n < 6 && n < line.n && line.p[n] == '#') n++;
  if (n == 0 || n >= line.n || (line.p[n] != ' ' && line.p[n] != '\t')) return 0;
  size_t a = n;
  while (a < line.n && (line.p[a] == ' ' || line.p[a] == '\t')) a++;
  size_t b = line.n;
  while (b > a && (line.p[b - 1] == ' ' || line.p[b - 1] == '\t')) b--;
  if (b <= a) return 0;
  *depth = (u32)n; *text = sl(line.p + a, b - a);
  return 1;
}

static int md_is_hr(Slice line) {
  if (!(sl_starts(line, "---") || sl_starts(line, "***"))) return 0;
  for (size_t i = 3; i < line.n; i++) if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  return 1;
}

static int md_ends_paragraph(Slice line) {
  Slice l; u32 d;
  if (md_is_blank(line)) return 1;
  if (md_fence_lang(line, &l)) return 1;
  if (md_heading_of(line, &d, &l)) return 1;
  if (md_is_hr(line)) return 1;
  return 0;
}

// (number, rest) for an ordered-list line, or 0.
static int md_ordered_of(Slice line, u32 *num, Slice *rest) {
  size_t i = 0;
  while (i < 9 && i < line.n && line.p[i] >= '0' && line.p[i] <= '9') i++;
  if (i == 0 || i + 1 >= line.n || line.p[i] != '.' || line.p[i + 1] != ' ') return 0;
  u32 v = 0;
  for (size_t k = 0; k < i; k++) v = v * 10 + (u32)(line.p[k] - '0');
  *num = v; *rest = sl(line.p + i + 2, line.n - (i + 2));
  return 1;
}

// CONTINUED = /^ {2,}\S/: two or more spaces, then a non-space.
static int md_is_continued(Slice line) {
  size_t i = 0;
  while (i < line.n && line.p[i] == ' ') i++;
  if (i < 2 || i >= line.n) return 0;
  size_t used; u32 cp = utf8_decode(line.p, line.n, i, &used);
  (void)used;
  return !is_js_space(cp);
}

static const u8 MD_ESCAPABLE[] = "\\`*_[]()#+-.!>";

// Whether position j is escaped: the run of '\' immediately before it
// has odd length.
static int md_is_escaped(const u8 *s, size_t from, size_t j) {
  size_t b = j, n = 0;
  while (b > from && s[b - 1] == '\\') { n++; b--; }
  return (n % 2) == 1;
}

static int md_find_unescaped(const u8 *s, size_t n, const u8 *what, size_t wl, size_t from, size_t *out) {
  if (wl == 0 || n < wl) return 0;
  size_t j = find_byte(s, n, from, what[0]);
  if (j >= n) return 0;
  for (;;) {
    if (j + wl <= n && memcmp(s + j, what, wl) == 0 && !md_is_escaped(s, from, j)) { *out = j; return 1; }
    size_t k = find_byte(s, n, j + 1, what[0]);
    if (k >= n) return 0;
    j = k;
  }
}

static int md_find_single_star(const u8 *s, size_t n, size_t from, size_t *out) {
  size_t j = find_byte(s, n, from, '*');
  if (j >= n) return 0;
  for (;;) {
    if (!md_is_escaped(s, from, j)) {
      if (j + 1 < n && s[j + 1] == '*') {
        size_t k = find_byte(s, n, j + 2, '*');
        if (k >= n) return 0;
        j = k; continue;
      }
      *out = j; return 1;
    }
    size_t k = find_byte(s, n, j + 1, '*');
    if (k >= n) return 0;
    j = k;
  }
}

static int md_has_ws(Slice s) {
  size_t i = 0;
  while (i < s.n) {
    size_t used; u32 cp = utf8_decode(s.p, s.n, i, &used);
    if (is_js_space(cp)) return 1;
    i += used;
  }
  return 0;
}

// Forward: link resolution is supplied by the build (LinkCtx).
typedef Slice (*LinkFn)(void *ctx, Slice target);

static void md_inline(Slice s, LinkFn link, void *lctx, Buf *out);

static void md_inline_rec(Slice s, LinkFn link, void *lctx, Buf *out) { md_inline(s, link, lctx, out); }

// SSSE3 "shufti" tables for md_inline's 9 specials: \ ` * [ & < > " '
// MD_LO_TBL[lo] = bitmask of hi nibbles h with ((h<<4)|lo) a special;
// MD_BIT_TBL[hi] = 1<<hi. A byte is special iff
// (MD_LO_TBL[lo] & MD_BIT_TBL[hi]) != 0. Two shuffles + one AND replace
// nine cmpeq + nine movemask per 16 bytes.
static const u8 MD_LO_TBL[16] = { 0x40,0x00,0x04,0x00,0x00,0x00,0x04,0x04,0x00,0x00,0x04,0x20,0x28,0x00,0x08,0x00 };
static const u8 MD_BIT_TBL[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };

static void md_inline(Slice s, LinkFn link, void *lctx, Buf *out) {
  __m128i lo_tbl = _mm_loadu_si128((const __m128i *)MD_LO_TBL);
  __m128i bit_tbl = _mm_loadu_si128((const __m128i *)MD_BIT_TBL);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  size_t n = s.n, i = 0;
  size_t plain = 0; // start of the not-yet-copied plain run
  unsigned mask = 0; size_t base = 0; // pending chunk mask + its base
  while (i < n) {
    size_t j;
    if (mask) {
      j = base + (size_t)__builtin_ctz(mask);
      mask &= mask - 1;
    } else if (i + 16 <= n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(s.p + i));
      __m128i lo = _mm_and_si128(chunk, lowmask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
      __m128i cand = _mm_and_si128(_mm_shuffle_epi8(lo_tbl, lo), _mm_shuffle_epi8(bit_tbl, hi));
      mask = (unsigned)(~_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero))) & 0xFFFFu;
      base = i;
      if (!mask) { i += 16; continue; }
      j = base + (size_t)__builtin_ctz(mask);
      mask &= mask - 1;
    } else {
      // Scalar tail: find the next special byte-wise.
      size_t k = i;
      while (k < n) {
        u8 c = s.p[k];
        if (c == '\\' || c == '`' || c == '*' || c == '[' || c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') break;
        k++;
      }
      if (k >= n) break;
      j = k;
    }
    if (j > plain) buf_put(out, s.p + plain, j - plain);
    i = j; plain = j;
    u8 c = s.p[i];
    switch (c) {
      case '\\': {
        if (i + 1 < n && memchr(MD_ESCAPABLE, s.p[i + 1], sizeof(MD_ESCAPABLE))) {
          u8 ch = s.p[i + 1];
          if (ch == '>') buf_cs(out, "&gt;");
          else buf_c(out, ch);
          i += 2;
        } else { buf_c(out, '\\'); i += 1; }
        break;
      }
      case '`': {
        size_t k = find_byte(s.p, n, i + 1, '`');
        if (k < n && k > i + 1) {
          buf_cs(out, "<code>");
          esc(out, sl(s.p + i + 1, k - (i + 1)));
          buf_cs(out, "</code>");
          i = k + 1;
        } else { buf_c(out, '`'); i += 1; }
        break;
      }
      case '*': {
        if (i + 1 < n && s.p[i + 1] == '*') {
          size_t k;
          if (md_find_unescaped(s.p, n, (const u8 *)"**", 2, i + 2, &k) && k > i + 2) {
            buf_cs(out, "<strong>");
            md_inline_rec(sl(s.p + i + 2, k - (i + 2)), link, lctx, out);
            buf_cs(out, "</strong>");
            i = k + 2;
          } else { buf_cs(out, "**"); i += 2; }
        } else {
          size_t k;
          if (md_find_single_star(s.p, n, i + 1, &k) && k > i + 1) {
            buf_cs(out, "<em>");
            md_inline_rec(sl(s.p + i + 1, k - (i + 1)), link, lctx, out);
            buf_cs(out, "</em>");
            i = k + 1;
          } else { buf_c(out, '*'); i += 1; }
        }
        break;
      }
      case '[': {
        size_t k;
        int has_k = md_find_unescaped(s.p, n, (const u8 *)"]", 1, i + 1, &k);
        size_t m = 0; int has_q = 0;
        if (has_k && k + 1 < n && s.p[k + 1] == '(') {
          size_t q = find_byte(s.p, n, k + 2, ')');
          if (q < n) { m = q; has_q = 1; }
        }
        Slice target = slc("");
        if (has_q) target = sl(s.p + k + 2, m - (k + 2));
        if (has_q && target.n > 0 && !md_has_ws(target)) {
          buf_cs(out, "<a href=\"");
          Slice href = link(lctx, target);
          esc(out, href);
          buf_cs(out, "\">");
          md_inline_rec(sl(s.p + i + 1, k - (i + 1)), link, lctx, out);
          buf_cs(out, "</a>");
          i = m + 1;
        } else { buf_c(out, '['); i += 1; }
        break;
      }
      case '&': buf_cs(out, "&amp;"); i++; break;
      case '<': buf_cs(out, "&lt;"); i++; break;
      case '>': buf_cs(out, "&gt;"); i++; break;
      case '"': buf_cs(out, "&quot;"); i++; break;
      case '\'': buf_cs(out, "&#39;"); i++; break;
      default: buf_c(out, c); i++; break;
    }
    // Keep the pending chunk mask valid: drop bits at or before the new
    // position (the input bytes never change, so later bits stay valid);
    // if the handler consumed past the chunk, discard the mask.
    if (i >= base + 16) mask = 0;
    else if (i > base) mask &= ~((1u << (i - base)) - 1u);
    plain = i;
  }
  if (plain < n) buf_put(out, s.p + plain, n - plain);
}

// { html, toc, excerpt } for a Markdown body. slug_cached calls the
// build's slug cache. `arena` owns the toc/id storage.
typedef Slice (*SlugFn)(void *ctx, Slice text);
static MdOut md_render(Slice src, LinkFn link, void *lctx, SlugFn slugfn, void *sctx, Arena *arena, StrMap *ids) {
  size_t nlines;
  Slice *lines = md_split_lines(arena, src, &nlines);
  Buf html = {0};
  buf_reserve(&html, src.n + src.n / 8 + 64);
  Buf joined = {0}; // scratch: line-join buffer, reused per block (no malloc per block)
  MdOut mo; memset(&mo, 0, sizeof(mo));
  int have_excerpt = 0;
  size_t i = 0;
  while (i < nlines) {
    Slice line = lines[i];
    if (md_is_blank(line)) { i++; continue; }
    Slice lang;
    if (md_fence_lang(line, &lang)) {
      // Collect body lines until close or end.
      size_t cap = 8, nb = 0;
      Slice *body = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
      i++;
      while (i < nlines && !md_is_fence_close(lines[i])) {
        if (nb == cap) {
          size_t nc = cap * 2;
          Slice *nb2 = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
          memcpy(nb2, body, nb * sizeof(Slice));
          body = nb2; cap = nc;
        }
        body[nb++] = lines[i]; i++;
      }
      i++; // past the close (or past the end)
      buf_cs(&html, "<pre><code");
      if (lang.n > 0) {
        buf_cs(&html, " class=\"language-");
        esc(&html, lang);
        buf_c(&html, '"');
      }
      buf_c(&html, '>');
      if (nb > 0) {
        // Join body lines with '\n', escape, trailing '\n'.
        joined.len = 0;
        for (size_t k = 0; k < nb; k++) {
          if (k > 0) buf_c(&joined, '\n');
          buf_sl(&joined, body[k]);
        }
        esc(&html, sl(joined.p, joined.len));
        buf_c(&html, '\n');
      }
      buf_cs(&html, "</code></pre>\n");
      continue;
    }
    u32 depth; Slice htext;
    if (md_heading_of(line, &depth, &htext)) {
      Slice base_slug = slugfn(sctx, htext);
      Slice id = unique_slug(arena, ids, base_slug);
      char hb[16]; int hm = snprintf(hb, sizeof(hb), "<h%u id=\"", depth);
      buf_put(&html, hb, (size_t)hm);
      buf_sl(&html, id);
      buf_cs(&html, "\">");
      size_t h_start = html.len;
      md_inline(htext, link, lctx, &html);
      size_t h_end = html.len;
      hm = snprintf(hb, sizeof(hb), "</h%u>", depth);
      buf_put(&html, hb, (size_t)hm);
      buf_c(&html, '\n');
      if (depth == 2 || depth == 3) {
        if (mo.ntoc == mo.captoc) {
          size_t nc = mo.captoc ? mo.captoc * 2 : 8;
          TocEntry *nt = (TocEntry *)arena_alloc(arena, nc * sizeof(TocEntry));
          if (mo.toc) memcpy(nt, mo.toc, mo.ntoc * sizeof(TocEntry));
          mo.toc = nt; mo.captoc = nc;
        }
        mo.toc[mo.ntoc].level = depth;
        mo.toc[mo.ntoc].id = id;
        mo.toc[mo.ntoc].html = arena_slice(arena, html.p + h_start, h_end - h_start);
        mo.ntoc++;
      }
      i++;
      continue;
    }
    if (md_is_hr(line)) { buf_cs(&html, "<hr>\n"); i++; continue; }
    if (line.n > 0 && line.p[0] == '>') {
      size_t cap = 8, nb = 0;
      Slice *body = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
      while (i < nlines && lines[i].n > 0 && lines[i].p[0] == '>') {
        Slice l = lines[i]; i++;
        Slice stripped = (l.n >= 2 && l.p[1] == ' ') ? sl(l.p + 2, l.n - 2) : sl(l.p + 1, l.n - 1);
        if (nb == cap) {
          size_t nc = cap * 2;
          Slice *nb2 = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
          memcpy(nb2, body, nb * sizeof(Slice));
          body = nb2; cap = nc;
        }
        body[nb++] = stripped;
      }
      // Trim each, drop empties.
      size_t ni = 0;
      for (size_t k = 0; k < nb; k++) {
        Slice t = md_trim_line(body[k]);
        if (t.n > 0) body[ni++] = t;
      }
      if (ni == 0) { buf_cs(&html, "<blockquote>\n</blockquote>\n"); }
      else {
        joined.len = 0;
        for (size_t k = 0; k < ni; k++) {
          if (k > 0) buf_c(&joined, '\n');
          buf_sl(&joined, body[k]);
        }
        buf_cs(&html, "<blockquote>\n<p>");
        md_inline(sl(joined.p, joined.len), link, lctx, &html);
        buf_cs(&html, "</p>\n</blockquote>\n");
      }
      continue;
    }
    int bullet = (line.n > 0 && (line.p[0] == '-' || line.p[0] == '*')) && line.n > 1 && line.p[1] == ' ';
    u32 onum = 0; Slice orest;
    int has_ordered = !bullet && md_ordered_of(line, &onum, &orest);
    if (bullet || has_ordered) {
      int is_ordered = has_ordered;
      u32 start = is_ordered ? onum : 1;
      // Items: each a list of trimmed line slices.
      typedef struct { Slice *parts; size_t n, cap; } Item;
      size_t capi = 8, nitems = 0;
      Item *items = (Item *)arena_alloc(arena, capi * sizeof(Item));
      while (i < nlines) {
        Slice l = lines[i];
        Slice rest; int is_item = 0;
        if (is_ordered) {
          u32 nn; Slice rr;
          if (md_ordered_of(l, &nn, &rr)) { rest = rr; is_item = 1; }
        } else if (l.n > 1 && (l.p[0] == '-' || l.p[0] == '*') && l.p[1] == ' ') {
          rest = sl(l.p + 2, l.n - 2); is_item = 1;
        }
        if (is_item) {
          if (nitems == capi) {
            size_t nc = capi * 2;
            Item *ni2 = (Item *)arena_alloc(arena, nc * sizeof(Item));
            memcpy(ni2, items, nitems * sizeof(Item));
            items = ni2; capi = nc;
          }
          Item *it = &items[nitems++];
          it->n = 0; it->cap = 4;
          it->parts = (Slice *)arena_alloc(arena, it->cap * sizeof(Slice));
          it->parts[it->n++] = md_trim_line(rest);
          i++;
        } else if (md_is_continued(l)) {
          Item *it = &items[nitems - 1];
          if (it->n == it->cap) {
            size_t nc = it->cap * 2;
            Slice *np = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
            memcpy(np, it->parts, it->n * sizeof(Slice));
            it->parts = np; it->cap = nc;
          }
          it->parts[it->n++] = md_trim_line(l);
          i++;
        } else break;
      }
      const char *tag = is_ordered ? "ol" : "ul";
      if (is_ordered && start != 1) {
        char ob[32]; int m = snprintf(ob, sizeof(ob), "<ol start=\"%u\">", start);
        buf_put(&html, ob, (size_t)m);
      } else {
        buf_c(&html, '<'); buf_cs(&html, tag); buf_c(&html, '>');
      }
      buf_c(&html, '\n');
      for (size_t k = 0; k < nitems; k++) {
        joined.len = 0;
        for (size_t q = 0; q < items[k].n; q++) {
          if (q > 0) buf_c(&joined, '\n');
          buf_sl(&joined, items[k].parts[q]);
        }
        buf_cs(&html, "<li>");
        md_inline(sl(joined.p, joined.len), link, lctx, &html);
        buf_cs(&html, "</li>\n");
      }
      buf_cs(&html, "</");
      buf_cs(&html, tag);
      buf_cs(&html, ">\n");
      continue;
    }
    // Paragraph.
    size_t cap = 4, nb = 0;
    Slice *body = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
    body[nb++] = md_trim_line(line);
    i++;
    while (i < nlines && !md_ends_paragraph(lines[i])) {
      if (nb == cap) {
        size_t nc = cap * 2;
        Slice *nb2 = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
        memcpy(nb2, body, nb * sizeof(Slice));
        body = nb2; cap = nc;
      }
      body[nb++] = md_trim_line(lines[i]);
      i++;
    }
    joined.len = 0;
    for (size_t k = 0; k < nb; k++) {
      if (k > 0) buf_c(&joined, '\n');
      buf_sl(&joined, body[k]);
    }
    buf_cs(&html, "<p>");
    size_t p_start = html.len;
    md_inline(sl(joined.p, joined.len), link, lctx, &html);
    size_t p_end = html.len;
    if (!have_excerpt) {
      mo.excerpt = arena_slice(arena, html.p + p_start, p_end - p_start);
      have_excerpt = 1;
    }
    buf_cs(&html, "</p>\n");
  }
  free(joined.p);
  mo.html = sl(html.p, html.len); // malloc'd; kept alive for the case
  return mo;
}



// ---- carried from v5: dir/resolve_path/slice_bounds/sort (v5 lines 2503-2634) ----

static Slice dir_of(Slice path) {
  size_t i = path.n;
  while (i > 0) {
    i--;
    if (path.p[i] == '/') return sl(path.p, i);
    if (i == 0) break;
  }
  return slc("");
}

static inline int is_ascii_alpha(u8 c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static inline int is_ascii_alnum(u8 c) { return is_ascii_alpha(c) || (c >= '0' && c <= '9'); }

static int is_scheme(Slice t) {
  if (t.n == 0 || !is_ascii_alpha(t.p[0])) return 0;
  for (size_t i = 1; i < t.n; i++) {
    u8 c = t.p[i];
    if (c == ':') return 1;
    if (!(is_ascii_alnum(c) || c == '+' || c == '.' || c == '-')) return 0;
  }
  return 0;
}

// Resolve `rel` against `dir` (both '/'-separated). Returns a slice with
// p==NULL when ".." would escape the root.
static Slice resolve_path(Arena *a, Slice dir, Slice rel) {
  size_t cap = 8, n = 0;
  Slice *parts = (Slice *)arena_alloc(a, cap * sizeof(Slice));
  if (dir.n > 0) {
    size_t start = 0;
    for (size_t i = 0; i <= dir.n; i++) {
      if (i == dir.n || dir.p[i] == '/') {
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *np = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(np, parts, n * sizeof(Slice));
          parts = np; cap = nc;
        }
        parts[n++] = sl(dir.p + start, i - start);
        start = i + 1;
      }
    }
  }
  size_t start = 0;
  for (size_t i = 0; i <= rel.n; i++) {
    if (i == rel.n || rel.p[i] == '/') {
      Slice seg = sl(rel.p + start, i - start);
      start = i + 1;
      if (seg.n == 0 || sl_eq_c(seg, ".")) continue;
      if (sl_eq_c(seg, "..")) {
        if (n == 0) return sl(NULL, 0);
        n--;
      } else {
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *np = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(np, parts, n * sizeof(Slice));
          parts = np; cap = nc;
        }
        parts[n++] = seg;
      }
    }
  }
  Buf b = {0};
  for (size_t i = 0; i < n; i++) {
    if (i > 0) buf_c(&b, '/');
    buf_sl(&b, parts[i]);
  }
  Slice r = arena_slice(a, b.p, b.len);
  free(b.p);
  return r;
}

// JS Array.prototype.slice bounds for one end index pair.
static void js_slice_bounds(size_t len, double start, double end, size_t *a, size_t *b) {
  double lf = (double)len;
  double na, nb;
  double t = trunc(start);
  na = t < 0.0 ? fmax(0.0, lf + t) : fmin(t, lf);
  t = trunc(end);
  nb = t < 0.0 ? fmax(0.0, lf + t) : fmin(t, lf);
  if (na > nb) { *a = (size_t)na; *b = (size_t)na; }
  else { *a = (size_t)na; *b = (size_t)nb; }
}

// Stable merge sort of an index array by a comparator over its values.
// (The old insertion sort was O(n^2): ~1.9M cmp_u16 calls per 3-case run
// for 200-600 files. Merge sort is O(n log n) and stable, so the order
// is identical.) Scratch is malloc'd once per call.
typedef int (*IdxCmp)(void *ctx, size_t a, size_t b);
static void sort_indices_rec(size_t *idx, size_t *tmp, size_t n, void *ctx, IdxCmp cmp) {
  if (n < 2) return;
  size_t m = n / 2;
  sort_indices_rec(idx, tmp, m, ctx, cmp);
  sort_indices_rec(idx + m, tmp + m, n - m, ctx, cmp);
  size_t i = 0, j = m, k = 0;
  while (i < m && j < n) {
    if (cmp(ctx, idx[i], idx[j]) <= 0) tmp[k++] = idx[i++];
    else tmp[k++] = idx[j++];
  }
  while (i < m) tmp[k++] = idx[i++];
  while (j < n) tmp[k++] = idx[j++];
  memcpy(idx, tmp, n * sizeof(size_t));
}
static void sort_indices(size_t *idx, size_t n, void *ctx, IdxCmp cmp) {
  if (n < 2) return;
  size_t *tmp = (size_t *)malloc(n * sizeof(size_t));
  sort_indices_rec(idx, tmp, n, ctx, cmp);
  free(tmp);
}

// Stable merge sort of slices by cmp_u16.
static void sort_slices_rec(Slice *a, Slice *tmp, size_t n) {
  if (n < 2) return;
  size_t m = n / 2;
  sort_slices_rec(a, tmp, m);
  sort_slices_rec(a + m, tmp + m, n - m);
  size_t i = 0, j = m, k = 0;
  while (i < m && j < n) {
    if (cmp_u16(a[i], a[j]) <= 0) tmp[k++] = a[i++];
    else tmp[k++] = a[j++];
  }
  while (i < m) tmp[k++] = a[i++];
  while (j < n) tmp[k++] = a[j++];
  memcpy(a, tmp, n * sizeof(Slice));
}
static void sort_slices(Slice *a, size_t n) {
  if (n < 2) return;
  Slice *tmp = (Slice *)malloc(n * sizeof(Slice));
  sort_slices_rec(a, tmp, n);
  free(tmp);
}



// ---- carried from v5: index_url/crlf_to_lf (v5 lines 2891-2920) ----

static Slice index_url(Arena *a, Slice base, u64 k) {
  Buf b = {0};
  buf_sl(&b, base);
  if (k == 1) buf_c(&b, '/');
  else {
    buf_cs(&b, "/page/");
    char kb[24];
    int m = snprintf(kb, sizeof(kb), "%llu", (unsigned long long)k);
    buf_put(&b, kb, (size_t)m);
    buf_c(&b, '/');
  }
  Slice r = arena_slice(a, b.p, b.len);
  free(b.p);
  return r;
}

// \r\n and lone \r become \n. Files without any \r keep their bytes.
static Slice crlf_to_lf(Arena *a, Slice content) {
  if (find_byte(content.p, content.n, 0, '\r') >= content.n) return content;
  u8 *out = (u8 *)arena_alloc(a, content.n ? content.n : 1);
  size_t o = 0, i = 0;
  while (i < content.n) {
    if (content.p[i] == '\r') {
      out[o++] = '\n';
      if (i + 1 < content.n && content.p[i + 1] == '\n') i += 2;
      else i += 1;
    } else out[o++] = content.p[i++];
  }
  return sl(out, o);
}


// ---- carried from v5: tpl tokenizer/compiler (v5 lines 1835-2172) ----

typedef enum { OP_TEXT, OP_VAR, OP_RAW, OP_JIF, OP_JMP, OP_FOR, OP_NEXT, OP_INC, OP_END } OpCode;
// ---------------------------------------------------------------- template parsing

#define INCLUDE_DEPTH 8

typedef struct {
  OpCode op;
  Slice text;              // OP_TEXT
  Slice *segs; size_t nsegs; // OP_VAR/OP_RAW/OP_JIF/OP_FOR paths
  int not;                 // OP_JIF
  int target;              // OP_JIF (else), OP_JMP (end), OP_FOR (after loop)
  Slice name;              // OP_FOR loop variable
  int body;                // OP_NEXT: first body instruction
  int tpl;                 // OP_INC: template index (-1 = missing)
  int cacheable;           // OP_INC
  // ---- v6: constant-folded resolution (filled by resolve_slots) ----
  u8 rkind;                // RK_* for OP_VAR/OP_RAW/OP_JIF/OP_FOR
  i16 rdepth;              // RK_LOOPVAR: compile-time loop depth
  u8 chain[6];             // resolved field codes for segs[1..]
  u8 nchain;               // number of chain entries (0 = the root itself)
} Instr;

typedef struct {
  Slice name;
  Instr *code; size_t n, cap;
  // Cacheability analysis of this template as an include fragment:
  int has_inc;
  int loop_outside; // reads `loop` with no fragment-local for
  Slice *free_names; size_t nfree, capfree;
} Tpl;

static void tpl_emit(Tpl *t, Arena *a, Instr in) {
  if (t->n == t->cap) {
    size_t nc = t->cap ? t->cap * 2 : 32;
    Instr *ni = (Instr *)arena_alloc(a, nc * sizeof(Instr));
    if (t->code) memcpy(ni, t->code, t->n * sizeof(Instr));
    t->code = ni; t->cap = nc;
  }
  t->code[t->n++] = in;
}

static Slice tmpl_err(Arena *a, Slice name, Slice msg) {
  Buf b = {0};
  buf_cs(&b, "template \""); buf_sl(&b, name); buf_cs(&b, "\": "); buf_sl(&b, msg);
  Slice r = arena_slice(a, b.p, b.len);
  free(b.p);
  return r;
}

static inline int is_path_start(u8 c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static inline int is_path_char(u8 c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

// PATH = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/
// On success returns 1 and fills *segs/*nsegs (arena). On failure returns
// 0 and sets *err to `"<json>" is not a path` (with the template prefix).
static int path_of(Slice name, Slice text, Arena *a, Slice **segs, size_t *nsegs, Slice *err) {
  int ok = text.n > 0;
  size_t i = 0;
  while (ok && i < text.n) {
    if (!is_path_start(text.p[i])) { ok = 0; break; }
    i++;
    while (i < text.n && is_path_char(text.p[i])) i++;
    if (i < text.n) {
      if (text.p[i] != '.') { ok = 0; break; }
      i++;
      if (i >= text.n) { ok = 0; break; }
    }
  }
  if (!ok) {
    Buf m = {0};
    json_write_str(&m, text);
    buf_cs(&m, " is not a path");
    *err = tmpl_err(a, name, sl(m.p, m.len));
    free(m.p);
    return 0;
  }
  size_t cap = 4, n = 0;
  Slice *ss = (Slice *)arena_alloc(a, cap * sizeof(Slice));
  size_t start = 0;
  for (size_t k = 0; k <= text.n; k++) {
    if (k == text.n || text.p[k] == '.') {
      if (n == cap) {
        size_t nc = cap * 2;
        Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
        memcpy(ns, ss, n * sizeof(Slice));
        ss = ns; cap = nc;
      }
      ss[n++] = sl(text.p + start, k - start);
      start = k + 1;
    }
  }
  *segs = ss; *nsegs = n;
  return 1;
}

static inline int only_spaces(Slice s, size_t a, size_t b) {
  for (size_t i = a; i < b; i++) if (s.p[i] != ' ' && s.p[i] != '\t') return 0;
  return 1;
}

static size_t find_subslice(const u8 *hay, size_t n, const u8 *needle, size_t nl, size_t from) {
  if (nl == 0 || n < nl) return n;
  for (size_t i = from; i + nl <= n; i++) if (memcmp(hay + i, needle, nl) == 0) return i;
  return n;
}

typedef enum { TK_TEXT, TK_VAR, TK_RAW, TK_TAG } TokKind;
typedef struct {
  TokKind kind;
  Slice text;              // TK_TEXT
  Slice *segs; size_t nsegs; // TK_VAR/TK_RAW
  Slice tag;               // TK_TAG
} Tok;

typedef struct { Tok *toks; size_t n, cap; } TokVec;
static void tok_push(TokVec *v, Arena *a, Tok t) {
  if (v->n == v->cap) {
    size_t nc = v->cap ? v->cap * 2 : 32;
    Tok *nt = (Tok *)arena_alloc(a, nc * sizeof(Tok));
    if (v->toks) memcpy(nt, v->toks, v->n * sizeof(Tok));
    v->toks = nt; v->cap = nc;
  }
  v->toks[v->n++] = t;
}

static int tokenize(Slice name, Slice src, Arena *a, TokVec *out, Slice *err) {
  Buf text = {0};
  size_t i = 0, n = src.n;
  while (i < n) {
    size_t open = find_byte(src.p, n, i, '{');
    if (open >= n || open == n - 1) {
      buf_put(&text, src.p + i, n - i);
      break;
    }
    u8 nx = src.p[open + 1];
    if (nx != '{' && nx != '%') {
      buf_put(&text, src.p + i, open + 1 - i);
      i = open + 1;
      continue;
    }
    buf_put(&text, src.p + i, open - i);
    if (nx == '{') {
      int raw = (open + 2 < n && src.p[open + 2] == '{');
      const u8 *close_pat = raw ? (const u8 *)"}}}" : (const u8 *)"}}";
      size_t cpl = raw ? 3 : 2;
      size_t start = open + cpl;
      size_t close = find_subslice(src.p, n, close_pat, cpl, start);
      if (close >= n) {
        *err = tmpl_err(a, name, slc(raw ? "a {{{ is not closed" : "a {{ is not closed"));
        free(text.p);
        return 0;
      }
      if (text.len > 0) {
        Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TEXT;
        t.text = arena_slice(a, text.p, text.len);
        tok_push(out, a, t);
        text.len = 0;
      }
      Slice inner = js_trim(sl(src.p + start, close - start));
      Slice *segs; size_t nsegs;
      if (!path_of(name, inner, a, &segs, &nsegs, err)) { free(text.p); return 0; }
      Tok t; memset(&t, 0, sizeof(t)); t.kind = raw ? TK_RAW : TK_VAR; t.segs = segs; t.nsegs = nsegs;
      tok_push(out, a, t);
      i = close + cpl;
      continue;
    }
    // {% ... %}
    size_t close = find_subslice(src.p, n, (const u8 *)"%}", 2, open + 2);
    if (close >= n) {
      *err = tmpl_err(a, name, slc("a {% is not closed"));
      free(text.p);
      return 0;
    }
    size_t end = close + 2;
    size_t line_start = 0;
    if (open > 0) {
      // rposition of '\n' in src[..open]
      size_t k = open;
      while (k > 0) { k--; if (src.p[k] == '\n') { line_start = k + 1; break; } if (k == 0) break; }
      // Note: the loop above stops at k==0 without checking src[0] if open>0
      // and no '\n' was found before k==0; line_start stays 0 either way.
    }
    size_t nl = find_byte(src.p, n, end, '\n');
    size_t line_end = nl < n ? nl : n;
    int standalone = only_spaces(src, line_start, open) && only_spaces(src, end, line_end);
    if (standalone) {
      size_t cut = open - line_start;
      text.len -= cut;
    }
    if (text.len > 0) {
      Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TEXT;
      t.text = arena_slice(a, text.p, text.len);
      tok_push(out, a, t);
      text.len = 0;
    }
    Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TAG;
    t.tag = js_trim(sl(src.p + open + 2, close - (open + 2)));
    tok_push(out, a, t);
    i = standalone ? (nl < n ? nl + 1 : n) : end;
  }
  if (text.len > 0) {
    Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TEXT;
    t.text = arena_slice(a, text.p, text.len);
    tok_push(out, a, t);
  }
  free(text.p);
  return 1;
}

static int has_js_ws(Slice s) {
  size_t i = 0;
  while (i < s.n) {
    size_t used; u32 cp = utf8_decode(s.p, s.n, i, &used);
    if (is_js_space(cp)) return 1;
    i += used;
  }
  return 0;
}

// /^for ([A-Za-z_][A-Za-z0-9_]*) in (\S+)$/
static int parse_for(Slice tag, Slice *name, Slice *path) {
  if (!sl_starts(tag, "for ")) return 0;
  Slice rest = sl(tag.p + 4, tag.n - 4);
  size_t i = 0;
  if (i >= rest.n || !is_path_start(rest.p[i])) return 0;
  i++;
  while (i < rest.n && is_path_char(rest.p[i])) i++;
  *name = sl(rest.p, i);
  if (i + 4 <= rest.n && memcmp(rest.p + i, " in ", 4) == 0) {
    Slice p = sl(rest.p + i + 4, rest.n - (i + 4));
    if (p.n > 0 && !has_js_ws(p)) { *path = p; return 1; }
  }
  return 0;
}

// /^if (not )?(\S+)$/
static int parse_if(Slice tag, int *not, Slice *path) {
  if (!sl_starts(tag, "if ")) return 0;
  Slice rest = sl(tag.p + 3, tag.n - 3);
  int nt = 0;
  if (sl_starts(rest, "not ")) { nt = 1; rest = sl(rest.p + 4, rest.n - 4); }
  if (rest.n > 0 && !has_js_ws(rest)) { *not = nt; *path = rest; return 1; }
  return 0;
}

// /^include "([a-z0-9-]+)"$/
static int parse_include(Slice tag, Slice *name) {
  if (!sl_starts(tag, "include \"") || tag.n < 11 || tag.p[tag.n - 1] != '"') return 0;
  Slice inner = sl(tag.p + 9, tag.n - 10);
  if (inner.n == 0) return 0;
  for (size_t i = 0; i < inner.n; i++) {
    u8 c = inner.p[i];
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return 0;
  }
  *name = inner;
  return 1;
}

// Compile tokens to flat bytecode. Returns 1 on success, 0 with *err.
static int compile_template(Slice name, TokVec *toks, Arena *a, Tpl *tpl, Slice *err) {
  memset(tpl, 0, sizeof(*tpl));
  tpl->name = name;
  typedef struct { int kind; int idx; int jmp_idx; int in_else; } CFrame; // kind 0=for 1=if
  size_t capf = 8, nf = 0;
  CFrame *stack = (CFrame *)arena_alloc(a, capf * sizeof(CFrame));
  Instr in;
  for (size_t ti = 0; ti < toks->n; ti++) {
    Tok *tok = &toks->toks[ti];
    memset(&in, 0, sizeof(in));
    switch (tok->kind) {
      case TK_TEXT: in.op = OP_TEXT; in.text = tok->text; tpl_emit(tpl, a, in); break;
      case TK_VAR: in.op = OP_VAR; in.segs = tok->segs; in.nsegs = tok->nsegs; tpl_emit(tpl, a, in); break;
      case TK_RAW: in.op = OP_RAW; in.segs = tok->segs; in.nsegs = tok->nsegs; tpl_emit(tpl, a, in); break;
      case TK_TAG: {
        Slice tag = tok->tag;
        Slice fname, fpath; int not; Slice ipath, iname;
        if (parse_for(tag, &fname, &fpath)) {
          Slice *segs; size_t nsegs;
          if (!path_of(name, fpath, a, &segs, &nsegs, err)) return 0;
          memset(&in, 0, sizeof(in)); in.op = OP_FOR; in.name = fname; in.segs = segs; in.nsegs = nsegs; in.target = -1;
          tpl_emit(tpl, a, in);
          if (nf == capf) { size_t nc = capf * 2; CFrame *ns = (CFrame *)arena_alloc(a, nc * sizeof(CFrame)); memcpy(ns, stack, nf * sizeof(CFrame)); stack = ns; capf = nc; }
          stack[nf].kind = 0; stack[nf].idx = (int)tpl->n - 1; stack[nf].jmp_idx = -1; stack[nf].in_else = 0; nf++;
        } else if (parse_if(tag, &not, &ipath)) {
          Slice *segs; size_t nsegs;
          if (!path_of(name, ipath, a, &segs, &nsegs, err)) return 0;
          memset(&in, 0, sizeof(in)); in.op = OP_JIF; in.not = not; in.segs = segs; in.nsegs = nsegs; in.target = -1;
          tpl_emit(tpl, a, in);
          if (nf == capf) { size_t nc = capf * 2; CFrame *ns = (CFrame *)arena_alloc(a, nc * sizeof(CFrame)); memcpy(ns, stack, nf * sizeof(CFrame)); stack = ns; capf = nc; }
          stack[nf].kind = 1; stack[nf].idx = (int)tpl->n - 1; stack[nf].jmp_idx = -1; stack[nf].in_else = 0; nf++;
        } else if (sl_eq_c(tag, "else")) {
          if (nf == 0 || stack[nf - 1].kind != 1 || stack[nf - 1].in_else) {
            *err = tmpl_err(a, name, slc("an else outside an if")); return 0;
          }
          // JIF jumps to the first instruction of the else branch, which
          // is the instruction AFTER the JMP emitted here.
          memset(&in, 0, sizeof(in)); in.op = OP_JMP; in.target = -1;
          tpl_emit(tpl, a, in);
          tpl->code[stack[nf - 1].idx].target = (int)tpl->n;
          stack[nf - 1].jmp_idx = (int)tpl->n - 1;
          stack[nf - 1].in_else = 1;
        } else if (sl_eq_c(tag, "endfor") || sl_eq_c(tag, "endif")) {
          int want = sl_eq_c(tag, "endfor") ? 0 : 1;
          if (nf == 0 || stack[nf - 1].kind != want) {
            Buf m = {0};
            buf_cs(&m, "an "); buf_sl(&m, tag); buf_cs(&m, " with no open ");
            buf_put(&m, tag.p + 3, tag.n - 3);
            *err = tmpl_err(a, name, sl(m.p, m.len)); free(m.p);
            return 0;
          }
          CFrame fr = stack[--nf];
          if (want == 0) {
            memset(&in, 0, sizeof(in)); in.op = OP_NEXT; in.body = fr.idx + 1;
            tpl_emit(tpl, a, in);
            tpl->code[fr.idx].target = (int)tpl->n;
          } else {
            if (fr.in_else) tpl->code[fr.jmp_idx].target = (int)tpl->n;
            else tpl->code[fr.idx].target = (int)tpl->n;
          }
        } else if (parse_include(tag, &iname)) {
          memset(&in, 0, sizeof(in)); in.op = OP_INC; in.tpl = -2; in.cacheable = 0;
          // Stash the include name in `text` until resolution.
          in.text = iname;
          tpl_emit(tpl, a, in);
        } else {
          Buf m = {0};
          buf_cs(&m, "{% "); buf_sl(&m, tag); buf_cs(&m, " %} is not a tag");
          *err = tmpl_err(a, name, sl(m.p, m.len)); free(m.p);
          return 0;
        }
        break;
      }
    }
  }
  if (nf > 0) {
    Buf m = {0};
    buf_cs(&m, "a ");
    buf_cs(&m, stack[nf - 1].kind == 0 ? "for" : "if");
    buf_cs(&m, " is not closed");
    *err = tmpl_err(a, name, sl(m.p, m.len)); free(m.p);
    return 0;
  }
  memset(&in, 0, sizeof(in)); in.op = OP_END;
  tpl_emit(tpl, a, in);
  return 1;
}


// ---- carried from v5: include resolve/cache analysis (v5 lines 2358-2481) ----
static void resolve_includes(Tpl *tpls, size_t ntpls) {
  for (size_t i = 0; i < ntpls; i++) {
    Tpl *t = &tpls[i];
    for (size_t j = 0; j < t->n; j++) {
      Instr *in = &t->code[j];
      if (in->op == OP_INC) {
        int found = -1;
        for (size_t k = 0; k < ntpls; k++) if (sl_eq(tpls[k].name, in->text)) { found = (int)k; break; }
        in->tpl = found;
      }
    }
  }
}

// Analysis pass A: for template t, compute has_inc / loop_outside / free names.
static void analyze_fragment(Tpl *t, Arena *a) {
  // Loop-name stack (arena array of Slices).
  size_t cap = 8, n = 0;
  Slice *stack = (Slice *)arena_alloc(a, cap * sizeof(Slice));
  t->free_names = NULL; t->nfree = 0; t->capfree = 0;
  for (size_t i = 0; i < t->n; i++) {
    Instr *in = &t->code[i];
    switch (in->op) {
      case OP_VAR: case OP_RAW: case OP_JIF: {
        if (in->nsegs == 0) break;
        Slice f = in->segs[0];
        if (sl_eq_c(f, "loop")) {
          if (n == 0) t->loop_outside = 1;
        } else {
          int bound = 0;
          for (size_t k = 0; k < n; k++) if (sl_eq(stack[k], f)) { bound = 1; break; }
          if (!bound) {
            int seen = 0;
            for (size_t k = 0; k < t->nfree; k++) if (sl_eq(t->free_names[k], f)) { seen = 1; break; }
            if (!seen) {
              if (t->nfree == t->capfree) {
                size_t nc = t->capfree ? t->capfree * 2 : 8;
                Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
                if (t->free_names) memcpy(ns, t->free_names, t->nfree * sizeof(Slice));
                t->free_names = ns; t->capfree = nc;
              }
              t->free_names[t->nfree++] = f;
            }
          }
        }
        break;
      }
      case OP_FOR: {
        if (in->nsegs > 0) {
          Slice f = in->segs[0];
          if (sl_eq_c(f, "loop")) {
            if (n == 0) t->loop_outside = 1;
          } else {
            int bound = 0;
            for (size_t k = 0; k < n; k++) if (sl_eq(stack[k], f)) { bound = 1; break; }
            if (!bound) {
              int seen = 0;
              for (size_t k = 0; k < t->nfree; k++) if (sl_eq(t->free_names[k], f)) { seen = 1; break; }
              if (!seen) {
                if (t->nfree == t->capfree) {
                  size_t nc = t->capfree ? t->capfree * 2 : 8;
                  Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
                  if (t->free_names) memcpy(ns, t->free_names, t->nfree * sizeof(Slice));
                  t->free_names = ns; t->capfree = nc;
                }
                t->free_names[t->nfree++] = f;
              }
            }
          }
        }
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(ns, stack, n * sizeof(Slice));
          stack = ns; cap = nc;
        }
        stack[n++] = in->name;
        break;
      }
      case OP_NEXT: if (n > 0) n--; break;
      case OP_INC: t->has_inc = 1; break;
      default: break;
    }
  }
}

// Analysis pass B: for each template, walk its code with a loop stack and
// mark each OP_INC cacheable when the fragment's free names are all shared
// (site/tags/recent, unshadowed) or the innermost loop variable.
static void analyze_cacheable(Tpl *tpls, size_t ntpls, Arena *a) {
  for (size_t ti = 0; ti < ntpls; ti++) {
    Tpl *t = &tpls[ti];
    size_t cap = 8, n = 0;
    Slice *stack = (Slice *)arena_alloc(a, cap * sizeof(Slice));
    for (size_t i = 0; i < t->n; i++) {
      Instr *in = &t->code[i];
      if (in->op == OP_FOR) {
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(ns, stack, n * sizeof(Slice));
          stack = ns; cap = nc;
        }
        stack[n++] = in->name;
      } else if (in->op == OP_NEXT) {
        if (n > 0) n--;
      } else if (in->op == OP_INC) {
        if (in->tpl < 0) continue;
        Tpl *frag = &tpls[in->tpl];
        if (frag->has_inc || frag->loop_outside) continue;
        int ok = 1;
        for (size_t fi = 0; fi < frag->nfree && ok; fi++) {
          Slice f = frag->free_names[fi];
          int in_loop_stack = 0;
          for (size_t k = 0; k < n; k++) if (sl_eq(stack[k], f)) { in_loop_stack = 1; break; }
          int shared = (sl_eq_c(f, "site") || sl_eq_c(f, "tags") || sl_eq_c(f, "recent")) && !in_loop_stack;
          int innermost = n > 0 && sl_eq(stack[n - 1], f);
          if (!shared && !innermost) ok = 0;
        }
        in->cacheable = ok;
      }
    }
  }
}

// ---- v6: AVX2 32-byte twin of next_json_esc (same classes, same tail) ----
// AVX2 twin of next_json_esc: 32-byte lanes, same return contract.
// Compiled with target("avx2") — the translation unit targets
// x86-64-v2, so the attribute is what makes the AVX2 intrinsics legal
// here; the caller only reaches this function when have_avx2 is set.
__attribute__((target("avx2")))
static size_t next_json_esc_avx2(const u8 *s, size_t n, size_t from) {
  size_t i = from;
  __m256i quote = _mm256_set1_epi8('"');
  __m256i backslash = _mm256_set1_epi8('\\');
  __m256i lead_ed = _mm256_set1_epi8((char)0xED);
  __m256i lo31 = _mm256_set1_epi8(0x1F);
  __m256i zero = _mm256_setzero_si256();
  while (i + 32 <= n) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(s + i));
    __m256i m = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(chunk, quote), _mm256_cmpeq_epi8(chunk, backslash)),
                              _mm256_or_si256(_mm256_cmpeq_epi8(chunk, lead_ed),
                                           _mm256_cmpeq_epi8(_mm256_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm256_movemask_epi8(m);
    if (mask) return i + (size_t)__builtin_ctz(mask);
    i += 32;
  }
  if (i < n && n >= 32) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(s + n - 32));
    __m256i m = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(chunk, quote), _mm256_cmpeq_epi8(chunk, backslash)),
                              _mm256_or_si256(_mm256_cmpeq_epi8(chunk, lead_ed),
                                           _mm256_cmpeq_epi8(_mm256_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm256_movemask_epi8(m);
    size_t skip = i - (n - 32);
    mask &= ~((1u << skip) - 1u);
    if (mask) return (n - 32) + (size_t)__builtin_ctz(mask);
    return n;
  }
  while (i < n) {
    u8 c = s[i];
    if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) return i;
    i++;
  }
  return n;
}

// ------------------------------------------------------------------ v6 flat JSON parser
// Parses cases.json straight into flat structs — no Json tree, no
// per-value allocation. Strings without escapes are direct slices of
// the input buffer (which outlives the whole buildCases call); decoded
// strings go to the parse arena. The grammar scan mirrors v5's
// j_number/j_string exactly (same acceptance, same doubles via strtod).
typedef struct { Slice name, src; } TplSrc;
typedef struct { Slice path, content; } InFile;
typedef struct {
  Slice title, base_url;
  double per_page, recent;
  int has_title, has_base_url, has_per_page, has_recent;
} SiteIn;
typedef struct {
  SiteIn site;
  TplSrc *tpls; size_t ntpls;
  InFile *files; size_t nfiles;
} CaseIn;
typedef struct { CaseIn *items; Slice *ids; size_t n; } CaseVec;

static inline void fp_ws(JParser *p) {
  while (p->pos < p->len) {
    u8 c = p->b[p->pos];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
    else break;
  }
}

static int fp_lit(JParser *p, const char *lit) {
  size_t n = strlen(lit);
  if (p->pos + n <= p->len && memcmp(p->b + p->pos, lit, n) == 0) { p->pos += n; return 1; }
  return 0;
}

// Parse a JSON string (pos at the opening '"'). Same zero-copy fast
// path as v5's j_string.
static int fp_string(JParser *p, Slice *out) {
  size_t start = p->pos + 1;
  size_t j = next_json_esc(p->b, p->len, start);
  if (j < p->len && p->b[j] == '"') {
    *out = sl(p->b + start, j - start);
    p->pos = j + 1;
    return 1;
  }
  p->pos++;
  Buf tmp = {0};
  for (;;) {
    size_t k = next_json_esc(p->b, p->len, p->pos);
    if (k > p->pos) { buf_put(&tmp, p->b + p->pos, k - p->pos); p->pos = k; }
    if (p->pos >= p->len) { free(tmp.p); return 0; }
    u8 c = p->b[p->pos];
    if (c == '"') { p->pos++; break; }
    if (c == '\\') {
      p->pos++;
      if (p->pos >= p->len) { free(tmp.p); return 0; }
      u8 e = p->b[p->pos]; p->pos++;
      switch (e) {
        case '"': buf_c(&tmp, '"'); break;
        case '\\': buf_c(&tmp, '\\'); break;
        case '/': buf_c(&tmp, '/'); break;
        case 'b': buf_c(&tmp, 0x08); break;
        case 'f': buf_c(&tmp, 0x0C); break;
        case 'n': buf_c(&tmp, '\n'); break;
        case 'r': buf_c(&tmp, '\r'); break;
        case 't': buf_c(&tmp, '\t'); break;
        case 'u': {
          u32 cp;
          if (!j_hex4(p, &cp)) { free(tmp.p); return 0; }
          if (cp >= 0xD800 && cp < 0xDC00) {
            if (p->pos + 1 < p->len && p->b[p->pos] == '\\' && p->b[p->pos + 1] == 'u') {
              size_t save = p->pos;
              p->pos += 2;
              u32 lo;
              if (j_hex4(p, &lo) && lo >= 0xDC00 && lo < 0xE000) {
                utf8_encode(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00), &tmp);
              } else { p->pos = save; utf8_encode(cp, &tmp); }
            } else utf8_encode(cp, &tmp);
          } else utf8_encode(cp, &tmp);
          break;
        }
        default: free(tmp.p); return 0;
      }
    } else if (c < 0x20) { free(tmp.p); return 0; }
    else { buf_c(&tmp, c); p->pos++; }
  }
  *out = arena_slice(p->arena, tmp.p ? tmp.p : (const u8 *)"", tmp.len);
  free(tmp.p);
  return 1;
}

static int fp_number(JParser *p, double *out) {
  size_t start = p->pos;
  if (p->pos < p->len && p->b[p->pos] == '-') p->pos++;
  if (p->pos >= p->len) return 0;
  if (p->b[p->pos] == '0') p->pos++;
  else if (p->b[p->pos] >= '1' && p->b[p->pos] <= '9') {
    while (p->pos < p->len && p->b[p->pos] >= '0' && p->b[p->pos] <= '9') p->pos++;
  } else return 0;
  if (p->pos < p->len && p->b[p->pos] == '.') {
    p->pos++;
    if (p->pos >= p->len || p->b[p->pos] < '0' || p->b[p->pos] > '9') return 0;
    while (p->pos < p->len && p->b[p->pos] >= '0' && p->b[p->pos] <= '9') p->pos++;
  }
  if (p->pos < p->len && (p->b[p->pos] == 'e' || p->b[p->pos] == 'E')) {
    p->pos++;
    if (p->pos < p->len && (p->b[p->pos] == '+' || p->b[p->pos] == '-')) p->pos++;
    if (p->pos >= p->len || p->b[p->pos] < '0' || p->b[p->pos] > '9') return 0;
    while (p->pos < p->len && p->b[p->pos] >= '0' && p->b[p->pos] <= '9') p->pos++;
  }
  char numbuf[64];
  size_t n = p->pos - start;
  if (n >= sizeof(numbuf)) return 0;
  memcpy(numbuf, p->b + start, n); numbuf[n] = 0;
  *out = strtod(numbuf, NULL);
  return 1;
}

static int fp_skip_value(JParser *p, int depth) {
  if (depth > 400 || p->pos >= p->len) return 0;
  u8 c = p->b[p->pos];
  switch (c) {
    case '{': {
      p->pos++;
      fp_ws(p);
      if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 1; }
      for (;;) {
        fp_ws(p);
        if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
        Slice k;
        if (!fp_string(p, &k)) return 0;
        fp_ws(p);
        if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
        p->pos++;
        fp_ws(p);
        if (!fp_skip_value(p, depth + 1)) return 0;
        fp_ws(p);
        if (p->pos >= p->len) return 0;
        if (p->b[p->pos] == ',') { p->pos++; continue; }
        if (p->b[p->pos] == '}') { p->pos++; return 1; }
        return 0;
      }
    }
    case '[': {
      p->pos++;
      fp_ws(p);
      if (p->pos < p->len && p->b[p->pos] == ']') { p->pos++; return 1; }
      for (;;) {
        fp_ws(p);
        if (!fp_skip_value(p, depth + 1)) return 0;
        fp_ws(p);
        if (p->pos >= p->len) return 0;
        if (p->b[p->pos] == ',') { p->pos++; continue; }
        if (p->b[p->pos] == ']') { p->pos++; return 1; }
        return 0;
      }
    }
    case '"': { Slice s; return fp_string(p, &s); }
    case 't': return fp_lit(p, "true");
    case 'f': return fp_lit(p, "false");
    case 'n': return fp_lit(p, "null");
    default: {
      double d;
      if (c == '-' || (c >= '0' && c <= '9')) return fp_number(p, &d);
      return 0;
    }
  }
}

static int fp_parse_site(JParser *p, SiteIn *s) {
  memset(s, 0, sizeof(*s));
  if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
  p->pos++;
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    Slice k;
    if (!fp_string(p, &k)) return 0;
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
    p->pos++;
    fp_ws(p);
    if (sl_eq_c(k, "title")) {
      if (!fp_string(p, &s->title)) return 0;
      s->has_title = 1;
    } else if (sl_eq_c(k, "base_url")) {
      if (!fp_string(p, &s->base_url)) return 0;
      s->has_base_url = 1;
    } else if (sl_eq_c(k, "per_page")) {
      if (!fp_number(p, &s->per_page)) return 0;
      s->has_per_page = 1;
    } else if (sl_eq_c(k, "recent")) {
      if (!fp_number(p, &s->recent)) return 0;
      s->has_recent = 1;
    } else {
      if (!fp_skip_value(p, 0)) return 0;
    }
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == '}') { p->pos++; return 1; }
    return 0;
  }
}

static int fp_parse_templates(JParser *p, TplSrc **out, size_t *nout) {
  if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
  p->pos++;
  size_t cap = 16, n = 0;
  TplSrc *arr = (TplSrc *)arena_alloc(p->arena, cap * sizeof(TplSrc));
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; *out = arr; *nout = 0; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    Slice k;
    if (!fp_string(p, &k)) return 0;
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
    p->pos++;
    fp_ws(p);
    Slice v;
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    if (!fp_string(p, &v)) return 0;
    if (n == cap) {
      size_t nc = cap * 2;
      TplSrc *na = (TplSrc *)arena_alloc(p->arena, nc * sizeof(TplSrc));
      memcpy(na, arr, n * sizeof(TplSrc));
      arr = na; cap = nc;
    }
    arr[n].name = k; arr[n].src = v; n++;
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == '}') { p->pos++; *out = arr; *nout = n; return 1; }
    return 0;
  }
}

static int fp_parse_files(JParser *p, InFile **out, size_t *nout) {
  if (p->pos >= p->len || p->b[p->pos] != '[') return 0;
  p->pos++;
  size_t cap = 64, n = 0;
  InFile *arr = (InFile *)arena_alloc(p->arena, cap * sizeof(InFile));
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == ']') { p->pos++; *out = arr; *nout = 0; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
    p->pos++;
    Slice path = slc(""), content = slc("");
    int has_path = 0, has_content = 0;
    fp_ws(p);
    if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 0; }
    for (;;) {
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
      Slice k;
      if (!fp_string(p, &k)) return 0;
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
      p->pos++;
      fp_ws(p);
      if (sl_eq_c(k, "path")) {
        if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
        if (!fp_string(p, &path)) return 0;
        has_path = 1;
      } else if (sl_eq_c(k, "content")) {
        if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
        if (!fp_string(p, &content)) return 0;
        has_content = 1;
      } else {
        if (!fp_skip_value(p, 0)) return 0;
      }
      fp_ws(p);
      if (p->pos >= p->len) return 0;
      if (p->b[p->pos] == ',') { p->pos++; continue; }
      if (p->b[p->pos] == '}') { p->pos++; break; }
      return 0;
    }
    if (!has_path || !has_content) return 0;
    if (n == cap) {
      size_t nc = cap * 2;
      InFile *na = (InFile *)arena_alloc(p->arena, nc * sizeof(InFile));
      memcpy(na, arr, n * sizeof(InFile));
      arr = na; cap = nc;
    }
    arr[n].path = path; arr[n].content = content; n++;
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == ']') { p->pos++; *out = arr; *nout = n; return 1; }
    return 0;
  }
}

static int fp_parse_case_input(JParser *p, CaseIn *c) {
  // Consumes the input object's own opening '{' (the caller only
  // consumed the enclosing case object's '{', if any).
  memset(c, 0, sizeof(*c));
  if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
  p->pos++;
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    Slice k;
    if (!fp_string(p, &k)) return 0;
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
    p->pos++;
    fp_ws(p);
    if (sl_eq_c(k, "site")) {
      if (!fp_parse_site(p, &c->site)) return 0;
    } else if (sl_eq_c(k, "templates")) {
      if (!fp_parse_templates(p, &c->tpls, &c->ntpls)) return 0;
    } else if (sl_eq_c(k, "files")) {
      if (!fp_parse_files(p, &c->files, &c->nfiles)) return 0;
    } else {
      if (!fp_skip_value(p, 0)) return 0;
    }
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == '}') { p->pos++; return 1; }
    return 0;
  }
}

static int fp_parse_cases(const char *json, size_t jlen, Arena *arena, CaseVec *out) {
  JParser pv = { (const u8 *)json, jlen, 0, arena };
  JParser *p = &pv;
  if (p->pos >= p->len || p->b[p->pos] != '[') return 0;
  p->pos++;
  size_t cap = 4, n = 0;
  CaseIn *arr = (CaseIn *)arena_alloc(arena, cap * sizeof(CaseIn));
  Slice *ids = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == ']') { p->pos++; out->items = arr; out->ids = ids; out->n = 0; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
    p->pos++;
    CaseIn ci;
    memset(&ci, 0, sizeof(ci));
    Slice id = slc("");
    int has_input = 0, has_id = 0;
    fp_ws(p);
    if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 0; }
    for (;;) {
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
      Slice k;
      if (!fp_string(p, &k)) return 0;
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
      p->pos++;
      fp_ws(p);
      if (sl_eq_c(k, "input")) {
        if (!fp_parse_case_input(p, &ci)) return 0;
        has_input = 1;
      } else if (sl_eq_c(k, "id")) {
        if (!fp_string(p, &id)) return 0;
        has_id = 1;
      } else {
        if (!fp_skip_value(p, 0)) return 0;
      }
      fp_ws(p);
      if (p->pos >= p->len) return 0;
      if (p->b[p->pos] == ',') { p->pos++; continue; }
      if (p->b[p->pos] == '}') { p->pos++; break; }
      return 0;
    }
    if (!has_input || !has_id) return 0;
    if (n == cap) {
      size_t nc = cap * 2;
      CaseIn *na = (CaseIn *)arena_alloc(arena, nc * sizeof(CaseIn));
      memcpy(na, arr, n * sizeof(CaseIn));
      arr = na;
      Slice *ni = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
      memcpy(ni, ids, n * sizeof(Slice));
      ids = ni; cap = nc;
    }
    arr[n] = ci;
    ids[n] = id;
    n++;
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == ']') { p->pos++; out->items = arr; out->ids = ids; out->n = n; return 1; }
    return 0;
  }
}

// ------------------------------------------------------------------ v6 flat values
// No Val tree: template values are small by-value registers (RVal)
// pointing at precomputed flat records. A "card" is just the source
// record itself; a "tag" is the Tag struct; the page object is the
// record. Nothing is allocated per card/page render.

// Field codes (resolved at template compile time).
enum {
  F_TITLE = 1, F_URL, F_DATE, F_DATE_LONG, F_SUMMARY, F_EXCERPT,
  F_READING_MINUTES, F_TAGS, F_SLUG, F_CONTENT, F_TOC, F_NEWER, F_OLDER,
  F_IS_POST, F_NAME, F_COUNT, F_POSTS, F_LEVEL, F_ID, F_HTML,
  F_NUMBER, F_TOTAL, F_NEWER_URL, F_OLDER_URL, F_LENGTH,
  F_INDEX, F_FIRST, F_LAST, F_BASE_URL
};
// Compile-time value types.
enum { T_UNK, T_STR, T_NUM, T_BOOL, T_ARR_REC, T_ARR_TAG, T_ARR_TAGREF, T_ARR_TOC, T_REC, T_TAG, T_TAGREF, T_TOC, T_PAGER, T_SITE, T_LOOP };
// Resolved root kinds (Instr.rkind).
enum { RK_NOTFOUND = 0, RK_DYN, RK_SITE, RK_TAGS, RK_RECENT, RK_PAGE, RK_POSTS, RK_PAGER, RK_TAG, RK_LOOP, RK_LOOPVAR };
// Runtime value kinds (RVal.kind).
enum { RV_UNDEF = 0, RV_NULL, RV_BOOL, RV_NUM, RV_STR, RV_ARR, RV_REC, RV_TAG, RV_TAGREF, RV_TOC, RV_PAGER, RV_SITE, RV_LOOP };
// Array element kinds (RVal.elem for RV_ARR).
enum { AE_REC_IDX = 1, AE_TAG, AE_TAG_PTR, AE_TAGREF, AE_TOC };

typedef struct {
  u8 kind;
  Slice s;        // RV_STR
  double num;     // RV_NUM
  int boolean;    // RV_BOOL
  const void *ptr; // RV_REC/TAG/TAGREF/TOC/PAGER/SITE/LOOP, or RV_ARR items
  size_t len;     // RV_ARR length
  u8 elem;        // RV_ARR element kind
} RVal;

typedef struct { Slice name, slug, url; } TagRef;

typedef struct {
  Slice path;
  FmData data;
  Slice body;      // after front-matter strip
  int post;
  Slice layout;
  Slice slug, out_dir, url;
  Slice html;      // markdown html (malloc'd buffer; leaked per case)
  TocEntry *toc; size_t ntoc;
  Slice excerpt;
  Slice date_long; // precomputed ("" when no date)
  double reading_minutes; // precomputed
  int newer, older; // source indices, -1 = none
  TagRef *tag_refs; size_t ntag_refs, captag_refs;
} CRec;

typedef struct {
  Slice name, slug, url;
  size_t *posts; size_t nposts, capposts; // source indices
} Tag;

typedef struct { double number, total; Slice url, newer_url, older_url; } Pager;

typedef struct LoopState LoopState;
struct LoopState {
  RVal arr;          // the array being iterated (kind RV_ARR)
  size_t idx;        // current index
  size_t len;        // cached = arr.len
  Slice var_name;
  u32 saved_shadow;  // shadow mask before this loop
};

typedef struct {
  Arena *arena;
  Slice *tpl_names; Tpl *tpls; int *tpl_ok; size_t ntpls;
  StrMap *slug_cache, *date_cache, *card_cache;
  CRec *sources; size_t nsources, capsources;
  struct { Slice path, content; } *statics; size_t nstatics, capstatics;
  StrMap *by_source, *static_paths, *broken;
  Slice base;        // base_url with trailing '/' trimmed
  Buf *out;
  int first;         // emit comma tracking
  Slice *report; size_t nreport, capreport;
  Buf page_buf;      // reusable per-page render buffer
  SiteIn *site;
  RVal tags_rv, recent_rv; // per-case globals (arrays)
} BuildCtx;

static int field_code_for(Slice seg) {
  // Linear over the field names (only used for DYN paths and at
  // compile time — never in the hot render loop).
  static const struct { const char *n; int f; } tab[] = {
    {"title", F_TITLE}, {"url", F_URL}, {"date", F_DATE}, {"date_long", F_DATE_LONG},
    {"summary", F_SUMMARY}, {"excerpt", F_EXCERPT}, {"reading_minutes", F_READING_MINUTES},
    {"tags", F_TAGS}, {"slug", F_SLUG}, {"content", F_CONTENT}, {"toc", F_TOC},
    {"newer", F_NEWER}, {"older", F_OLDER}, {"is_post", F_IS_POST}, {"name", F_NAME},
    {"count", F_COUNT}, {"posts", F_POSTS}, {"level", F_LEVEL}, {"id", F_ID},
    {"html", F_HTML}, {"number", F_NUMBER}, {"total", F_TOTAL},
    {"newer_url", F_NEWER_URL}, {"older_url", F_OLDER_URL}, {"length", F_LENGTH},
    {"index", F_INDEX}, {"first", F_FIRST}, {"last", F_LAST}, {"base_url", F_BASE_URL},
  };
  for (size_t i = 0; i < sizeof(tab) / sizeof(tab[0]); i++)
    if (sl_eq_c(seg, tab[i].n)) return tab[i].f;
  return 0;
}

// One field step on a runtime value. Returns 0 when the field does
// not exist on that value (v5's obj_get returning 0).
static int rval_step(BuildCtx *ctx, RVal v, int f, RVal *out) {
  memset(out, 0, sizeof(*out));
  switch (v.kind) {
    case RV_REC: {
      CRec *s = (CRec *)v.ptr;
      switch (f) {
        case F_TITLE: out->kind = RV_STR; out->s = s->data.title; return 1;
        case F_URL: out->kind = RV_STR; out->s = s->url; return 1;
        case F_DATE: out->kind = RV_STR; out->s = s->data.has_date ? s->data.date : slc(""); return 1;
        case F_DATE_LONG: out->kind = RV_STR; out->s = s->date_long; return 1;
        case F_SUMMARY: out->kind = RV_STR; out->s = s->data.has_summary ? s->data.summary : slc(""); return 1;
        case F_EXCERPT: out->kind = RV_STR; out->s = s->excerpt; return 1;
        case F_READING_MINUTES: out->kind = RV_NUM; out->num = s->reading_minutes; return 1;
        case F_TAGS: out->kind = RV_ARR; out->ptr = s->tag_refs; out->len = s->ntag_refs; out->elem = AE_TAGREF; return 1;
        case F_SLUG: out->kind = RV_STR; out->s = s->slug; return 1;
        case F_CONTENT: out->kind = RV_STR; out->s = s->html; return 1;
        case F_TOC: out->kind = RV_ARR; out->ptr = s->toc; out->len = s->ntoc; out->elem = AE_TOC; return 1;
        case F_NEWER:
          if (s->newer < 0) { out->kind = RV_NULL; return 1; }
          out->kind = RV_REC; out->ptr = &ctx->sources[s->newer]; return 1;
        case F_OLDER:
          if (s->older < 0) { out->kind = RV_NULL; return 1; }
          out->kind = RV_REC; out->ptr = &ctx->sources[s->older]; return 1;
        case F_IS_POST: out->kind = RV_BOOL; out->boolean = s->post; return 1;
        default: return 0;
      }
    }
    case RV_TAG: {
      Tag *t = (Tag *)v.ptr;
      switch (f) {
        case F_NAME: out->kind = RV_STR; out->s = t->name; return 1;
        case F_SLUG: out->kind = RV_STR; out->s = t->slug; return 1;
        case F_URL: out->kind = RV_STR; out->s = t->url; return 1;
        case F_COUNT: out->kind = RV_NUM; out->num = (double)t->nposts; return 1;
        case F_POSTS: out->kind = RV_ARR; out->ptr = t->posts; out->len = t->nposts; out->elem = AE_REC_IDX; return 1;
        default: return 0;
      }
    }
    case RV_TAGREF: {
      TagRef *t = (TagRef *)v.ptr;
      switch (f) {
        case F_NAME: out->kind = RV_STR; out->s = t->name; return 1;
        case F_SLUG: out->kind = RV_STR; out->s = t->slug; return 1;
        case F_URL: out->kind = RV_STR; out->s = t->url; return 1;
        default: return 0;
      }
    }
    case RV_TOC: {
      TocEntry *t = (TocEntry *)v.ptr;
      switch (f) {
        case F_LEVEL: out->kind = RV_NUM; out->num = (double)t->level; return 1;
        case F_ID: out->kind = RV_STR; out->s = t->id; return 1;
        case F_HTML: out->kind = RV_STR; out->s = t->html; return 1;
        default: return 0;
      }
    }
    case RV_PAGER: {
      Pager *pg = (Pager *)v.ptr;
      switch (f) {
        case F_NUMBER: out->kind = RV_NUM; out->num = pg->number; return 1;
        case F_TOTAL: out->kind = RV_NUM; out->num = pg->total; return 1;
        case F_URL: out->kind = RV_STR; out->s = pg->url; return 1;
        case F_NEWER_URL: out->kind = RV_STR; out->s = pg->newer_url; return 1;
        case F_OLDER_URL: out->kind = RV_STR; out->s = pg->older_url; return 1;
        default: return 0;
      }
    }
    case RV_SITE: {
      BuildCtx *c = (BuildCtx *)v.ptr;
      switch (f) {
        case F_TITLE: out->kind = RV_STR; out->s = c->site->title; return 1;
        case F_BASE_URL: out->kind = RV_STR; out->s = c->base; return 1;
        default: return 0;
      }
    }
    case RV_ARR:
      if (f == F_LENGTH) { out->kind = RV_NUM; out->num = (double)v.len; return 1; }
      return 0;
    case RV_LOOP: {
      LoopState *ls = (LoopState *)v.ptr;
      switch (f) {
        case F_INDEX: out->kind = RV_NUM; out->num = (double)ls->idx + 1.0; return 1;
        case F_FIRST: out->kind = RV_BOOL; out->boolean = (ls->idx == 0); return 1;
        case F_LAST: out->kind = RV_BOOL; out->boolean = (ls->idx + 1 >= ls->len); return 1;
        case F_LENGTH: out->kind = RV_NUM; out->num = (double)ls->len; return 1;
        default: return 0;
      }
    }
    default: return 0;
  }
}

// The item of an array RVal at index idx.
static int arr_item(BuildCtx *ctx, RVal *arr, size_t idx, RVal *out) {
  if (arr->kind != RV_ARR || idx >= arr->len) return 0;
  memset(out, 0, sizeof(*out));
  switch (arr->elem) {
    case AE_REC_IDX: {
      size_t si = ((const size_t *)arr->ptr)[idx];
      out->kind = RV_REC; out->ptr = &ctx->sources[si];
      return 1;
    }
    case AE_TAG:
      out->kind = RV_TAG; out->ptr = &((const Tag *)arr->ptr)[idx];
      return 1;
    case AE_TAG_PTR:
      out->kind = RV_TAG; out->ptr = ((Tag *const *)arr->ptr)[idx];
      return 1;
    case AE_TAGREF:
      out->kind = RV_TAGREF; out->ptr = &((const TagRef *)arr->ptr)[idx];
      return 1;
    case AE_TOC:
      out->kind = RV_TOC; out->ptr = &((const TocEntry *)arr->ptr)[idx];
      return 1;
    default: return 0;
  }
}

// ------------------------------------------------------------------ v6 slot compiler
// Constant folding: every template path is resolved at compile time to
// a root kind + a chain of field codes, so the hot render loop does no
// string comparison and no frame search for the common paths. Anything
// that cannot be resolved statically stays RK_DYN and is looked up at
// runtime (free names of include fragments, loop variables of unknown
// item type, over-long paths).

typedef struct { Slice name; int item_type; } CLoopVar;

// Map one field step from a current compile-time type. Returns 0 when
// the field does not exist on that type (the whole path is then
// NOTFOUND, exactly like v5's lookup returning 0).
static int field_step_type(int type, Slice seg, int *field_out, int *next_out) {
  int f = field_code_for(seg);
  if (!f) return 0;
  switch (type) {
    case T_REC:
      switch (f) {
        case F_TITLE: case F_URL: case F_DATE: case F_DATE_LONG:
        case F_SUMMARY: case F_EXCERPT: case F_SLUG: case F_CONTENT:
          *field_out = f; *next_out = T_STR; return 1;
        case F_READING_MINUTES: *field_out = f; *next_out = T_NUM; return 1;
        case F_TAGS: *field_out = f; *next_out = T_ARR_TAGREF; return 1;
        case F_TOC: *field_out = f; *next_out = T_ARR_TOC; return 1;
        case F_NEWER: case F_OLDER: *field_out = f; *next_out = T_REC; return 1;
        case F_IS_POST: *field_out = f; *next_out = T_BOOL; return 1;
        default: return 0;
      }
    case T_TAG:
      switch (f) {
        case F_NAME: case F_SLUG: case F_URL: *field_out = f; *next_out = T_STR; return 1;
        case F_COUNT: *field_out = f; *next_out = T_NUM; return 1;
        case F_POSTS: *field_out = f; *next_out = T_ARR_REC; return 1;
        default: return 0;
      }
    case T_TAGREF:
      switch (f) {
        case F_NAME: case F_SLUG: case F_URL: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_TOC:
      switch (f) {
        case F_LEVEL: *field_out = f; *next_out = T_NUM; return 1;
        case F_ID: case F_HTML: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_PAGER:
      switch (f) {
        case F_NUMBER: case F_TOTAL: *field_out = f; *next_out = T_NUM; return 1;
        case F_URL: case F_NEWER_URL: case F_OLDER_URL: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_SITE:
      switch (f) {
        case F_TITLE: case F_BASE_URL: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_LOOP:
      switch (f) {
        case F_INDEX: case F_LENGTH: *field_out = f; *next_out = T_NUM; return 1;
        case F_FIRST: case F_LAST: *field_out = f; *next_out = T_BOOL; return 1;
        default: return 0;
      }
    case T_ARR_REC: case T_ARR_TAG: case T_ARR_TAGREF: case T_ARR_TOC:
      if (f == F_LENGTH) { *field_out = f; *next_out = T_NUM; return 1; }
      return 0;
    default: return 0;
  }
}

// Resolve a path's root (first segment) against the compile-time loop
// stack and the fixed roots. Returns the root's value type.
static int resolve_root(Slice s0, CLoopVar *lv, size_t nlv, u8 *rk, i16 *rdepth) {
  if (sl_eq_c(s0, "loop")) { *rk = RK_LOOP; *rdepth = 0; return T_LOOP; }
  for (size_t i = nlv; i-- > 0;) {
    if (sl_eq(lv[i].name, s0)) { *rk = RK_LOOPVAR; *rdepth = (i16)i; return lv[i].item_type; }
  }
  if (sl_eq_c(s0, "site")) { *rk = RK_SITE; return T_SITE; }
  if (sl_eq_c(s0, "tags")) { *rk = RK_TAGS; return T_ARR_TAG; }
  if (sl_eq_c(s0, "recent")) { *rk = RK_RECENT; return T_ARR_REC; }
  if (sl_eq_c(s0, "page")) { *rk = RK_PAGE; return T_REC; }
  if (sl_eq_c(s0, "posts")) { *rk = RK_POSTS; return T_ARR_REC; }
  if (sl_eq_c(s0, "pager")) { *rk = RK_PAGER; return T_PAGER; }
  if (sl_eq_c(s0, "tag")) { *rk = RK_TAG; return T_TAG; }
  *rk = RK_DYN; return T_UNK;
}

// Resolve one path into an instruction's slot fields. Returns the value
// type of the resolved path (T_UNK for DYN/NOTFOUND).
static int resolve_path_slots(Slice *segs, size_t nsegs, CLoopVar *lv, size_t nlv, Instr *in) {
  in->rkind = RK_NOTFOUND; in->rdepth = 0; in->nchain = 0;
  if (nsegs == 0) return T_UNK;
  u8 rk; i16 rd = 0;
  int type = resolve_root(segs[0], lv, nlv, &rk, &rd);
  if (rk == RK_DYN) { in->rkind = RK_DYN; return T_UNK; }
  in->rkind = rk; in->rdepth = rd;
  for (size_t i = 1; i < nsegs; i++) {
    int f, nx;
    if (type == T_UNK || !field_step_type(type, segs[i], &f, &nx)) {
      // Unresolvable tail: a loop variable of unknown item type stays
      // dynamic; anything else is a path that does not exist.
      if (rk == RK_LOOPVAR && type == T_UNK) { in->rkind = RK_DYN; in->nchain = 0; return T_UNK; }
      in->rkind = RK_NOTFOUND; in->nchain = 0; return T_UNK;
    }
    if (in->nchain >= 6) { in->rkind = RK_DYN; in->nchain = 0; return T_UNK; }
    in->chain[in->nchain++] = (u8)f;
    type = nx;
  }
  return type;
}

// Constant-fold every path in a compiled template. Walks the code with
// a compile-time loop-variable stack; OP_FOR pushes its variable (typed
// by its source array), OP_NEXT pops it.
static void resolve_slots(Tpl *t) {
  CLoopVar stack[32];
  size_t depth = 0, nlv = 0;
  for (size_t i = 0; i < t->n; i++) {
    Instr *in = &t->code[i];
    switch (in->op) {
      case OP_VAR: case OP_RAW: case OP_JIF:
        // With a loop stack deeper than the table, no loop variable is
        // statically known — resolve without it (roots/DYN still work).
        resolve_path_slots(in->segs, in->nsegs, stack, depth > 32 ? 0 : nlv, in);
        break;
      case OP_FOR: {
        int type = resolve_path_slots(in->segs, in->nsegs, stack, depth > 32 ? 0 : nlv, in);
        int item = T_UNK;
        if (type == T_ARR_REC) item = T_REC;
        else if (type == T_ARR_TAG) item = T_TAG;
        else if (type == T_ARR_TAGREF) item = T_TAGREF;
        else if (type == T_ARR_TOC) item = T_TOC;
        if (depth < 32) { stack[depth].name = in->name; stack[depth].item_type = item; nlv = depth + 1; }
        depth++;
        break;
      }
      case OP_NEXT:
        if (depth > 0) depth--;
        if (depth < 32) nlv = depth;
        break;
      default: break;
    }
  }
}

// ------------------------------------------------------------------ v6 template runtime

typedef struct { const u8 *ptr; size_t len; } CacheEnt;

typedef struct { Slice name; RVal val; } DynBind;

typedef struct {
  Tpl *tpls; size_t ntpls;
  StrMap *card_cache;
  Buf *out;           // per-page buffer (reset per render_page call)
  Arena *arena;
  BuildCtx *bctx;
  // Roots bound for this render call (site/tags/recent always bound).
  RVal site, tags, recent, page, posts, pager, tag;
  u32 root_mask;      // bit per RK_* root kind bound
  u32 shadow_mask;    // root names shadowed by active loop vars
  LoopState *loops; size_t nloops, caploops;
  DynBind *frames; size_t nframes, capframes;
  int depth;
  Buf scratch;
} Renderer;

static u32 root_bit_for_name(Slice name) {
  if (sl_eq_c(name, "site")) return 1u << RK_SITE;
  if (sl_eq_c(name, "tags")) return 1u << RK_TAGS;
  if (sl_eq_c(name, "recent")) return 1u << RK_RECENT;
  if (sl_eq_c(name, "page")) return 1u << RK_PAGE;
  if (sl_eq_c(name, "posts")) return 1u << RK_POSTS;
  if (sl_eq_c(name, "pager")) return 1u << RK_PAGER;
  if (sl_eq_c(name, "tag")) return 1u << RK_TAG;
  return 0;
}

static int root_rval(Renderer *r, int rk, RVal *out) {
  if (!(r->root_mask & (1u << rk))) return 0;
  switch (rk) {
    case RK_SITE: *out = r->site; return 1;
    case RK_TAGS: *out = r->tags; return 1;
    case RK_RECENT: *out = r->recent; return 1;
    case RK_PAGE: *out = r->page; return 1;
    case RK_POSTS: *out = r->posts; return 1;
    case RK_PAGER: *out = r->pager; return 1;
    case RK_TAG: *out = r->tag; return 1;
    default: return 0;
  }
}

static int walk_chain(Renderer *r, RVal v, const u8 *chain, size_t nchain, RVal *out) {
  for (size_t i = 0; i < nchain; i++) {
    RVal nx;
    if (!rval_step(r->bctx, v, chain[i], &nx)) return 0;
    v = nx;
  }
  *out = v;
  return 1;
}

// Dynamic path evaluation (RK_DYN, and shadowed roots): full frame
// search like v5's lookup, then field steps by name.
static int dyn_eval(Renderer *r, Slice *segs, size_t nsegs, RVal *out) {
  if (nsegs == 0) return 0;
  RVal v;
  if (sl_eq_c(segs[0], "loop")) {
    if (r->nloops == 0) return 0;
    memset(&v, 0, sizeof(v));
    v.kind = RV_LOOP; v.ptr = &r->loops[r->nloops - 1];
  } else {
    int found = 0;
    for (size_t i = r->nframes; i-- > 0;) {
      if (sl_eq(r->frames[i].name, segs[0])) { v = r->frames[i].val; found = 1; break; }
    }
    if (!found) {
      u8 rk = 0;
      if (sl_eq_c(segs[0], "site")) rk = RK_SITE;
      else if (sl_eq_c(segs[0], "tags")) rk = RK_TAGS;
      else if (sl_eq_c(segs[0], "recent")) rk = RK_RECENT;
      else if (sl_eq_c(segs[0], "page")) rk = RK_PAGE;
      else if (sl_eq_c(segs[0], "posts")) rk = RK_POSTS;
      else if (sl_eq_c(segs[0], "pager")) rk = RK_PAGER;
      else if (sl_eq_c(segs[0], "tag")) rk = RK_TAG;
      else return 0;
      if (!root_rval(r, rk, &v)) return 0;
    }
  }
  for (size_t i = 1; i < nsegs; i++) {
    int f = field_code_for(segs[i]);
    if (!f) return 0;
    RVal nx;
    if (!rval_step(r->bctx, v, f, &nx)) return 0;
    v = nx;
  }
  *out = v;
  return 1;
}

// Evaluate a resolved slot to an RVal. Returns 0 when the path does
// not exist (renders as nothing / falsy, like v5's lookup returning 0).
static int eval_slot(Renderer *r, Instr *in, size_t loop_base, RVal *out) {
  switch (in->rkind) {
    case RK_NOTFOUND: return 0;
    case RK_DYN: return dyn_eval(r, in->segs, in->nsegs, out);
    case RK_LOOP: {
      if (r->nloops == 0) return 0;
      RVal v; memset(&v, 0, sizeof(v));
      v.kind = RV_LOOP; v.ptr = &r->loops[r->nloops - 1];
      return walk_chain(r, v, in->chain, in->nchain, out);
    }
    case RK_LOOPVAR: {
      size_t idx = loop_base + (size_t)in->rdepth;
      if (idx >= r->nloops) return 0;
      // The dyn frame at the same index holds the current item.
      RVal v = r->frames[idx].val;
      return walk_chain(r, v, in->chain, in->nchain, out);
    }
    default: {
      // A statically resolved root: if a same-named loop variable
      // shadows it (only possible via a caller's loop in a fragment),
      // fall back to the dynamic path.
      if (r->shadow_mask & (1u << in->rkind)) return dyn_eval(r, in->segs, in->nsegs, out);
      RVal v;
      if (!root_rval(r, in->rkind, &v)) return 0;
      return walk_chain(r, v, in->chain, in->nchain, out);
    }
  }
}

// JavaScript's String(v) for template values, written into `out`.
static void write_rval(Buf *out, BuildCtx *ctx, RVal *v) {
  switch (v->kind) {
    case RV_STR: buf_sl(out, v->s); break;
    case RV_NUM: js_num(out, v->num); break;
    case RV_BOOL: buf_cs(out, v->boolean ? "true" : "false"); break;
    case RV_ARR: {
      for (size_t i = 0; i < v->len; i++) {
        if (i > 0) buf_cs(out, ", ");
        RVal item;
        if (arr_item(ctx, v, i, &item)) write_rval(out, ctx, &item);
      }
      break;
    }
    default: break; // objects, null, undef -> ""
  }
}

static int rval_truthy(RVal *v) {
  switch (v->kind) {
    case RV_NUM: return v->num != 0.0;
    case RV_ARR: return v->len > 0;
    case RV_STR: return v->s.n > 0;
    case RV_BOOL: return v->boolean;
    case RV_NULL: case RV_UNDEF: return 0;
    default: return 1; // objects are truthy
  }
}

static void render_tpl(Renderer *r, Tpl *tpl) {
  Buf *out = r->out;
  size_t base = r->nloops; // loop-stack depth at this template's entry
  size_t pc = 0;
  while (pc < tpl->n) {
    Instr *in = &tpl->code[pc];
    switch (in->op) {
      case OP_TEXT: buf_sl(out, in->text); pc++; break;
      case OP_VAR: {
        RVal ref;
        if (eval_slot(r, in, base, &ref)) {
          if (ref.kind == RV_STR) esc(out, ref.s);
          else {
            r->scratch.len = 0;
            write_rval(&r->scratch, r->bctx, &ref);
            esc(out, sl(r->scratch.p, r->scratch.len));
          }
        }
        pc++;
        break;
      }
      case OP_RAW: {
        RVal ref;
        if (eval_slot(r, in, base, &ref)) write_rval(out, r->bctx, &ref);
        pc++;
        break;
      }
      case OP_JIF: {
        RVal ref;
        int t = eval_slot(r, in, base, &ref) ? rval_truthy(&ref) : 0;
        if (t != in->not) pc++;
        else pc = (size_t)in->target;
        break;
      }
      case OP_JMP: pc = (size_t)in->target; break;
      case OP_FOR: {
        RVal ref;
        if (!eval_slot(r, in, base, &ref) || ref.kind != RV_ARR || ref.len == 0) {
          pc = (size_t)in->target;
          break;
        }
        if (r->nloops == r->caploops) {
          size_t nc = r->caploops ? r->caploops * 2 : 8;
          r->loops = (LoopState *)realloc(r->loops, nc * sizeof(LoopState));
          r->caploops = nc;
        }
        if (r->nframes == r->capframes) {
          size_t nc = r->capframes ? r->capframes * 2 : 8;
          r->frames = (DynBind *)realloc(r->frames, nc * sizeof(DynBind));
          r->capframes = nc;
        }
        LoopState *ls = &r->loops[r->nloops++];
        memset(ls, 0, sizeof(*ls));
        ls->arr = ref; ls->idx = 0; ls->len = ref.len; ls->var_name = in->name;
        ls->saved_shadow = r->shadow_mask;
        u32 bit = root_bit_for_name(in->name);
        if (bit) r->shadow_mask |= bit;
        RVal item;
        arr_item(r->bctx, &ref, 0, &item);
        r->frames[r->nframes].name = in->name;
        r->frames[r->nframes].val = item;
        r->nframes++;
        pc++;
        break;
      }
      case OP_NEXT: {
        LoopState *ls = &r->loops[r->nloops - 1];
        ls->idx++;
        if (ls->idx >= ls->len) {
          r->shadow_mask = ls->saved_shadow;
          r->nloops--;
          r->nframes--;
          pc++;
        } else {
          RVal item;
          arr_item(r->bctx, &ls->arr, ls->idx, &item);
          r->frames[r->nframes - 1].val = item;
          pc = (size_t)in->body;
        }
        break;
      }
      case OP_INC: {
        if (r->depth >= INCLUDE_DEPTH || in->tpl < 0) { pc++; break; }
        Tpl *frag = &r->tpls[in->tpl];
        if (in->cacheable) {
          const void *itemptr = NULL;
          if (r->nloops) {
            RVal it = r->frames[r->nframes - 1].val;
            itemptr = it.ptr;
          }
          u8 key[12];
          memcpy(key, &in->tpl, 4);
          uintptr_t up = (uintptr_t)itemptr;
          memcpy(key + 4, &up, 8);
          Slice k = sl(key, 12);
          CacheEnt *hit = (CacheEnt *)strmap_get(r->card_cache, k);
          if (hit) { buf_put(out, hit->ptr, hit->len); pc++; break; }
          size_t start = out->len;
          r->depth++;
          render_tpl(r, frag);
          r->depth--;
          CacheEnt *ne = (CacheEnt *)arena_alloc(r->arena, sizeof(CacheEnt));
          ne->len = out->len - start;
          ne->ptr = arena_slice(r->arena, out->p + start, ne->len).p;
          strmap_put(r->card_cache, k, ne);
          pc++;
          break;
        }
        r->depth++;
        render_tpl(r, frag);
        r->depth--;
        pc++;
        break;
      }
      case OP_END: return;
    }
  }
}

// ------------------------------------------------------------------ v6 build pipeline

static StrMap *strmap_new(Arena *a) {
  StrMap *m = (StrMap *)arena_alloc(a, sizeof(StrMap));
  strmap_init(m, a);
  return m;
}

static size_t strmap_keys(StrMap *m, Slice *out, size_t cap) {
  size_t n = 0;
  for (size_t i = 0; i < m->cap; i++) {
    if (m->slots[i].key) {
      if (n < cap) out[n] = sl(m->slots[i].key, m->slots[i].klen);
      n++;
    }
  }
  return n;
}

static Slice slug_cached(BuildCtx *ctx, Slice text) {
  void *hit = strmap_get(ctx->slug_cache, text);
  if (hit) return *(Slice *)hit;
  Slice s = slugify(ctx->arena, text);
  Slice *sp = (Slice *)arena_alloc(ctx->arena, sizeof(Slice));
  *sp = s;
  strmap_put(ctx->slug_cache, text, sp);
  return s;
}

static Slice long_date_cached(BuildCtx *ctx, Slice date) {
  void *hit = strmap_get(ctx->date_cache, date);
  if (hit) return *(Slice *)hit;
  Slice s = long_date(ctx->arena, date);
  Slice *sp = (Slice *)arena_alloc(ctx->arena, sizeof(Slice));
  *sp = s;
  strmap_put(ctx->date_cache, date, sp);
  return s;
}

static Slice slug_cached_fn(void *vctx, Slice text) {
  return slug_cached((BuildCtx *)vctx, text);
}

typedef struct {
  Slice dir, src_path, base;
  CRec *sources; size_t nsources;
  StrMap *by_source, *static_paths, *broken;
  Arena *arena;
} LinkCtx2;

static Slice build_link(void *vctx, Slice target) {
  LinkCtx2 *ctx = (LinkCtx2 *)vctx;
  if (is_scheme(target) || sl_starts(target, "/") || sl_starts(target, "#")) return target;
  size_t hash = find_byte(target.p, target.n, 0, '#');
  Slice rel, frag;
  if (hash < target.n) { rel = sl(target.p, hash); frag = sl(target.p + hash, target.n - hash); }
  else { rel = target; frag = slc(""); }
  int have_path = 0;
  Slice path = sl(NULL, 0);
  if (rel.n > 0) {
    Slice rp = resolve_path(ctx->arena, ctx->dir, rel);
    if (rp.p) { path = rp; have_path = 1; }
  }
  if (have_path) {
    void *hit = strmap_get(ctx->by_source, path);
    if (hit) {
      size_t si = (size_t)((uintptr_t)hit - 1);
      Buf b = {0};
      buf_sl(&b, ctx->sources[si].url);
      buf_sl(&b, frag);
      Slice r = arena_slice(ctx->arena, b.p, b.len);
      free(b.p);
      return r;
    }
    if (strmap_get(ctx->static_paths, path)) {
      Buf b = {0};
      buf_sl(&b, ctx->base);
      buf_c(&b, '/');
      buf_sl(&b, path);
      buf_sl(&b, frag);
      Slice r = arena_slice(ctx->arena, b.p, b.len);
      free(b.p);
      return r;
    }
  }
  Buf line = {0};
  buf_cs(&line, "broken-link ");
  buf_sl(&line, ctx->src_path);
  buf_cs(&line, " -> ");
  buf_sl(&line, target);
  Slice ls = arena_slice(ctx->arena, line.p, line.len);
  free(line.p);
  strmap_put(ctx->broken, ls, (void *)1);
  return target;
}

// Each tag name once per slug, in its own order (dedup by slug).
static size_t own_tags(BuildCtx *ctx, size_t si, Slice **out_names) {
  CRec *s = &ctx->sources[si];
  StrMap *seen = strmap_new(ctx->arena);
  size_t cap = 4, n = 0;
  Slice *names = (Slice *)arena_alloc(ctx->arena, cap * sizeof(Slice));
  if (s->data.has_tags) {
    for (size_t i = 0; i < s->data.ntags; i++) {
      Slice name = s->data.tags[i];
      Slice slug = slug_cached(ctx, name);
      if (strmap_get(seen, slug)) continue;
      strmap_put(seen, slug, (void *)1);
      if (n == cap) {
        size_t nc = cap * 2;
        Slice *nn = (Slice *)arena_alloc(ctx->arena, nc * sizeof(Slice));
        memcpy(nn, names, n * sizeof(Slice));
        names = nn; cap = nc;
      }
      names[n++] = name;
    }
  }
  *out_names = names;
  return n;
}

static void tagref_push(CRec *s, Arena *a, Slice name, Slice slug, Slice url) {
  if (s->ntag_refs == s->captag_refs) {
    size_t nc = s->captag_refs ? s->captag_refs * 2 : 4;
    TagRef *nr = (TagRef *)arena_alloc(a, nc * sizeof(TagRef));
    if (s->tag_refs) memcpy(nr, s->tag_refs, s->ntag_refs * sizeof(TagRef));
    s->tag_refs = nr; s->captag_refs = nc;
  }
  s->tag_refs[s->ntag_refs].name = name;
  s->tag_refs[s->ntag_refs].slug = slug;
  s->tag_refs[s->ntag_refs].url = url;
  s->ntag_refs++;
}

typedef struct {
  RVal page, posts, pager, tag;
  u32 mask; // ROOT_BIT per bound root (page/posts/pager/tag)
} Roots;

// Render `layout` with the given roots (+ site/tags/recent) into
// ctx->page_buf. Returns 0 when the template is missing or failed.
static int render_page_c(BuildCtx *ctx, Slice layout, Roots *roots) {
  int ti = -1;
  for (size_t i = 0; i < ctx->ntpls; i++) {
    if (sl_eq(ctx->tpl_names[i], layout)) { ti = (int)i; break; }
  }
  if (ti < 0 || !ctx->tpl_ok[ti]) return 0;
  Renderer r;
  memset(&r, 0, sizeof(r));
  r.tpls = ctx->tpls; r.ntpls = ctx->ntpls;
  r.card_cache = ctx->card_cache; r.out = &ctx->page_buf; r.arena = ctx->arena;
  r.bctx = ctx; r.depth = 0;
  memset(&r.site, 0, sizeof(r.site));
  r.site.kind = RV_SITE; r.site.ptr = ctx;
  r.tags = ctx->tags_rv;
  r.recent = ctx->recent_rv;
  r.page = roots->page; r.posts = roots->posts; r.pager = roots->pager; r.tag = roots->tag;
  r.root_mask = (1u << RK_SITE) | (1u << RK_TAGS) | (1u << RK_RECENT) | roots->mask;
  ctx->page_buf.len = 0;
  render_tpl(&r, &ctx->tpls[ti]);
  free(r.loops); free(r.frames); free(r.scratch.p);
  return 1;
}

static void emit_c(BuildCtx *ctx, Slice k, Slice v) {
  if (!ctx->first) buf_c(ctx->out, ',');
  ctx->first = 0;
  json_write_str(ctx->out, k);
  buf_c(ctx->out, ':');
  json_write_str(ctx->out, v);
}

static void report_push(BuildCtx *ctx, Slice line) {
  if (ctx->nreport == ctx->capreport) {
    size_t nc = ctx->capreport ? ctx->capreport * 2 : 16;
    Slice *nr = (Slice *)arena_alloc(ctx->arena, nc * sizeof(Slice));
    if (ctx->report) memcpy(nr, ctx->report, ctx->nreport * sizeof(Slice));
    ctx->report = nr; ctx->capreport = nc;
  }
  ctx->report[ctx->nreport++] = line;
}

static int template_exists(BuildCtx *ctx, Slice layout) {
  for (size_t i = 0; i < ctx->ntpls; i++) {
    if (sl_eq(ctx->tpl_names[i], layout)) return ctx->tpl_ok[i];
  }
  return 0;
}

static int cmp_tpl_names(void *vctx, size_t a, size_t b) {
  TplSrc *t = (TplSrc *)vctx;
  return cmp_u16(t[a].name, t[b].name);
}

static int cmp_files(void *vctx, size_t a, size_t b) {
  InFile *f = (InFile *)vctx;
  return cmp_u16(f[a].path, f[b].path);
}

static int cmp_posts(void *vctx, size_t a, size_t b) {
  BuildCtx *ctx = (BuildCtx *)vctx;
  CRec *sa = &ctx->sources[a], *sb = &ctx->sources[b];
  int c = cmp_u16(sb->data.date, sa->data.date);
  if (c) return c;
  return cmp_u16(sa->slug, sb->slug);
}

static int cmp_tags(void *vctx, size_t a, size_t b) {
  Tag *tags = (Tag *)vctx;
  Tag *ta = &tags[a], *tb = &tags[b];
  if (ta->nposts != tb->nposts) return ta->nposts > tb->nposts ? -1 : 1;
  return cmp_u16(ta->slug, tb->slug);
}

static void source_push(BuildCtx *ctx, CRec s) {
  if (ctx->nsources == ctx->capsources) {
    size_t nc = ctx->capsources ? ctx->capsources * 2 : 64;
    ctx->sources = (CRec *)realloc(ctx->sources, nc * sizeof(CRec));
    ctx->capsources = nc;
  }
  ctx->sources[ctx->nsources++] = s;
}

static void static_push(BuildCtx *ctx, Slice path, Slice content) {
  if (ctx->nstatics == ctx->capstatics) {
    size_t nc = ctx->capstatics ? ctx->capstatics * 2 : 16;
    ctx->statics = realloc(ctx->statics, nc * sizeof(*ctx->statics));
    ctx->capstatics = nc;
  }
  ctx->statics[ctx->nstatics].path = path;
  ctx->statics[ctx->nstatics].content = content;
  ctx->nstatics++;
}

// ------------------------------------------------------------------ build_site
static int build_site(CaseIn *input, Buf *out, Arena *arena) {
  BuildCtx ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.arena = arena; ctx.out = out; ctx.first = 1;
  ctx.slug_cache = strmap_new(arena);
  ctx.date_cache = strmap_new(arena);
  ctx.card_cache = strmap_new(arena);
  ctx.by_source = strmap_new(arena);
  ctx.static_paths = strmap_new(arena);
  ctx.broken = strmap_new(arena);

  SiteIn *site = &input->site;
  if (!site->has_title || !site->has_base_url || !site->has_per_page || !site->has_recent) return 0;
  Slice site_title = site->title;
  Slice base_url = site->base_url;
  double per_page = site->per_page;
  double recent_n = site->recent;
  ctx.site = site;

  Slice base = base_url;
  while (base.n > 0 && base.p[base.n - 1] == '/') base.n--;
  ctx.base = base;

  // Templates, in code-unit order of name; one that does not parse is
  // reported and left out.
  size_t nt = input->ntpls;
  size_t *torder = (size_t *)malloc((nt ? nt : 1) * sizeof(size_t));
  for (size_t i = 0; i < nt; i++) torder[i] = i;
  sort_indices(torder, nt, input->tpls, cmp_tpl_names);
  ctx.ntpls = nt;
  ctx.tpl_names = (Slice *)arena_alloc(arena, (nt ? nt : 1) * sizeof(Slice));
  ctx.tpls = (Tpl *)arena_alloc(arena, (nt ? nt : 1) * sizeof(Tpl));
  ctx.tpl_ok = (int *)arena_alloc(arena, (nt ? nt : 1) * sizeof(int));
  for (size_t i = 0; i < nt; i++) {
    size_t mi = torder[i];
    Slice name = input->tpls[mi].name;
    ctx.tpl_names[i] = name;
    Slice src = input->tpls[mi].src;
    TokVec toks; memset(&toks, 0, sizeof(toks));
    Slice err;
    if (!tokenize(name, src, arena, &toks, &err)) {
      report_push(&ctx, err);
      ctx.tpl_ok[i] = 0;
      continue;
    }
    if (!compile_template(name, &toks, arena, &ctx.tpls[i], &err)) {
      report_push(&ctx, err);
      ctx.tpl_ok[i] = 0;
      continue;
    }
    resolve_slots(&ctx.tpls[i]);
    ctx.tpl_ok[i] = 1;
  }
  free(torder);
  resolve_includes(ctx.tpls, nt);
  for (size_t i = 0; i < nt; i++) if (ctx.tpl_ok[i]) analyze_fragment(&ctx.tpls[i], arena);
  analyze_cacheable(ctx.tpls, nt, arena);

  // Sources, in code-unit order of path.
  size_t nf = input->nfiles;
  size_t *forder = (size_t *)malloc((nf ? nf : 1) * sizeof(size_t));
  for (size_t i = 0; i < nf; i++) forder[i] = i;
  sort_indices(forder, nf, input->files, cmp_files);
  for (size_t i = 0; i < nf; i++) {
    InFile *f = &input->files[forder[i]];
    Slice path = f->path, content = f->content;
    if (sl_ends(path, ".md")) {
      CRec s;
      memset(&s, 0, sizeof(s));
      s.path = path;
      s.body = crlf_to_lf(arena, content);
      s.newer = -1; s.older = -1;
      source_push(&ctx, s);
    } else {
      static_push(&ctx, path, content);
    }
  }
  free(forder);

  // Front matter + refusals, in path order.
  size_t w = 0;
  for (size_t r = 0; r < ctx.nsources; r++) {
    CRec *s = &ctx.sources[r];
    Fm fm;
    Slice err;
    if (!parse_front_matter(s->body, arena, &fm, &err)) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": ");
      buf_sl(&line, err);
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    s->data = fm.data;
    s->body = fm.body;
    s->post = sl_starts(s->path, "posts/") ? 1 : 0;
    const char *refused = NULL;
    if (sl_starts(s->path, "tags/")) refused = "tags/ is the generator's own";
    else if (sl_starts(s->path, "page/")) refused = "page/ is the generator's own";
    if (refused) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": ");
      buf_cs(&line, refused);
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    if (s->post && !s->data.has_date) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": a post needs a date");
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    Slice layout;
    if (s->data.has_layout) layout = s->data.layout;
    else layout = s->post ? slc("post") : slc("page");
    if (!template_exists(&ctx, layout)) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": layout \"");
      buf_sl(&line, layout);
      buf_cs(&line, "\" is no template that parses");
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    s->layout = layout;
    if (s->data.draft) {
      Buf line = {0};
      buf_cs(&line, "draft ");
      buf_sl(&line, s->path);
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    ctx.sources[w++] = *s;
  }
  ctx.nsources = w;

  // Permalinks (only the slug part is deduplicated).
  StrMap *taken = strmap_new(arena);
  for (size_t idx = 0; idx < ctx.nsources; idx++) {
    CRec *s = &ctx.sources[idx];
    Slice dir = s->post ? slc("posts") : dir_of(s->path);
    StrMap *tt = (StrMap *)strmap_get(taken, dir);
    if (!tt) {
      tt = strmap_new(arena);
      if (dir.n == 0) {
        strmap_put(tt, slc("posts"), (void *)1);
        strmap_put(tt, slc("tags"), (void *)1);
        strmap_put(tt, slc("page"), (void *)1);
      }
      strmap_put(taken, dir, tt);
    }
    Slice base_slug = s->data.has_slug ? s->data.slug : slug_cached(&ctx, s->data.title);
    Slice sg = unique_slug(arena, tt, base_slug);
    s->slug = sg;
    Slice out_dir;
    if (dir.n == 0) out_dir = sg;
    else {
      Buf b = {0};
      buf_sl(&b, dir); buf_c(&b, '/'); buf_sl(&b, sg);
      out_dir = arena_slice(arena, b.p, b.len);
      free(b.p);
    }
    s->out_dir = out_dir;
    Buf u = {0};
    buf_sl(&u, base); buf_c(&u, '/'); buf_sl(&u, out_dir); buf_c(&u, '/');
    s->url = arena_slice(arena, u.p, u.len);
    free(u.p);
    strmap_put(ctx.by_source, s->path, (void *)(uintptr_t)(idx + 1));
  }

  // Markdown for every source; then precompute the flat card fields
  // (reading time, long date) once, here, instead of per render.
  for (size_t i = 0; i < ctx.nstatics; i++)
    strmap_put(ctx.static_paths, ctx.statics[i].path, (void *)1);
  for (size_t idx = 0; idx < ctx.nsources; idx++) {
    CRec *s = &ctx.sources[idx];
    Slice dir = dir_of(s->path);
    LinkCtx2 lctx;
    memset(&lctx, 0, sizeof(lctx));
    lctx.dir = dir; lctx.src_path = s->path; lctx.base = base;
    lctx.sources = ctx.sources; lctx.nsources = ctx.nsources;
    lctx.by_source = ctx.by_source; lctx.static_paths = ctx.static_paths;
    lctx.broken = ctx.broken; lctx.arena = arena;
    StrMap *ids = strmap_new(arena);
    MdOut md = md_render(s->body, build_link, &lctx, slug_cached_fn, &ctx, arena, ids);
    s->html = md.html;
    s->toc = md.toc;
    s->ntoc = md.ntoc;
    s->excerpt = md.excerpt;
    size_t nw = words_of(s->body);
    u32 reading = (u32)((nw + 199) / 200);
    if (reading < 1) reading = 1;
    s->reading_minutes = (double)reading;
    s->date_long = s->data.has_date ? long_date_cached(&ctx, s->data.date) : slc("");
  }

  // Collections.
  size_t *posts = (size_t *)malloc((ctx.nsources ? ctx.nsources : 1) * sizeof(size_t));
  size_t nposts = 0;
  for (size_t i = 0; i < ctx.nsources; i++) if (ctx.sources[i].post) posts[nposts++] = i;
  sort_indices(posts, nposts, &ctx, cmp_posts);
  for (size_t k = 0; k < nposts; k++) {
    size_t pi = posts[k];
    ctx.sources[pi].newer = k == 0 ? -1 : (int)posts[k - 1];
    ctx.sources[pi].older = k + 1 == nposts ? -1 : (int)posts[k + 1];
  }

  // Tags. Each post's tags (deduped by slug, in the post's own order)
  // join the tag's post list; tags are then sorted by post count, then
  // slug. Non-post sources get tag references without a post list.
  StrMap *by_slug = strmap_new(arena);
  size_t captags = 16, ntags = 0;
  Tag *tags_arr = (Tag *)malloc(captags * sizeof(Tag));
  size_t *tag_order = (size_t *)malloc((ctx.nsources ? ctx.nsources : 1) * sizeof(size_t));
  size_t ntag_order = 0;
  for (size_t k = 0; k < nposts; k++) {
    size_t pi = posts[k];
    Slice *names;
    size_t nnames = own_tags(&ctx, pi, &names);
    for (size_t q = 0; q < nnames; q++) {
      Slice name = names[q];
      Slice slug = slug_cached(&ctx, name);
      void *hit = strmap_get(by_slug, slug);
      size_t ti;
      if (!hit) {
        if (ntags == captags) { captags *= 2; tags_arr = (Tag *)realloc(tags_arr, captags * sizeof(Tag)); }
        ti = ntags++;
        tags_arr[ti].name = name;
        tags_arr[ti].slug = slug;
        Buf u = {0};
        buf_sl(&u, base); buf_cs(&u, "/tags/"); buf_sl(&u, slug); buf_c(&u, '/');
        tags_arr[ti].url = arena_slice(arena, u.p, u.len);
        free(u.p);
        tags_arr[ti].posts = NULL; tags_arr[ti].nposts = 0; tags_arr[ti].capposts = 0;
        strmap_put(by_slug, slug, (void *)(uintptr_t)(ti + 1));
        tag_order[ntag_order++] = ti;
      } else ti = (size_t)((uintptr_t)hit - 1);
      Tag *t = &tags_arr[ti];
      if (t->nposts == t->capposts) {
        size_t nc = t->capposts ? t->capposts * 2 : 4;
        t->posts = (size_t *)realloc(t->posts, nc * sizeof(size_t));
        t->capposts = nc;
      }
      t->posts[t->nposts++] = pi;
      tagref_push(&ctx.sources[pi], arena, t->name, t->slug, t->url);
    }
  }
  for (size_t i = 0; i < ctx.nsources; i++) {
    if (ctx.sources[i].post) continue;
    Slice *names;
    size_t nnames = own_tags(&ctx, i, &names);
    for (size_t q = 0; q < nnames; q++) {
      Slice name = names[q];
      Slice slug = slug_cached(&ctx, name);
      void *hit = strmap_get(by_slug, slug);
      if (hit) {
        Tag *t = &tags_arr[(size_t)((uintptr_t)hit - 1)];
        tagref_push(&ctx.sources[i], arena, t->name, t->slug, t->url);
      } else {
        tagref_push(&ctx.sources[i], arena, name, slug, slc(""));
      }
    }
  }
  sort_indices(tag_order, ntag_order, tags_arr, cmp_tags);

  // Per-case globals (flat arrays, no Val construction).
  {
    Tag **sorted = (Tag **)arena_alloc(arena, (ntag_order ? ntag_order : 1) * sizeof(Tag *));
    for (size_t k = 0; k < ntag_order; k++) sorted[k] = &tags_arr[tag_order[k]];
    memset(&ctx.tags_rv, 0, sizeof(ctx.tags_rv));
    ctx.tags_rv.kind = RV_ARR; ctx.tags_rv.ptr = sorted; ctx.tags_rv.len = ntag_order; ctx.tags_rv.elem = AE_TAG_PTR;
  }
  {
    size_t r0, r1;
    js_slice_bounds(nposts, 0.0, recent_n, &r0, &r1);
    memset(&ctx.recent_rv, 0, sizeof(ctx.recent_rv));
    ctx.recent_rv.kind = RV_ARR; ctx.recent_rv.ptr = posts + r0; ctx.recent_rv.len = r1 - r0; ctx.recent_rv.elem = AE_REC_IDX;
  }

  StrMap *out_paths = strmap_new(arena);
  // Layouts.
  for (size_t idx = 0; idx < ctx.nsources; idx++) {
    CRec *s = &ctx.sources[idx];
    Roots roots;
    memset(&roots, 0, sizeof(roots));
    roots.page.kind = RV_REC; roots.page.ptr = s;
    roots.mask = (1u << RK_PAGE);
    if (!render_page_c(&ctx, s->layout, &roots)) return 0;
    Buf pb = {0};
    buf_sl(&pb, s->out_dir);
    buf_cs(&pb, "/index.html");
    Slice p = arena_slice(arena, pb.p, pb.len);
    free(pb.p);
    strmap_put(out_paths, p, (void *)1);
    emit_c(&ctx, p, sl(ctx.page_buf.p, ctx.page_buf.len));
  }
  // Index pages.
  double pages_n = per_page == 0.0 ? INFINITY : fmax(1.0, ceil((double)nposts / per_page));
  {
    u64 k = 1;
    while ((double)k <= pages_n) {
      size_t a, b;
      js_slice_bounds(nposts, (double)(k - 1) * per_page, (double)k * per_page, &a, &b);
      Pager pager;
      pager.number = (double)k;
      pager.total = pages_n;
      pager.url = index_url(arena, base, k);
      pager.newer_url = k == 1 ? slc("") : index_url(arena, base, k - 1);
      pager.older_url = (double)k == pages_n ? slc("") : index_url(arena, base, k + 1);
      Roots roots;
      memset(&roots, 0, sizeof(roots));
      roots.posts.kind = RV_ARR; roots.posts.ptr = posts + a; roots.posts.len = b - a; roots.posts.elem = AE_REC_IDX;
      roots.pager.kind = RV_PAGER; roots.pager.ptr = &pager;
      roots.mask = (1u << RK_POSTS) | (1u << RK_PAGER);
      if (render_page_c(&ctx, slc("index"), &roots)) {
        Slice p;
        if (k == 1) p = slc("index.html");
        else {
          Buf pb = {0};
          buf_cs(&pb, "page/");
          char kb[24];
          int m = snprintf(kb, sizeof(kb), "%llu", (unsigned long long)k);
          buf_put(&pb, kb, (size_t)m);
          buf_cs(&pb, "/index.html");
          p = arena_slice(arena, pb.p, pb.len);
          free(pb.p);
        }
        strmap_put(out_paths, p, (void *)1);
        emit_c(&ctx, p, sl(ctx.page_buf.p, ctx.page_buf.len));
      }
      k++;
      if (!isfinite(pages_n)) break;
    }
  }
  // Tag pages.
  for (size_t ti = 0; ti < ntag_order; ti++) {
    Tag *t = &tags_arr[tag_order[ti]];
    Roots roots;
    memset(&roots, 0, sizeof(roots));
    roots.tag.kind = RV_TAG; roots.tag.ptr = t;
    roots.mask = (1u << RK_TAG);
    if (render_page_c(&ctx, slc("tag"), &roots)) {
      Buf pb = {0};
      buf_cs(&pb, "tags/");
      buf_sl(&pb, t->slug);
      buf_cs(&pb, "/index.html");
      Slice p = arena_slice(arena, pb.p, pb.len);
      free(pb.p);
      strmap_put(out_paths, p, (void *)1);
      emit_c(&ctx, p, sl(ctx.page_buf.p, ctx.page_buf.len));
    }
  }
  // Tags index.
  {
    Roots roots;
    memset(&roots, 0, sizeof(roots));
    if (render_page_c(&ctx, slc("tags"), &roots)) {
      Slice p = slc("tags/index.html");
      strmap_put(out_paths, p, (void *)1);
      emit_c(&ctx, p, sl(ctx.page_buf.p, ctx.page_buf.len));
    }
  }
  // Statics (shadow check): a static whose path a built page already
  // has is reported, not emitted.
  for (size_t i = 0; i < ctx.nstatics; i++) {
    Slice p = ctx.statics[i].path;
    if (strmap_get(out_paths, p)) {
      Buf line = {0};
      buf_cs(&line, "shadowed ");
      buf_sl(&line, p);
      buf_cs(&line, ": a built page has its path");
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
    } else {
      strmap_put(out_paths, p, (void *)1);
      emit_c(&ctx, p, ctx.statics[i].content);
    }
  }
  // Report: all report lines plus all broken-link lines, sorted
  // together, then the "built N pages..." summary line.
  {
    size_t nbroken = ctx.broken->len;
    Slice *lines = (Slice *)arena_alloc(arena, (ctx.nreport + nbroken + 1) * sizeof(Slice));
    size_t nlines = 0;
    for (size_t i = 0; i < ctx.nreport; i++) lines[nlines++] = ctx.report[i];
    nlines += strmap_keys(ctx.broken, lines + nlines, nbroken);
    sort_slices(lines, nlines);
    Buf rt = {0};
    for (size_t i = 0; i < nlines; i++) {
      buf_sl(&rt, lines[i]);
      buf_c(&rt, '\n');
    }
    size_t page_count = 0;
    for (size_t i = 0; i < ctx.nsources; i++) if (!ctx.sources[i].post) page_count++;
    char last[256];
    int m = snprintf(last, sizeof(last),
                     "built %zu pages, %zu posts, %zu tags and %zu static files",
                     page_count, nposts, ntag_order, ctx.nstatics);
    buf_put(&rt, last, (size_t)m);
    buf_c(&rt, '\n');
    emit_c(&ctx, slc("build-report.txt"), sl(rt.p, rt.len));
    free(rt.p);
  }
  (void)site_title;
  free(ctx.sources); free(ctx.statics); free(posts); free(tags_arr); free(tag_order);
  free(ctx.page_buf.p);
  return 1;
}

// ------------------------------------------------------------------ run_cases / run_one
//
// answers.json is a JSON ARRAY of answer objects, one per case, in the
// cases' order (the baseline driver: answers.push(run(c.input))).
// run_one answers ONE case input object ({site, templates, files}) with
// one answer object — the solve.js run() contract.

// The whole cases.json array at once -> answers.json array bytes.
static int run_cases(const char *json, size_t jlen, Buf *out) {
  Arena arena_storage;
  Arena *arena = &arena_storage;
  arena_init(arena);
  CaseVec cases; memset(&cases, 0, sizeof(cases));
  if (!fp_parse_cases(json, jlen, arena, &cases)) { arena_free(arena); return 0; }
  buf_c(out, '[');
  for (size_t ci = 0; ci < cases.n; ci++) {
    if (ci > 0) buf_c(out, ',');
    buf_c(out, '{');
    if (!build_site(&cases.items[ci], out, arena)) { arena_free(arena); return 0; }
    buf_c(out, '}');
  }
  buf_c(out, ']');
  arena_free(arena);
  return 1;
}

// One case input object at once -> one answer object.
static int run_one(const char *json, size_t jlen, Buf *out) {
  Arena arena_storage;
  Arena *arena = &arena_storage;
  arena_init(arena);
  JParser p = { (const u8 *)json, jlen, 0, arena };
  fp_ws(&p);
  if (p.pos >= p.len || p.b[p.pos] != '{') { arena_free(arena); return 0; }
  CaseIn ci;
  memset(&ci, 0, sizeof(ci));
  int ok = fp_parse_case_input(&p, &ci);
  if (!ok) { arena_free(arena); return 0; }
  buf_c(out, '{');
  ok = build_site(&ci, out, arena);
  buf_c(out, '}');
  arena_free(arena);
  return ok;
}

// ------------------------------------------------------------------ N-API glue

// The output buffer is malloc'd by the engine; hand it to JS as an
// external Buffer that frees it on GC instead of copying 7+ MB.
static void free_buf_finalize(napi_env env, void *data, void *hint) {
  (void)env; (void)hint;
  free(data);
}

static napi_value js_build_cases(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
    napi_throw_error(env, NULL, "buildCases expects one Buffer argument");
    return NULL;
  }
  void *data = NULL;
  size_t len = 0;
  if (napi_get_buffer_info(env, argv[0], &data, &len) != napi_ok) {
    napi_throw_error(env, NULL, "buildCases expects a Buffer");
    return NULL;
  }
  Buf out = {0};
  if (!run_cases((const char *)data, len, &out)) {
    free(out.p);
    abort(); // structural error on a valid case is impossible; match the reference
  }
  napi_value result = NULL;
  if (napi_create_external_buffer(env, out.len, out.p, free_buf_finalize, NULL, &result) != napi_ok) {
    free(out.p);
    napi_throw_error(env, NULL, "cannot create result buffer");
    return NULL;
  }
  return result;
}

static napi_value js_build_one(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
    napi_throw_error(env, NULL, "buildOne expects one Buffer argument");
    return NULL;
  }
  void *data = NULL;
  size_t len = 0;
  if (napi_get_buffer_info(env, argv[0], &data, &len) != napi_ok) {
    napi_throw_error(env, NULL, "buildOne expects a Buffer");
    return NULL;
  }
  Buf out = {0};
  if (!run_one((const char *)data, len, &out)) {
    free(out.p);
    abort();
  }
  napi_value result = NULL;
  if (napi_create_external_buffer(env, out.len, out.p, free_buf_finalize, NULL, &result) != napi_ok) {
    free(out.p);
    napi_throw_error(env, NULL, "cannot create result buffer");
    return NULL;
  }
  return result;
}

static napi_value senja_init(napi_env env, napi_value exports) {
  have_avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
  napi_value fn1, fn2, ran;
  napi_create_function(env, "buildCases", NAPI_AUTO_LENGTH, js_build_cases, NULL, &fn1);
  napi_set_named_property(env, exports, "buildCases", fn1);
  napi_create_function(env, "buildOne", NAPI_AUTO_LENGTH, js_build_one, NULL, &fn2);
  napi_set_named_property(env, exports, "buildOne", fn2);
  napi_create_int32(env, 0, &ran);
  napi_set_named_property(env, exports, "cliRan", ran);

  // CLI fast path: when the addon's loader is the entry's main script
  // (node /bot/main.ts), run the whole pipeline here at load time and
  // exit via the exit_group syscall, skipping Node teardown. The write
  // must complete first; on any failure return normally so the JS
  // driver can fall back to the JS/WASM path.
  // Read the process argv from /proc/self/cmdline (Linux).
  {
    int fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd >= 0) {
      char cb[1024];
      ssize_t n = read(fd, cb, sizeof(cb) - 1);
      close(fd);
      if (n > 0) {
        cb[n] = 0;
        // Count NUL-separated fields.
        int count = 0;
        for (ssize_t i = 0; i < n; i++) if (cb[i] == 0) count++;
        if (count >= 2) {
          // argv[1] is the second field.
          const char *p = cb + strlen(cb) + 1;
          size_t len = strlen(p);
          if (len >= 7 && strcmp(p + len - 7, "main.ts") == 0) {
            const char *in_dir = getenv("MAKEFUN_INPUT_DIR");
            const char *out_dir = getenv("MAKEFUN_OUTPUT_DIR");
            if (!in_dir) in_dir = "/input";
            if (!out_dir) out_dir = "/output";
            char in_path[4096], out_path[4096];
            snprintf(in_path, sizeof(in_path), "%s/cases.json", in_dir);
            snprintf(out_path, sizeof(out_path), "%s/answers.json", out_dir);
            int ifd = open(in_path, O_RDONLY);
            if (ifd >= 0) {
              struct stat st;
              if (fstat(ifd, &st) == 0 && st.st_size > 0) {
                char *json = (char *)malloc((size_t)st.st_size + 1);
                ssize_t got = 0;
                while (got < st.st_size) {
                  ssize_t r = read(ifd, json + got, (size_t)st.st_size - (size_t)got);
                  if (r <= 0) break;
                  got += r;
                }
                close(ifd);
                if (got == st.st_size) {
                  Buf out = {0};
                  if (run_cases(json, (size_t)got, &out)) {
                    int ofd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                    if (ofd >= 0) {
                      size_t off = 0;
                      int wok = 1;
                      while (off < out.len) {
                        ssize_t w = write(ofd, out.p + off, out.len - off);
                        if (w <= 0) { wok = 0; break; }
                        off += (size_t)w;
                      }
                      close(ofd);
                      free(out.p);
                      free(json);
                      if (wok) {
                        napi_create_int32(env, 1, &ran);
                        napi_set_named_property(env, exports, "cliRan", ran);
                        syscall(SYS_exit_group, 0);
                      }
                    }
                  }
                  free(out.p);
                }
                free(json);
              } else close(ifd);
            }
          }
        }
      }
    }
  }
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, senja_init)
