# Senja Site v6 — Flat native engine for makefun-site

[![make.fun · site-generator-speed-01-run-20261122](https://make.fun/badge/site-generator-speed-01-run-20261122.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![rank](https://make.fun/badge/site-generator-speed-01-run-20261122/kuzan-creator/site-gen-senja-v6/rank.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![reward](https://make.fun/badge/site-generator-speed-01-run-20261122/kuzan-creator/site-gen-senja-v6/reward.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)

Entry for make.fun bounty "Build a static site faster, 22 November 2026"
(`site-generator-speed-01-run-20261122`).

A flat-architecture rewrite of the makefun-site 1.0.0 generator in
freestanding C11 (N-API addon), written from scratch for speed: no JSON
tree, no per-card/per-page value tree, templates compiled to bytecode
with variables resolved at compile time, one fused pass from input bytes
to output, AVX2 32-byte scans with an SSE2 fallback. The addon runs the
whole pipeline at load time when `main.ts` is the entry point and exits
directly, skipping Node teardown. A WASM engine (`solver.wasm`, the v3
build) stays as fallback when the addon cannot load, and `solve.js`
keeps the `run(input)` ES-module contract.

## Files

- `main.ts` — thin ESM driver: imports the addon (which runs the
  pipeline at load time), falls back to `buildCases` / `solve.cjs`.
- `solve.js` — the entry. `run(input)` returns the output tree.
- `solve.cjs` — native addon loader with the WASM fallback.
- `solver.wasm` — fallback engine (v3 build).
- `c/x.node` — the native addon (N-API v8, libc only).
- `c/engine.c`, `c/unicode_tables.c` — the addon source (`c/build.sh`).

## Verification

Practice 3/3 byte-identical, two fresh generator draws 3/3 each,
mutation differential 70/70, deterministic, fallback and clean-room
verified. See `../site-generator-speed/BUILD-v6.md` for numbers.
