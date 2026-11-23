// Senja Site v6 — thin ESM driver.
//
// The native addon (c/x.node) runs the whole pipeline at load time when
// this file is the entry point (node main.ts): it reads cases.json,
// builds every case, writes answers.json, and exits directly via the
// exit_group syscall, skipping Node teardown. The JS below only runs when
// that fast path did not trigger (different argv, or a load-time failure
// that left cliRan at 0) — then it does the same work through the addon's
// buildCases, or through the WASM fallback in solve.cjs when the addon
// cannot load at all. The addon is imported first, before any file is
// read, so the fast path costs no JS-side I/O.
import { readFileSync, writeFileSync } from 'node:fs';

let m = null;
try {
  m = (await import('./c/x.node')).default;
} catch {}

const d = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const o = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

if (m && !m.cliRan) {
  writeFileSync(o + '/answers.json', m.buildCases(readFileSync(d + '/cases.json')));
} else if (!m) {
  const { runAll } = await import('./solve.cjs');
  writeFileSync(o + '/answers.json', runAll(readFileSync(d + '/cases.json')));
}
