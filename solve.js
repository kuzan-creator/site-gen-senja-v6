// Senja Site v6 — ES module contract: run(input) answers one case.
//
// The native addon (c/x.node) does the work; solve.cjs loads it, with the
// v3 WASM engine as fallback when the addon cannot load. This file stays
// a plain ES module so the baseline driver's `await import('./solve.js')`
// keeps working exactly as the competition.yaml contract requires.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { run } = require('./solve.cjs');

export { run };
