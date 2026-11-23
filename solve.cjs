// Senja Site v6 — native addon loader with a WASM fallback.
//
// The fast path is the native C addon (c/x.node, N-API): it runs the
// whole generator in C and returns the answers bytes. v5's main.ts loads
// the addon directly and only reaches this module when that fails; this
// module is the fallback path and the run() compat entry point. If the
// addon cannot load (dlopen error, missing symbol, wrong architecture),
// this module falls back to the v3 WASM engine (solver.wasm, already
// verified) so the entry still scores instead of failing outright.
// Note: the addon returns an external Buffer (zero-copy); the WASM path
// still copies its output (callWasm slices).
//
// run(input) answers one case (a parsed JS object) and returns its output
// tree, exactly like the baseline driver expects. runAll(bytes) answers
// the whole cases.json array text at once and returns the answers.json
// bytes — the fast path the v4 driver uses, so no JSON work happens in JS.

const { readFileSync } = require('node:fs');
const { join } = require('node:path');

let native = null;
try {
  const m = require('./c/x.node');
  // Only trust the addon if both entry points are present; a half-loaded
  // or mismatched binary falls back to WASM instead of crashing the run.
  if (m && typeof m.buildCases === 'function' && typeof m.buildOne === 'function') native = m;
} catch (e) {
  native = null;
}

let wasm = null;
function loadWasm() {
  if (wasm) return wasm;
  const bytes = readFileSync(join(__dirname, 'solver.wasm'));
  const mod = new WebAssembly.Module(bytes);
  const inst = new WebAssembly.Instance(mod, {});
  wasm = {
    mem: inst.exports.memory,
    inputPtr: inst.exports.input_ptr,
    buildCases: inst.exports.build_cases,
    buildOne: inst.exports.build_one,
  };
  return wasm;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

/** Call a WASM entry point with `data` bytes; return a COPY of its output. */
function callWasm(fn, data) {
  const w = loadWasm();
  const ptr = w.inputPtr(data.length);
  new Uint8Array(w.mem.buffer, ptr, data.length).set(data);
  const r = fn(ptr, data.length);
  const outPtr = Number(r & 0xFFFFFFFFn);
  const outLen = Number(r >> 32n);
  return new Uint8Array(w.mem.buffer, outPtr, outLen).slice();
}

/** One case, as a parsed object — the baseline driver's contract. */
function run(input) {
  const data = enc.encode(JSON.stringify(input));
  if (native) return JSON.parse(dec.decode(native.buildOne(Buffer.from(data))));
  return JSON.parse(dec.decode(callWasm(wasm.buildOne, data)));
}

/** The whole cases.json array, as bytes — returns the answers.json bytes. */
function runAll(input) {
  const data = Buffer.isBuffer(input) ? input : Buffer.from(input);
  if (native) return native.buildCases(data);
  const w = loadWasm();
  return callWasm(w.buildCases, data);
}

module.exports = { runAll, run };
