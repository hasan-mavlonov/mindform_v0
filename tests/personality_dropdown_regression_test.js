"use strict";
/* Regression test for the BFI-2-S "dropdown opens then immediately closes" bug.
 *
 * Run with: node tests/personality_dropdown_regression_test.js
 *
 * Root cause (see bench/personality/live.py's poll() docstring): poll() called
 * render() unconditionally every 600ms, and render() replaces #root's whole
 * innerHTML every time -- which destroys a native <details> element's "open"
 * DOM state even when nothing about the underlying data changed. The fix
 * gates render() on a real change to the polled state.
 *
 * This test does not drive a real browser DOM (no jsdom dependency in this
 * project). It verifies the actual mechanism of the bug and its fix directly:
 * whether the #root element's innerHTML is REASSIGNED when /api/state returns
 * byte-identical JSON across consecutive polls. That reassignment is exactly
 * what tears down an open <details> node; counting it is a faithful,
 * sufficient regression test for this fix without needing a full DOM.
 */

const fs = require("fs");
const path = require("path");
const vm = require("vm");

// Extract the live JS straight from the real source at run time -- this test
// must exercise the actual shipped code, not a hand-copied snapshot of it
// that could quietly drift out of sync with a later edit.
const livePyPath = path.join(__dirname, "..", "bench", "personality", "live.py");
const livePySource = fs.readFileSync(livePyPath, "utf8");
const scriptMatch = livePySource.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error(`no <script> block found in ${livePyPath}`);
// Strip the auto-run IIFE at the bottom so the test harness controls timing
// itself instead of a live setInterval firing during the test.
const js = scriptMatch[1].replace(
  /\(async\(\)=>\{ await setup\(\); await poll\(\); setInterval\(poll,600\); \}\)\(\);/, "");

const SETUP = {
  instrument: {id: "bfi2s", name: "BFI-2-S", full_name: "Big Five Inventory-2, Short Form",
    domain_order: ["Extraversion","Agreeableness","Conscientiousness","Negative Emotionality","Open-Mindedness"],
    license_notice: "test license"},
  characters: [{character: "CHAR_01", dev_prepared: true, full_prepared: false,
    dev_memories: 50, total_memories: 1000}],
};

function stateRunning(itemIndex) {
  return {status: "running", error: null, character: "CHAR_01", tier: "dev",
    instrument: SETUP.instrument, item_index: itemIndex, total_items: 30,
    current_item: {number: itemIndex, text: "x", domain: "Extraversion", facet: "Sociability"},
    answered: [], result: null, comparison: null, elapsed_s: itemIndex};
}

function stateDone() {
  return {status: "done", error: null, character: "CHAR_01", tier: "dev",
    instrument: SETUP.instrument, item_index: 30, total_items: 30,
    current_item: null, answered: [],
    result: {run_id: "r1", character: "CHAR_01", occupation: "tester", tier: "dev",
      instrument: {...SETUP.instrument, reliability_note: "n", license_notice: "l"},
      snapshot: {memories: 50, snapshot_id: "abc123def456"},
      items: [{item_number: 1, domain: "Extraversion", facet: "Sociability"}],
      scored: {domain_scores: {Extraversion: 3.0}, facet_scores: {Sociability: 3.0}}},
    comparison: {protocol_note: "note", rows: {O:{name:"Openness",internal:0,bfi2:0,hidden_target:0,bfi2_vs_hidden_pct:100},
      C:{name:"Conscientiousness",internal:0,bfi2:0,hidden_target:0,bfi2_vs_hidden_pct:100},
      E:{name:"Extraversion",internal:0,bfi2:0,hidden_target:0,bfi2_vs_hidden_pct:100},
      A:{name:"Agreeableness",internal:0,bfi2:0,hidden_target:0,bfi2_vs_hidden_pct:100},
      N:{name:"Neuroticism",internal:0,bfi2:0,hidden_target:0,bfi2_vs_hidden_pct:100}},
      summary: {bfi2_vs_hidden_overall_pct: 100, internal_vs_hidden_overall_pct: 100,
        internal_vs_bfi2_overall_pct: 100},
      ground_truth_disclosure: "disclosed"},
    elapsed_s: 12.0};
}

function makeFakeElement(id) {
  return {id, _innerHTML: "", value: "",
    get innerHTML() { return this._innerHTML; },
    set innerHTML(v) { this._innerHTML = v; paintCount++; },
  };
}

let paintCount = 0;
const elements = {};
function getElementById(id) {
  if (!elements[id]) elements[id] = makeFakeElement(id);
  return elements[id];
}

let nextState = null;
async function fakeFetch(url) {
  if (String(url).includes("/api/setup")) return {json: async () => SETUP};
  if (String(url).includes("/api/state")) return {json: async () => nextState};
  return {json: async () => ({})};
}

const sandbox = {
  document: {getElementById},
  fetch: fakeFetch,
  console,
  setInterval: () => {},   // the harness drives poll() itself
  event: null,
};
vm.createContext(sandbox);
vm.runInContext(js, sandbox);

let failures = 0;
function check(label, ok) {
  console.log((ok ? "  PASS  " : "  FAIL  ") + label);
  if (!ok) failures++;
}

(async () => {
  await sandbox.setup();

  console.log("\n1. first poll (idle -> running) paints exactly once");
  nextState = stateRunning(1);
  await sandbox.poll();
  check("root.innerHTML was assigned on the first, genuinely-new state", paintCount === 1);

  console.log("\n2. identical state polled again does NOT repaint (the actual bug)");
  const before = paintCount;
  await sandbox.poll();               // same object -- byte-identical JSON
  check("no repaint on an unchanged poll (this is what keeps a <details> open)",
        paintCount === before);
  await sandbox.poll();
  await sandbox.poll();
  check("still no repaint after several more unchanged polls",
        paintCount === before);

  console.log("\n3. a real change (new item answered) DOES repaint, exactly once");
  nextState = stateRunning(2);
  await sandbox.poll();
  check("repaint count increased by exactly 1 for one real change",
        paintCount === before + 1);

  console.log("\n4. transition to done repaints once, then goes quiet again");
  const beforeDone = paintCount;
  nextState = stateDone();
  await sandbox.poll();
  check("repaints once for the running->done transition", paintCount === beforeDone + 1);
  const afterDone = paintCount;
  await sandbox.poll();
  await sandbox.poll();
  await sandbox.poll();
  check("a finished run polled repeatedly causes NO further repaints "
       + "(elapsed_s freezes; nothing left to legitimately redraw)",
       paintCount === afterDone);

  console.log("\n" + (failures === 0 ? "ALL CHECKS PASSED" : `FAILURES: ${failures}`));
  process.exit(failures === 0 ? 0 : 1);
})();
