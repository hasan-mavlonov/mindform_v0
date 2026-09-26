"""RUN HEART-BENCH — a clickable, watchable benchmark runner.

    python -m bench.heart.live            # then open http://127.0.0.1:8500

Click a mode, click Run, and watch HEART-Bench test MindForm question by
question: the scenario, the four options, the answer as it commits, then the
official ground truth revealed only after that, then the running score.

This is a front end over the harness that already exists. It calls
``runner.execute_question`` and ``runner.grade_question`` -- the same two
functions the CLI uses -- so the integrity guarantees (snapshot restored and
re-hashed per question, prompt leak-checked before sending, answer key untouched
until the answer is written) hold identically here. Every run still writes a
full ``events.jsonl`` through the normal Logbook.

Modes, and the honesty rule between them:

  quick      ~20 official MCQs against an already-prepared snapshot
  character  every official MCQ for one character, same snapshot
  full       prepares the character from all 1,000 memories first, then runs
             every question

Quick and character runs against a 50-memory snapshot are DEVELOPMENT tests.
They use official questions and official scoring, but the character was formed
from a fraction of its memories, so the number is not a protocol-faithful
HEART-Bench score. The UI labels this on every screen and in the export; it is
never presented as the official benchmark.
"""

import argparse
import json
import os
import signal
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bench.heart import (arms, checkpoint, formation, heartdata, mfadapter, runner,
                         snapshots)
from bench.heart.config import (
    RESULTS_ROOT, MODEL, TEMPERATURE, MAX_TOKENS, TOP_K, REASONING_EFFORT,
    require_heart_bench,
)
from bench.heart.logbook import Logbook, new_run_id

ARM_LABEL = {"naive_rag": "Naive RAG", "mindform_d1": "MindForm D1",
             "mindform_d2": "MindForm D2"}
ARM_ORDER = ["naive_rag", "mindform_d1", "mindform_d2"]
QUICK_N = 20

# A short pause between committing an answer and revealing the key, purely so
# the moment is legible on screen. The ordering is enforced in code regardless
# (execute_question never reads the key); this only paces the display.
REVEAL_PAUSE_S = float(os.environ.get("HEART_REVEAL_PAUSE", "0.8"))


class Run:
    """Everything the page needs, behind one lock."""

    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.stop_flag = threading.Event()
        self.reset()

    def reset(self):
        with self.lock:
            self.status = "idle"          # idle|preparing|running|stopping|stopped|done|error
            self.error = None
            self.run_id = None
            self.run_dir = None
            self.mode = None
            self.character = None
            self.selected_arms = []
            self.protocol = None          # dict describing faithfulness
            self.snapshot = None
            self.prepare = {"active": False, "done": 0, "total": 0, "eta_s": None}
            self.formation = None         # live Monitor payload while forming
            self.monitor = None
            self.stopped_at_memory = None
            self.total_questions = 0
            self.question_index = 0
            self.current = None
            self.history = []
            self.tallies = {}
            self.integrity = {"leaks": 0, "mutations": 0, "snapshot_id": None,
                              "snapshot_stable": True}
            self.perf = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                         "latency_ms_total": 0.0, "cost_usd": 0.0, "cost_known": False}
            self.started_at = None
            self.finished_at = None

    # -- snapshot for the page ------------------------------------------------
    def payload(self):
        with self.lock:
            elapsed = ((self.finished_at or time.time()) - self.started_at) \
                if self.started_at else 0
            done = sum(n for _, n in self.tallies.values()) if self.tallies else 0
            total_cells = self.total_questions * max(len(self.selected_arms), 1)
            return {
                "status": self.status, "error": self.error,
                "run_id": self.run_id, "mode": self.mode, "character": self.character,
                "arms": self.selected_arms, "protocol": self.protocol,
                "snapshot": self.snapshot,
                "prepare": dict(self.prepare),
                "formation": (self.monitor.payload() if self.monitor else self.formation),
                "stopped_at_memory": self.stopped_at_memory,
                "question_index": self.question_index,
                "total_questions": self.total_questions,
                "cells_done": done, "cells_total": total_cells,
                "current": self.current,
                "history": self.history,
                "tallies": {a: list(self.tallies.get(a, (0, 0))) for a in self.selected_arms},
                "integrity": dict(self.integrity),
                "perf": dict(self.perf),
                "elapsed_s": round(elapsed, 1),
                "config": {"model": MODEL, "temperature": TEMPERATURE,
                           "max_tokens": MAX_TOKENS, "top_k": TOP_K,
                           "reasoning_effort": REASONING_EFFORT},
            }

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)


RUN = Run()


# ---------------------------------------------------------------------------
def _protocol_for(mode, snap, character):
    """Say plainly whether this run is protocol-faithful, and why not."""
    depth = (snap or {}).get("memories", 0)
    full = depth >= snapshots.FULL_DEPTH
    total = len(heartdata.load_characters()[character].get("episodic_memory_set") or [])
    if mode == "full" or full:
        return {"faithful": True, "tier": "FULL PROTOCOL BENCHMARK",
                "memories": depth or total, "total_memories": total,
                "note": "Character formed from its complete memory set. "
                        "Official questions, official scoring."}
    return {"faithful": False, "tier": "DEVELOPMENT TEST",
            "memories": depth, "total_memories": total,
            "note": f"Character formed from {depth} of {total} memories. Official "
                    f"questions and official scoring, but NOT a protocol-faithful "
                    f"HEART-Bench score — the character is under-formed."}


def _worker(mode, character, selected_arms, n_questions, tier="dev",
            fresh=False):
    log = None
    try:
        chars = heartdata.load_characters()
        scen = heartdata.load_scenarios()
        char = chars[character]

        run_id = new_run_id(f"live-{mode}-{character.lower()}")
        log = Logbook(run_id, RESULTS_ROOT)
        RUN.set(run_id=run_id, run_dir=log.dir, status="preparing")

        # ---- snapshot: reuse the requested tier, or build the full one -----
        # Full Protocol Benchmark always targets the full 1000-memory snapshot,
        # regardless of which tier was selected in the character picker -- that
        # selector exists for quick/character mode, where the tier choice is
        # the whole point (never silently substitute one tier for the other).
        snap = snapshots.best_for_tier(character, "full" if mode == "full" else tier)
        if mode == "full" and not (snap and snap["full_protocol"]):
            all_mem = heartdata.ingestible_memories(char)
            name = f"heartbench {character}"
            decision = formation.plan(character, name, all_mem)
            if decision["action"] == "blocked" and not fresh:
                raise RuntimeError(
                    f"{character} has a checkpoint that cannot be resumed "
                    f"({decision['reason']}). Choose \u201cStart over\u201d to discard "
                    f"{decision['done']} formed memories and begin again.")

            monitor = formation.Monitor(character, len(all_mem),
                                        done_at_start=decision.get("done", 0), echo=True)
            RUN.set(prepare={"active": True, "done": decision.get("done", 0),
                             "total": len(all_mem), "eta_s": None},
                    monitor=monitor)
            log.event("run_start", mode=mode, character=character, arms=selected_arms,
                      n_memories=len(all_mem), model=MODEL, temperature=TEMPERATURE,
                      max_tokens=MAX_TOKENS, top_k=TOP_K, ui="live",
                      resume_action=decision["action"], resume_from=decision.get("done", 0))

            # The formation lock is what stops a second browser tab -- or a CLI run --
            # from forming the same character into the same files at the same time.
            with formation.FormationLock(character):
                formation.form(character, char, all_mem, name, log, monitor,
                               cancel=RUN.stop_flag.is_set, decision=decision,
                               fresh=fresh)
            frozen_id, _ = mfadapter.freeze(name, log.dir, "frozen")
            snap = {"memories": len(all_mem), "full_protocol": True,
                    "snapshot_id": frozen_id, "run_dir": log.dir, "label": "frozen",
                    "bench_name": name, "run_id": run_id, "character": character}
            RUN.set(prepare={"active": False, "done": len(all_mem),
                             "total": len(all_mem), "eta_s": 0},
                    formation=monitor.payload(), monitor=None)
        elif not snap:
            if tier == "full":
                raise RuntimeError(
                    f"{character} has no full (1000-memory) snapshot prepared. Run "
                    f"Full Protocol Benchmark for this character first, or pick its "
                    f"dev snapshot instead if one exists.")
            raise RuntimeError(
                f"{character} has no dev snapshot prepared. Prepare one with:\n"
                f"  python -m bench.heart.runner --stage 0 --character {character}\n"
                f"or run Full Protocol Benchmark for the full 1000-memory character.")

        name = snap["bench_name"]
        snap_run_dir, snap_label = snap["run_dir"], snap["label"]
        frozen_id = snap["snapshot_id"]
        mfadapter.register_id_map(heartdata.ingestible_memories(char))

        questions = heartdata.questions_for(character)
        if mode == "quick":
            questions = questions[:n_questions or QUICK_N]
        forbidden = heartdata.forbidden_strings(char, questions)
        char_pub = heartdata.character_public(char)

        RUN.set(status="running", total_questions=len(questions), snapshot=snap,
                protocol=_protocol_for(mode, snap, character),
                started_at=time.time())
        with RUN.lock:
            RUN.integrity["snapshot_id"] = frozen_id
        if not (mode == "full" and snap.get("run_id") == run_id):
            log.event("run_start", mode=mode, character=character, arms=selected_arms,
                      n_questions=len(questions), model=MODEL, temperature=TEMPERATURE,
                      max_tokens=MAX_TOKENS, top_k=TOP_K, snapshot_id=frozen_id,
                      snapshot_memories=snap["memories"],
                      protocol_faithful=snap["memories"] >= snapshots.FULL_DEPTH,
                      ui="live")

        tallies = {}
        for qi, question in enumerate(questions, 1):
            if RUN.stop_flag.is_set():
                RUN.set(status="stopped")
                log.event("run_stopped", after_questions=qi - 1)
                break
            scenario = scen[question["scenario_id"]]
            row = {"n": qi, "question_id": question["question_id"],
                   "scenario_name": scenario.get("name"), "arms": {}}

            RUN.set(question_index=qi, current={
                "n": qi, "total": len(questions),
                "question_id": question["question_id"],
                "character": character,
                "scenario": {"name": scenario.get("name"),
                             "context": scenario.get("context_text"),
                             "setting": scenario.get("setting")},
                "trigger": scenario.get("trigger_event"),
                "options": heartdata.public_options(question),
                "arm": None, "stage": "starting", "selected": None,
                "ground_truth": None, "correct": None, "per_arm": {},
            })

            for arm in selected_arms:
                if RUN.stop_flag.is_set():
                    break

                def on_stage(kind, payload):
                    with RUN.lock:
                        if RUN.current:
                            RUN.current["arm"] = arm
                            RUN.current["stage"] = kind
                            if kind == "answering":
                                RUN.current["memories"] = payload["memories"]
                                RUN.current["state"] = payload["state"]
                                RUN.current["retrieval"] = payload["retrieval"]

                committed = runner.execute_question(
                    log, arm, char, char_pub, scenario, question, name,
                    snap_run_dir, frozen_id, forbidden,
                    snapshot_label=snap_label, on_stage=on_stage)

                # committed, key still untouched — show the answer alone first
                with RUN.lock:
                    RUN.current["stage"] = "committed"
                    RUN.current["selected"] = committed["selected_answer"]
                    RUN.current["per_arm"][arm] = {
                        "selected": committed["selected_answer"],
                        "ground_truth": None, "correct": None,
                        "latency_ms": committed["latency_ms"],
                        "input_tokens": committed["input_tokens"],
                        "output_tokens": committed["output_tokens"],
                        "raw_output": committed["raw_output"],
                        "prompt": committed["prompt"],
                        "prompt_sha256": committed["prompt_sha256"],
                        "memories": committed["memories_retrieved"],
                        "state": committed["mindform_state"],
                        "retrieval": committed["retrieval"],
                        "parse_method": committed["parse_method"],
                        "errors": committed["errors"],
                    }
                    RUN.perf["calls"] += 1
                    RUN.perf["input_tokens"] += committed["input_tokens"] or 0
                    RUN.perf["output_tokens"] += committed["output_tokens"] or 0
                    RUN.perf["latency_ms_total"] += committed["latency_ms"] or 0
                    if committed["cost_usd"] is not None:
                        RUN.perf["cost_usd"] += committed["cost_usd"]
                        RUN.perf["cost_known"] = True
                time.sleep(REVEAL_PAUSE_S)

                g = runner.grade_question(log, committed, question, name, tallies)
                with RUN.lock:
                    RUN.current["stage"] = "revealed"
                    RUN.current["ground_truth"] = g["ground_truth"]
                    RUN.current["correct"] = g["correct"]
                    RUN.current["per_arm"][arm]["ground_truth"] = g["ground_truth"]
                    RUN.current["per_arm"][arm]["correct"] = g["correct"]
                    RUN.tallies = dict(tallies)
                    if g["state_mutated"]:
                        RUN.integrity["mutations"] += 1
                        RUN.integrity["snapshot_stable"] = False
                row["arms"][arm] = {
                    "selected": committed["selected_answer"],
                    "ground_truth": g["ground_truth"], "correct": g["correct"],
                    "latency_ms": committed["latency_ms"],
                    "input_tokens": committed["input_tokens"],
                    "output_tokens": committed["output_tokens"],
                    "raw_output": committed["raw_output"],
                    "prompt": committed["prompt"],
                    "prompt_sha256": committed["prompt_sha256"],
                    "memories": committed["memories_retrieved"],
                    "state": committed["mindform_state"],
                    "retrieval": committed["retrieval"],
                }
                if g["state_mutated"]:
                    raise runner.RunInvalid(
                        f"state mutated while answering {question['question_id']}")

            row["ground_truth"] = next(
                (v["ground_truth"] for v in row["arms"].values()), None)
            row["options"] = heartdata.public_options(question)
            row["scenario"] = {"name": scenario.get("name"),
                               "context": scenario.get("context_text")}
            row["trigger"] = scenario.get("trigger_event")
            with RUN.lock:
                RUN.history.append(row)

        with RUN.lock:
            if RUN.status == "running":
                RUN.status = "done"
            RUN.finished_at = time.time()
        log.event("run_end", tallies={a: list(tallies.get(a, (0, 0)))
                                      for a in selected_arms},
                  status=RUN.status)

    except formation.FormationPaused as exc:
        # The checkpoint is already on disk; this is a stop, not a loss.
        with RUN.lock:
            mon = RUN.monitor
        RUN.set(status="error", error=str(exc), finished_at=time.time(),
                formation=(mon.payload() if mon else None), monitor=None,
                stopped_at_memory=(mon.done if mon else None))
    except runner.RunInvalid as exc:
        RUN.set(status="error", error=f"RUN INVALID — {exc}", finished_at=time.time())
        with RUN.lock:
            RUN.integrity["leaks"] += 1 if "LEAK" in str(exc) else 0
        if log:
            log.event("run_invalid", reason=str(exc))
    except (mfadapter.Cancelled, KeyboardInterrupt) as exc:
        # Cooperative stop: the engine saves a character only at the end of a
        # memory, so whatever was in flight simply did not happen. The checkpoint
        # already on disk is the truth, and it is what the next run resumes from.
        with RUN.lock:
            mon = RUN.monitor
        done = mon.done if mon else None
        RUN.set(status="stopped", finished_at=time.time(),
                formation=(mon.payload() if mon else None), monitor=None,
                stopped_at_memory=done)
        if mon:
            mon.note(f"stopped safely at {mon.done}/{mon.total} — "
                     f"resume will continue from memory {mon.done + 1}", "■")
        if log:
            log.event("formation_stopped", character=character, completed=done,
                      reason=str(exc))
    except Exception as exc:
        RUN.set(status="error", error=f"{type(exc).__name__}: {exc}",
                finished_at=time.time())
        traceback.print_exc()
        if log:
            log.event("run_error", error=str(exc))


def start_run(mode, character, selected_arms, n_questions=None, tier="dev",
              fresh=False):
    with RUN.lock:
        if RUN.status in ("running", "preparing"):
            return False, "a run is already in progress"
    RUN.reset()
    RUN.stop_flag.clear()
    RUN.set(mode=mode, character=character, selected_arms=selected_arms,
            status="preparing", started_at=time.time())
    t = threading.Thread(target=_worker,
                         args=(mode, character, selected_arms, n_questions, tier,
                               fresh),
                         daemon=True)
    RUN.thread = t
    t.start()
    return True, "started"


# ---------------------------------------------------------------------------
def export_payload():
    p = RUN.payload()
    p["exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p["disclaimer"] = (
        "Internal R&D. HEART-Bench data is CC-BY-NC and its commercial licence is "
        "unresolved. " + ((p.get("protocol") or {}).get("note") or ""))
    return p


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype="application/json"):
        body = (obj if isinstance(obj, bytes)
                else json.dumps(obj).encode("utf-8") if ctype == "application/json"
                else obj.encode("utf-8"))
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            return self._send(RUN.payload())
        if self.path.startswith("/api/setup"):
            return self._send({
                "characters": snapshots.catalogue(),
                "arms": [{"id": a, "label": ARM_LABEL[a]} for a in ARM_ORDER],
                "quick_n": QUICK_N,
                "config": {"model": MODEL, "temperature": TEMPERATURE,
                           "top_k": TOP_K, "reasoning_effort": REASONING_EFFORT},
            })
        if self.path.startswith("/api/export"):
            body = json.dumps(export_payload(), indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition",
                             f'attachment; filename="{RUN.run_id or "heartbench"}.json"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        return self._send(PAGE, ctype="text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path.startswith("/api/start"):
            ok, msg = start_run(body.get("mode", "quick"),
                                body.get("character", "CHAR_01"),
                                body.get("arms") or ["mindform_d2"],
                                body.get("n_questions"),
                                body.get("tier", "dev"),
                                bool(body.get("fresh")))
            return self._send({"ok": ok, "message": msg}, 200 if ok else 409)
        if self.path.startswith("/api/stop"):
            RUN.stop_flag.set()
            with RUN.lock:
                if RUN.status in ("running", "preparing"):
                    RUN.status = "stopping"
            return self._send({"ok": True})
        return self._send({"ok": False, "message": "unknown endpoint"}, 404)

    def log_message(self, *a):
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RUN HEART-BENCH</title>
<style>
:root{
  color-scheme:light;
  --bg:#f2f1ee; --card:#fff; --sunk:#fcfcfb;
  --ink:#0b0b0b; --sec:#52514e; --mut:#83817a; --rule:rgba(11,11,11,.12);
  --a1:#2a78d6; --a2:#eb6834; --a3:#1baf7a;
  --good:#1a7f4b; --good-bg:rgba(26,127,75,.10);
  --bad:#b3261e;  --bad-bg:rgba(179,38,30,.10);
  --warn:#9a6206; --warn-bg:rgba(154,98,6,.12);
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
  color-scheme:dark;
  --bg:#111110; --card:#1e1e1d; --sunk:#171716;
  --ink:#fff; --sec:#c3c2b7; --mut:#8f8e86; --rule:rgba(255,255,255,.15);
  --a1:#3987e5; --a2:#d95926; --a3:#199e70;
  --good:#4ec98a; --good-bg:rgba(78,201,138,.14);
  --bad:#ff8a7a;  --bad-bg:rgba(255,138,122,.14);
  --warn:#f0b458; --warn-bg:rgba(240,180,88,.14);
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);padding:20px}
.wrap{max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
h1{font-size:22px;margin:0;letter-spacing:-.02em}
h1 .dim{color:var(--mut);font-weight:400}
.sub{color:var(--sec);font-size:13.5px;margin:6px 0 16px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:18px}
.sunk{background:var(--sunk)}
label{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--mut);
      display:block;margin-bottom:5px;font-weight:600}
select,button{font:inherit;border-radius:8px;border:1px solid var(--rule);
      background:var(--card);color:var(--ink);padding:9px 12px}
button{cursor:pointer;font-weight:600}
button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:var(--ink);color:var(--bg);border-color:var(--ink);padding:11px 22px}
button.danger{color:var(--bad);border-color:var(--bad)}
.controls{display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap}
.controls > div{display:flex;flex-direction:column}
.armbox{display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:13px}
.armbox label{text-transform:none;letter-spacing:0;font-size:13px;color:var(--ink);
      margin:0;display:flex;gap:6px;align-items:center;font-weight:400;cursor:pointer}
.tier{display:inline-block;font:600 11px/1 var(--mono);letter-spacing:.08em;
      padding:6px 10px;border-radius:6px;text-transform:uppercase}
.tier.dev{background:var(--warn-bg);color:var(--warn)}
.tier.full{background:var(--good-bg);color:var(--good)}
.bar{height:7px;background:var(--sunk);border-radius:99px;overflow:hidden;
     border:1px solid var(--rule)}
.bar span{display:block;height:100%;background:var(--ink);transition:width .3s}
.qhead{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.qnum{font:600 13px/1 var(--mono);color:var(--mut);letter-spacing:.06em}
.scen{font-size:19px;font-weight:600;margin:10px 0 6px;letter-spacing:-.01em}
.ctx{color:var(--sec);font-size:14px}
.trig{font-family:var(--mono);font-size:12.5px;background:var(--sunk);
      border:1px solid var(--rule);border-radius:8px;padding:10px 12px;margin-top:12px;
      color:var(--sec)}
.opts{display:flex;flex-direction:column;gap:8px;margin-top:14px}
.opt{display:grid;grid-template-columns:30px 1fr;gap:10px;padding:11px 13px;
     border:1px solid var(--rule);border-radius:9px;font-size:14px;background:var(--card);
     transition:background .2s,border-color .2s}
.opt b{font:700 14px/1.4 var(--mono)}
.opt.picked{border-color:var(--ink);background:var(--sunk)}
.opt.truth{border-color:var(--good);background:var(--good-bg)}
.opt.wrong{border-color:var(--bad);background:var(--bad-bg)}
.verdict{margin-top:16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
         font-size:16px;font-weight:600}
.verdict .ok{color:var(--good)} .verdict .no{color:var(--bad)}
.thinking{display:inline-flex;gap:8px;align-items:center;color:var(--mut);
          font-size:15px;font-weight:500}
.dot{width:8px;height:8px;border-radius:50%;background:var(--mut);
     animation:pulse 1.1s infinite ease-in-out}
@keyframes pulse{0%,100%{opacity:.25;transform:scale(.85)}50%{opacity:1;transform:scale(1.15)}}
.scores{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.score{border:1px solid var(--rule);border-radius:10px;padding:12px 14px;background:var(--card)}
.score .nm{font:600 11px/1 var(--mono);letter-spacing:.07em;text-transform:uppercase}
.score .pc{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:6px}
.score .fr{font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}
.score.a1{border-left:3px solid var(--a1)} .score.a2{border-left:3px solid var(--a2)}
.score.a3{border-left:3px solid var(--a3)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule)}
th{font:600 10.5px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}
tr.qrow{cursor:pointer} tr.qrow:hover{background:var(--sunk)}
td.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.exp{background:var(--sunk);font-size:13px}
.exp .sec{margin:10px 0}
.exp h4{margin:0 0 6px;font:600 11px/1 var(--mono);letter-spacing:.08em;
        text-transform:uppercase;color:var(--mut)}
pre{white-space:pre-wrap;word-break:break-word;font-family:var(--mono);font-size:11.5px;
    background:var(--card);border:1px solid var(--rule);border-radius:7px;padding:10px;
    max-height:280px;overflow:auto;margin:0}
.mem{display:grid;grid-template-columns:26px 54px 1fr;gap:8px;font-family:var(--mono);
     font-size:11px;padding:3px 0;border-bottom:1px solid var(--rule)}
.muted{color:var(--mut)} .small{font-size:12px}
.banner{border-radius:10px;padding:12px 15px;font-weight:600;font-size:14px}
.banner.err{background:var(--bad-bg);color:var(--bad);border:1px solid var(--bad)}
.banner.info{background:var(--sunk);border:1px solid var(--rule);color:var(--sec);font-weight:400}
.done{text-align:center;padding:10px 0}
.done .big{font-size:30px;font-weight:700;letter-spacing:-.02em}
.chk{display:flex;flex-direction:column;gap:5px;font-size:13.5px;margin-top:12px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px 18px;
  margin-top:14px}
.fgrid .k{font:600 10.5px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;
  color:var(--mut)}
.fgrid .v{font-size:17px;font-weight:600;margin-top:3px;letter-spacing:-.01em}
.fgrid .v.sm{font-size:13.5px;font-weight:500}
.stages{display:flex;flex-direction:column;gap:4px;margin-top:12px;font:12.5px/1.5 var(--mono)}
.stages div{display:flex;gap:8px}
.stages .lbl{flex:1}
.waiting{color:var(--mut);font-style:italic}
.feed{margin-top:12px;max-height:230px;overflow-y:auto;background:var(--sunk);
  border:1px solid var(--rule);border-radius:8px;padding:10px 12px;
  font:12px/1.75 var(--mono)}
.feed div{display:flex;gap:9px;white-space:pre-wrap}
.feed .ts{color:var(--mut);flex:none}
.feed .ic{flex:none;width:1em;text-align:center}
.tag{display:inline-block;font:600 10px/1.6 var(--mono);letter-spacing:.07em;
  padding:1px 7px;border-radius:99px;border:1px solid var(--rule);color:var(--sec);
  text-transform:uppercase;margin-left:8px;vertical-align:2px}
.tag.ok{border-color:var(--good);color:var(--good)}
.tag.warn{border-color:var(--bad);color:var(--bad)}
</style></head><body>
<div class="wrap">
  <h1>RUN HEART-BENCH <span class="dim">· MindForm</span></h1>
  <p class="sub">Does it behave like that person? — MindForm's formed character
    is given real, unseen scenarios and checked against a hidden, official answer.</p>
  <div id="r-controls"></div>
  <div id="r-banners"></div>
  <div id="r-formation"></div>
  <div id="r-progress"></div>
  <div id="r-main"></div>
</div>
<script>
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const ARMCLS={naive_rag:"a1",mindform_d1:"a2",mindform_d2:"a3"};
const ARMNM={naive_rag:"Naive RAG",mindform_d1:"MindForm D1",mindform_d2:"MindForm D2"};
let SETUP=null, STATE=null, OPEN=new Set(), DEBUG=false, RESEARCH_OPEN=false;
function toggleResearch(){ RESEARCH_OPEN=!RESEARCH_OPEN; render(); }
let PICKED=new Set(["mindform_d2"]), MODE="quick", CHARSEL="CHAR_01::dev";
function pick(a,on){ on?PICKED.add(a):PICKED.delete(a); }

async function setup(){ SETUP=await (await fetch("/api/setup")).json(); }
let WAS_RUNNING=false;
async function poll(){
  try{
    const j=await (await fetch("/api/state",{cache:"no-store"})).json();
    STATE=j;
    const now=isRunning();
    // A run that has just ended has almost certainly moved the checkpoint, and
    // the catalogue is what decides whether the button says Run or Resume -- so
    // refetch it once here instead of making the user reload the page.
    if(WAS_RUNNING && !now){ WAS_RUNNING=false; await setup(); }
    if(now) WAS_RUNNING=true;
    render();
  }catch(e){}
}
function reselect(){
  // The picked mode/character decide whether this is a Run or a Resume, so the
  // controls have to repaint on change -- safe here, nothing is in flight.
  MODE=document.getElementById("mode").value;
  CHARSEL=document.getElementById("char").value;
  render();
}

async function launch(fresh){
  const mode=document.getElementById("mode").value;
  // The character selector's value is "CHAR_id::tier" -- dev and full snapshots
  // are never the same thing, so which tier was picked always travels with the
  // character id, right through to the /api/start body.
  const [character,tier]=document.getElementById("char").value.split("::");
  const arms=[...document.querySelectorAll(".arm:checked")].map(e=>e.value);
  if(!arms.length){ alert("Pick at least one system to test."); return; }
  const c=SETUP.characters.find(x=>x.character===character);
  const f=c.formation||{};
  if(mode==="full"){
    if(fresh){
      // Destructive: this is the only path that throws formed memories away, so
      // it is never reached by clicking the normal button.
      if(!confirm(`Start over and DELETE existing progress for ${character}?\n\n`
        +`${f.done||0} formed memories will be permanently discarded and the character `
        +`will be rebuilt from memory 1. At the measured rate that is about `
        +`${(c.total_memories*33/3600).toFixed(1)} hours of work.\n\nThis cannot be undone.`)) return;
      if(!confirm(`Really delete ${f.done||0} formed memories for ${character}?`)) return;
    } else if(f.resumable){
      if(!confirm(`Resume ${character} from memory ${f.resume_from}?\n\n`
        +`${f.done} of ${f.total} memories are already formed and checkpointed. `
        +`About ${(((c.total_memories-f.done)*33)/3600).toFixed(1)} hours remain.`)) return;
    } else if(f.state==="corrupt"||f.state==="stale"){
      alert(`${character} has a checkpoint that cannot be resumed:\n\n${f.detail}\n\n`
        +`Use “Start over” to discard it and form the character again.`);
      return;
    } else if(!c.full_prepared){
      if(!confirm(`Full Protocol Benchmark for ${character}\n\nThis forms the character `
        +`from all ${c.total_memories} memories before answering. Measured rate is ~33 s `
        +`per memory, so expect roughly ${(c.total_memories*33/3600).toFixed(1)} hours `
        +`before the first question.\n\nProgress is checkpointed after every memory, so `
        +`you can stop and resume at any time.\n\nStart?`)) return;
    }
  } else if(tier==="full" && !c.full_prepared){
    alert(`${character}'s full ${c.total_memories}-memory snapshot hasn't been prepared yet, `
      +`so Quick Test and Character Test have nothing to answer from at that tier.`
      +(f.resumable?`\n\nIts formation is checkpointed at ${f.done}/${f.total} memories — `
        +`switch to Full Protocol Benchmark to resume it.`:"")
      +`\n\nRun a Full Protocol Benchmark for this character first, or pick its dev snapshot.`);
    return;
  } else if(tier!=="full" && !c.dev_prepared){
    alert(`${character} has no dev snapshot prepared, so Quick Test and Character Test have `
      +`nothing to answer from.\n\nPrepare one with:\n`
      +`  python -m bench.heart.runner --stage 0 --character ${character}\n`
      +`or run a Full Protocol Benchmark for the full 1,000-memory character.`);
    return;
  }
  Object.keys(LAST).forEach(k=>delete LAST[k]);   // force a full repaint on start
  FEED_N=0;
  const r=await fetch("/api/start",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({mode,character,tier,arms,fresh:!!fresh})});
  if(!r.ok){ const j=await r.json(); alert(j.message); }
  poll();
}
async function start(){ return launch(false); }
async function startOver(){ return launch(true); }

async function stop(){
  const btn=event?.target;
  if(btn){ btn.disabled=true; btn.textContent="Stopping…"; }
  await fetch("/api/stop",{method:"POST"});
  poll();
}

function controls(){
  const running=STATE && ["running","preparing","stopping"].includes(STATE.status);
  const chars=(SETUP?.characters||[]);
  const curMode=document.getElementById("mode")?.value||MODE;
  const curChar=document.getElementById("char")?.value||CHARSEL;
  MODE=curMode; CHARSEL=curChar;
  // Each character offers up to two distinct options -- a dev (partial-memory)
  // snapshot and the full 1,000-memory one -- so a 50-memory dev run is never
  // presented, or selectable, as if it were the full benchmark.
  const opts=chars.flatMap(c=>{
    const rows=[];
    if(c.dev_prepared){
      const v=`${c.character}::dev`;
      rows.push(`<option value="${v}" ${v===curChar?"selected":""}>`
        + `${c.character} — dev snapshot (${c.dev_memories} memories) — `
        + `${c.questions} questions</option>`);
    }
    const fv=`${c.character}::full`;
    const f=c.formation||{};
    // A half-formed character is neither "prepared" nor "not prepared" -- saying
    // "not prepared" is what made an interrupted 252-memory run look like it had
    // never happened, so its progress is named right here in the option.
    let fullTag;
    if(c.full_prepared) fullTag="prepared";
    else if(f.resumable) fullTag=`${f.done}/${f.total} memories — resumable`;
    else if(f.state==="corrupt"||f.state==="stale")
      fullTag=`${f.done}/${f.total} memories — checkpoint unusable`;
    else fullTag="not prepared";
    rows.push(`<option value="${fv}" ${fv===curChar?"selected":""}>`
      + `${c.character} — full snapshot (${c.total_memories} memories, ${fullTag}) — `
      + `${c.questions} questions</option>`);
    return rows;
  }).join("");
  const [curId,curTier]=(curChar||"").split("::");
  const curCharObj=chars.find(c=>c.character===curId)||{};
  // Resume only makes sense for the full tier under Full Protocol -- that is the
  // only run that forms anything, so it is the only one with progress to resume.
  const curF=(curTier==="full"&&curMode==="full")?(curCharObj.formation||null):null;
  const resumable=!!(curF&&curF.resumable);
  return `<div class="card"><div class="controls">
    <div><label>Mode</label><select id="mode" ${running?"disabled":""} onchange="reselect()">
      <option value="quick" ${curMode==="quick"?"selected":""}>Quick Test — ~${SETUP?.quick_n||20} questions</option>
      <option value="character" ${curMode==="character"?"selected":""}>Character Test — all questions</option>
      <option value="full" ${curMode==="full"?"selected":""}>Full Protocol Benchmark — ingest 1,000 memories first</option>
    </select>
    <p class="small muted" style="margin-top:6px">${({
      quick: "Answers a handful of real, unseen scenarios against this character's "
            + "existing snapshot — the fastest way to watch it work.",
      character: "Answers every real, unseen scenario for this character, against "
                + "its existing snapshot.",
      full: "Forms this character from all 1,000 memories first — the expensive "
           + "step, typically hours — then answers every scenario. An already "
           + "formed (or partly formed) character resumes instead of restarting: "
           + "this should not be re-run just to see results.",
    })[curMode]}</p></div>
    <div><label>Character</label><select id="char" ${running?"disabled":""}
      onchange="reselect()">${opts}</select></div>
    <div><label>Systems</label><div class="armbox">${
      ["naive_rag","mindform_d1","mindform_d2"].map(a=>{
        // While a run is in flight the boxes mirror what is actually running,
        // so the controls never disagree with the live results below.
        const on = running ? (STATE.arms||[]).includes(a) : PICKED.has(a);
        return `<label><input type="checkbox" class="arm" value="${a}" ${on?"checked":""}
          ${running?"disabled":""} onchange="pick('${a}',this.checked)"> ${ARMNM[a]}</label>`;
      }).join("")}
    </div></div>
    <div class="row">
      <button class="primary" onclick="start()" ${running?"disabled":""}>${
        resumable?"Resume Benchmark":"Run Benchmark"}</button>
      <button class="danger" onclick="stop()" ${running?"":"disabled"}>Stop</button>
      ${resumable?`<button onclick="startOver()" ${running?"disabled":""}>Start over…</button>`:""}
    </div>
    ${resumable?`<div class="small muted" style="margin-top:10px">
      ${esc(curCharObj.character)} is part-formed: <b>${curF.done} / ${curF.total}</b>
      memories checkpointed${curF.updated_at?` at ${esc(curF.updated_at)}`:""}.
      Resume continues from memory <b>${curF.resume_from}</b>.
      ${curF.adopted?"(recovered from an earlier run's leftover state)":""}</div>`:""}
    ${curF && (curF.state==="corrupt"||curF.state==="stale")?`<div class="banner err"
      style="margin-top:10px">Checkpoint at ${curF.done}/${curF.total} cannot be
      resumed: ${esc(curF.detail||"")}. Use “Start over” to discard it.</div>`:""}
    </div>
    <div class="small muted" style="margin-top:12px">
      model <span class="mono">${esc(SETUP?.config?.model)}</span> ·
      temp ${SETUP?.config?.temperature} · top-k ${SETUP?.config?.top_k} ·
      reasoning ${esc(SETUP?.config?.reasoning_effort||"default")} ·
      every arm uses the same model and settings
    </div>
    ${researchDetails()}
    </div>`;
}
// A plain-language summary of what this test actually checks under the hood.
// Custom click handler + JS-tracked RESEARCH_OPEN (folded into r-controls'
// paint signature above), not a native <details> -- this block lives inside
// the same region that legitimately repaints on ordinary interaction (picking
// an arm, switching mode or character), and a native <details>'s "open"
// attribute would be wiped by that repaint even though nothing about ITS
// content changed. Tracking the open state in JS instead of the DOM survives
// any repaint, by construction, rather than by coincidence.
function researchDetails(){
  const body = RESEARCH_OPEN ? `<div class="small muted" style="margin-top:8px;line-height:1.6">
      HEART gives the formed character unseen behavioral situations and checks
      whether it makes the same choice as the person it was formed to be —
      compared against a hidden, official ground-truth answer it never sees.
      <br><br>
      Three ways of answering are compared side by side:
      <br>&nbsp;&nbsp;<b>Naive RAG</b> — plain memory lookup, no persistent state
      <br>&nbsp;&nbsp;<b>MindForm D1</b> — MindForm's own retrieval, same prompt shape as Naive RAG
      <br>&nbsp;&nbsp;<b>MindForm D2</b> — D1 plus the character's persistent formed-personality state
      <br><br>
      Every prompt is scanned before it's sent to make sure the hidden answer
      (and every other withheld label) never leaks in, the frozen snapshot is
      re-verified byte-for-byte after every question, and the ground truth is
      only revealed after an answer is already committed — so nothing here can
      see the answer key in advance.
    </div>` : "";
  return `<div style="margin-top:10px">
    <div class="small" style="cursor:pointer;color:var(--a1)" onclick="toggleResearch()">
      Research details ${RESEARCH_OPEN?"▴":"▾"}</div>
    ${body}</div>`;
}

function tierBanner(){
  const p=STATE?.protocol; if(!p) return "";
  return `<div class="card" style="padding:14px 18px">
    <div class="row"><span class="tier ${p.faithful?'full':'dev'}">${esc(p.tier)}</span>
    <span class="small muted">${esc(p.note)}</span></div></div>`;
}

function fmtDur(s){
  if(s==null||!isFinite(s)) return "—";
  s=Math.max(0,Math.round(s));
  if(s<90) return `${s}s`;
  const h=Math.floor(s/3600), m=Math.round((s%3600)/60);
  return h?`${h}h ${m}m`:`${m}m`;
}

// The activity feed is appended to, never rebuilt: rewriting it each poll would
// throw away the reader's scroll position several times a second.
let FEED_N=0;
function feedAppend(entries){
  const box=document.getElementById("feedbox");
  if(!box) return;
  if(entries.length<FEED_N){ box.innerHTML=""; FEED_N=0; }   // new run, new feed
  const atBottom=box.scrollHeight-box.scrollTop-box.clientHeight<40;
  for(let i=FEED_N;i<entries.length;i++){
    const e=entries[i], row=document.createElement("div");
    row.innerHTML=`<span class="ts">${esc(e.t)}</span>`
                 +`<span class="ic">${esc(e.icon)}</span>`
                 +`<span>${esc(e.text)}</span>`;
    box.appendChild(row);
  }
  FEED_N=entries.length;
  if(atBottom) box.scrollTop=box.scrollHeight;
}

function formationPanel(){
  const f=STATE?.formation;
  const el=document.getElementById("r-formation");
  if(!f){ if(LAST["r-formation"]!==null){ LAST["r-formation"]=null; el.innerHTML=""; FEED_N=0; } return; }
  // The shell (everything but the feed) is cheap to rebuild and has no controls
  // in it, so it repaints freely; the feed is appended to separately.
  const active=STATE.prepare?.active;
  const stage=f.stage?`${esc(f.stage)} — waiting for the model`:
              (active?"between stages":"—");
  const slow=f.stage&&f.stage_elapsed_s>15;
  const eta=f.eta_s!=null?fmtDur(f.eta_s):
    `not yet — needs ${5-Math.min(5,f.done-f.resumed_from)} more memories`;
  const stages=(f.stages_done||[]).map(s=>
    `<div><span class="ic">${s.error?"!":"✓"}</span><span class="lbl">${esc(s.label)}</span>`
    +`<span class="muted">${s.seconds}s</span></div>`).join("")
    +(f.stage?`<div class="waiting"><span class="ic">→</span><span class="lbl">`
      +`${esc(f.stage)} — waiting…</span><span class="muted">`
      +`${f.stage_elapsed_s??0}s</span></div>`:"");
  const sig=JSON.stringify([f.done,f.total,f.stage,f.stage_elapsed_s,f.current_memory,
    f.stages_done,f.eta_s,f.avg_s,f.avg_last10_s,f.llm_calls,f.retries,f.errors,
    f.last_checkpoint,f.paused_reason,Math.round(f.elapsed_s||0),active]);
  if(LAST["r-formation"]!==sig){
    LAST["r-formation"]=sig;
    const pct=f.total?100*f.done/f.total:0;
    el.innerHTML=`<div class="card">
      <div class="row" style="justify-content:space-between">
        <label style="margin:0">Forming ${esc(f.character)} from its memories
          ${f.resumed_from?`<span class="tag ok">resumed at ${f.resumed_from}</span>`:""}
        </label>
        <span class="qnum">${f.done} / ${f.total} · ${f.percent}%</span>
      </div>
      <div class="bar" style="margin-top:10px"><span style="width:${pct}%"></span></div>
      ${f.paused_reason?`<div class="banner err" style="margin-top:12px">
        ${esc(f.paused_reason)}</div>`:""}
      <div class="fgrid">
        <div><div class="k">Current memory</div><div class="v sm">
          ${esc(f.current_memory?.anon_id||"—")}</div></div>
        <div><div class="k">Current stage</div><div class="v sm ${slow?"waiting":""}">
          ${stage}${slow?` · ${f.stage_elapsed_s}s`:""}</div></div>
        <div><div class="k">Elapsed</div><div class="v">${fmtDur(f.elapsed_s)}</div></div>
        <div><div class="k">Avg / memory</div><div class="v">
          ${f.avg_s!=null?f.avg_s+"s":"—"}</div></div>
        <div><div class="k">Avg last 10</div><div class="v">
          ${f.avg_last10_s!=null?f.avg_last10_s+"s":"—"}</div></div>
        <div><div class="k">Est. remaining</div><div class="v">${eta}</div></div>
        <div><div class="k">LLM calls</div><div class="v">
          ${(f.llm_calls||0).toLocaleString()}</div></div>
        <div><div class="k">Retries / errors</div><div class="v">
          ${f.retries||0} / ${f.errors||0}</div></div>
        <div><div class="k">Last checkpoint</div><div class="v sm">
          ${f.last_checkpoint?`memory ${f.last_checkpoint.done}<br>
            <span class="muted">${esc(f.last_checkpoint.at)}</span>`:"—"}</div></div>
        <div><div class="k">Resume status</div><div class="v sm">
          <span class="tag ok">SAFE</span></div></div>
      </div>
      ${stages?`<div class="stages">${stages}</div>`:""}
      <div class="feed" id="feedbox"></div>
      <div class="small muted" style="margin-top:8px">Every completed memory is
        checkpointed to disk. Stopping here — or Ctrl+C in the terminal — resumes from
        the next memory, never from the beginning.</div>
    </div>`;
    FEED_N=0;
  }
  feedAppend(f.feed||[]);
}

function progress(){
  if(!STATE) return "";
  if(STATE.prepare?.active) return "";
  if(!STATE.total_questions) return "";
  const pct=100*STATE.cells_done/Math.max(STATE.cells_total,1);
  return `<div class="card" style="padding:14px 18px">
    <div class="row" style="justify-content:space-between">
      <span class="qnum">QUESTION ${STATE.question_index} / ${STATE.total_questions}</span>
      <span class="small muted">${STATE.cells_done} / ${STATE.cells_total} answers ·
        ${(STATE.elapsed_s/60).toFixed(1)} min</span></div>
    <div class="bar" style="margin-top:9px"><span style="width:${pct}%"></span></div></div>`;
}

function liveQuestion(){
  const c=STATE?.current; if(!c) return "";
  const per=c.per_arm||{};
  let optCls=l=>{
    if(c.ground_truth&&l===c.ground_truth) return "opt truth";
    const picks=Object.values(per).map(v=>v.selected);
    if(c.ground_truth&&picks.includes(l)) return "opt wrong";
    if(picks.includes(l)) return "opt picked";
    return "opt";
  };
  const opts=(c.options||[]).map(o=>{
    const who=Object.entries(per).filter(([,v])=>v.selected===o.label)
              .map(([a])=>ARMNM[a]).join(", ");
    return `<div class="${optCls(o.label)}"><b>${o.label}</b><div>${esc(o.content)}
      ${who?`<div class="small muted" style="margin-top:5px">← ${esc(who)}</div>`:""}</div></div>`;
  }).join("");

  let status="";
  if(c.stage==="answering"||c.stage==="retrieving"||c.stage==="restoring"||c.stage==="starting"){
    status=`<div class="thinking"><span class="dot"></span>
      ${ARMNM[c.arm]||"MindForm"} is ${c.stage==="answering"?"answering":c.stage}…</div>`;
  } else if(c.stage==="committed"){
    status=`<div class="verdict"><span>${ARMNM[c.arm]}: <b>${esc(c.selected)}</b></span>
      <span class="muted small">answer committed — revealing ground truth…</span></div>`;
  } else if(c.stage==="revealed"){
    const rows=Object.entries(per).map(([a,v])=>
      `<span>${ARMNM[a]}: <b>${esc(v.selected)}</b>
        <span class="${v.correct?'ok':'no'}">${v.correct?"✅":"❌"}</span></span>`).join("");
    status=`<div class="verdict">${rows}
      <span class="muted">Ground truth: <b>${esc(c.ground_truth)}</b></span></div>`;
  }

  const trig=c.trigger||{};
  return `<div class="card">
    <div class="qhead"><span class="qnum">${esc(c.character)} · QUESTION ${c.n} / ${c.total}</span>
      <span class="qnum muted">${esc(c.question_id)}</span></div>
    <div class="scen">${esc(c.scenario?.name)}</div>
    <div class="ctx">${esc((c.scenario?.context||"").slice(0,420))}${(c.scenario?.context||"").length>420?"…":""}</div>
    ${trig.message_content?`<div class="trig"><b>${esc(trig.sender||"")}:</b>
      ${esc(String(trig.message_content).slice(0,300))}…</div>`:""}
    <div class="opts">${opts}</div>
    ${status}</div>`;
}

function scores(){
  if(!STATE?.arms?.length) return "";
  const cards=STATE.arms.map(a=>{
    const [c,n]=STATE.tallies[a]||[0,0];
    const pc=n?(100*c/n).toFixed(1):"—";
    return `<div class="score ${ARMCLS[a]}"><div class="nm">${ARMNM[a]}</div>
      <div class="pc">${pc}${n?"%":""}</div><div class="fr">${c} / ${n} correct</div></div>`;
  }).join("");
  return `<div class="card"><label>Running score</label>
    <div class="scores">${cards}</div>
    ${STATE.arms.length>1?'<div class="small muted" style="margin-top:10px">Same model, '
      +'same settings, same questions. Differences between arms are not statistically '
      +'meaningful at small question counts.</div>':""}</div>`;
}

function history(){
  const h=STATE?.history||[]; if(!h.length) return "";
  const arms=STATE.arms;
  const rows=h.map(r=>{
    const cells=arms.map(a=>{
      const v=r.arms[a];
      return v?`<td class="mono">${esc(v.selected||"∅")} ${v.correct?"✅":"❌"}</td>`
              :'<td class="muted">—</td>';
    }).join("");
    const open=OPEN.has(r.n);
    let exp="";
    if(open){
      const blocks=arms.filter(a=>r.arms[a]).map(a=>{
        const v=r.arms[a];
        const mems=(v.memories||[]).slice(0,DEBUG?30:6).map((m,i)=>
          `<div class="mem"><span class="muted">${i+1}</span>
           <span>${(m.score??0).toFixed(3)}</span>
           <span>${esc(m.anon_id)} · ${esc((m.text||"").slice(0,90))}…</span></div>`).join("");
        const st=v.state?`<div class="sec"><h4>MindForm state injected</h4><pre>${
          esc(JSON.stringify(v.state,null,1))}</pre></div>`:"";
        return `<div class="sec"><h4>${ARMNM[a]} — chose ${esc(v.selected)} ${
            v.correct?"✅":"❌"} · ${v.latency_ms} ms · ${v.input_tokens}→${v.output_tokens} tok</h4>
          <div class="sec"><h4>retrieved memories (${(v.memories||[]).length})</h4>${mems}</div>
          ${st}
          <div class="sec"><h4>raw model output</h4><pre>${esc(v.raw_output||"")}</pre></div>
          ${DEBUG?`<div class="sec"><h4>exact prompt · sha ${esc((v.prompt_sha256||"").slice(0,12))}</h4>
            <pre>${esc(v.prompt||"")}</pre></div>`:""}
        </div>`;
      }).join("");
      const opts=(r.options||[]).map(o=>`<div class="opt"><b>${o.label}</b><div>${
        esc(o.content)}</div></div>`).join("");
      exp=`<tr class="exp"><td colspan="${arms.length+3}">
        <div class="sec"><h4>scenario</h4><b>${esc(r.scenario?.name)}</b>
          <div class="small muted">${esc(r.scenario?.context||"")}</div></div>
        <div class="sec"><h4>options</h4><div class="opts">${opts}</div></div>
        ${blocks}</td></tr>`;
    }
    return `<tr class="qrow" onclick="toggle(${r.n})">
      <td class="mono">Q${String(r.n).padStart(2,"0")}</td>
      <td class="small">${esc(r.scenario_name||"")}</td>
      ${cells}<td class="mono">${esc(r.ground_truth||"")}</td></tr>${exp}`;
  }).join("");
  return `<div class="card"><div class="row" style="justify-content:space-between">
      <label style="margin:0">Question history</label>
      <label style="margin:0;text-transform:none;letter-spacing:0;font-size:12.5px;
        color:var(--sec);display:flex;gap:6px;align-items:center;cursor:pointer">
        <input type="checkbox" ${DEBUG?"checked":""} onchange="DEBUG=this.checked;render()">
        Debug (show exact prompts)</label></div>
    <div class="small muted" style="margin:6px 0 10px">Click a row to expand.</div>
    <table><thead><tr><th>#</th><th>Scenario</th>
      ${arms.map(a=>`<th>${ARMNM[a]}</th>`).join("")}<th>Truth</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
}

function toggle(n){ OPEN.has(n)?OPEN.delete(n):OPEN.add(n); render(); }

function finished(){
  if(!STATE||!["done","stopped"].includes(STATE.status)) return "";
  const p=STATE.protocol||{};
  const blocks=STATE.arms.map(a=>{
    const [c,n]=STATE.tallies[a]||[0,0];
    return `<div><div class="nm small muted" style="font-family:var(--mono);
      text-transform:uppercase;letter-spacing:.07em">${ARMNM[a]}</div>
      <div class="big">${n?(100*c/n).toFixed(1):"—"}%</div>
      <div class="small muted">${c} / ${n} correct</div></div>`;
  }).join("");
  const ig=STATE.integrity;
  return `<div class="card"><div class="done">
    <div class="qnum">${STATE.status==="stopped"?"STOPPED EARLY":"COMPLETE"}</div>
    <h2 style="margin:6px 0 4px">HEART-Bench ${esc(p.tier||"")}</h2>
    <div class="small muted">${esc(p.note||"")}</div>
    <div class="scores" style="margin-top:18px">${blocks}</div>
    <div class="chk" style="text-align:left;max-width:420px;margin:18px auto 0">
      <span class="${ig.leaks?'no':'ok'}" style="color:${ig.leaks?'var(--bad)':'var(--good)'}">
        ${ig.leaks?"✗":"✓"} ${ig.leaks?ig.leaks+" ground-truth leak(s) detected":"No ground-truth leakage"}</span>
      <span style="color:${ig.snapshot_stable?'var(--good)':'var(--bad)'}">
        ${ig.snapshot_stable?"✓":"✗"} Snapshot ${ig.snapshot_stable?"unchanged":"MUTATED"}
        <span class="muted mono">${esc((ig.snapshot_id||"").slice(0,12))}</span></span>
      <span style="color:var(--good)">✓ All questions logged
        <span class="muted mono">${esc(STATE.run_id||"")}</span></span>
    </div>
    <div class="row" style="justify-content:center;margin-top:18px">
      <button onclick="document.querySelector('table')?.scrollIntoView({behavior:'smooth'})">
        View question results</button>
      <button onclick="location.href='/api/export'">Export JSON</button>
    </div></div></div>`;
}

// Replacing a region's innerHTML destroys every node inside it -- including the
// button the user is in the middle of clicking. A click only fires if mousedown
// and mouseup land on the SAME element, so a control that is re-rendered every
// poll can never be clicked at all. That is what made the Stop button dead
// during a run: STATE.elapsed_s changes on every single poll, so the old
// whole-page diff never matched and the page was rebuilt at 600ms forever.
// Each region now re-renders only when its OWN inputs change, and the controls
// region is frozen outright while a run is in flight.
const LAST={};
function paint(id, sig, build){
  if(LAST[id]===sig) return;
  LAST[id]=sig;
  document.getElementById(id).innerHTML=build();
}
function isRunning(){
  return !!(STATE && ["running","preparing","stopping"].includes(STATE.status));
}
function render(){
  const running=isRunning();
  // While a run is in flight nothing in the controls is editable, so their
  // content is pinned: no re-render, no destroyed Stop button.
  paint("r-controls", JSON.stringify([running, MODE, CHARSEL, [...PICKED].sort(),
    RESEARCH_OPEN,
    running?null:(SETUP?.characters||[]).map(c=>[c.character,c.dev_prepared,
      c.full_prepared,c.dev_memories,c.total_memories,c.questions,
      c.formation?.state,c.formation?.done,c.formation?.resumable])]), controls);
  paint("r-banners", JSON.stringify([STATE?.error,STATE?.status,
    STATE?.stopped_at_memory,STATE?.protocol?.tier]), banners);
  formationPanel();
  paint("r-progress", JSON.stringify([STATE?.prepare?.active,STATE?.question_index,
    STATE?.total_questions,STATE?.cells_done,STATE?.cells_total,
    Math.round(STATE?.elapsed_s||0)]), progress);
  paint("r-main", JSON.stringify([STATE?.status,STATE?.current,STATE?.history?.length,
    STATE?.tallies,STATE?.perf,[...OPEN],DEBUG,
    STATE?.history?.length?STATE.history[STATE.history.length-1]:null]),
    ()=>finished()+liveQuestion()+scores()+history());
}
function banners(){
  let h="";
  if(STATE?.error) h+=`<div class="banner err">${esc(STATE.error)}</div>`;
  if(STATE?.status==="stopping")
    h+=`<div class="banner info"><b>Stopping…</b> finishing the memory in flight and
        saving the checkpoint. Nothing already completed is lost.</div>`;
  if(STATE?.status==="stopped" && STATE?.stopped_at_memory!=null)
    h+=`<div class="banner info"><b>Stopped safely at
        ${STATE.stopped_at_memory} / ${STATE.formation?.total||"?"} memories.</b><br>
        Resume will continue from memory ${STATE.stopped_at_memory+1}. The checkpoint
        is on disk; the server is still running.</div>`;
  if(STATE?.status==="idle")
    h+=`<div class="banner info">Pick a mode and click <b>Run Benchmark</b>. Quick Test
        against a prepared character takes a couple of minutes.</div>`;
  h+=tierBanner();
  return h;
}

(async()=>{ await setup(); await poll(); setInterval(poll,600); })();
</script></body></html>"""


def _drain_worker(reason, timeout=90):
    """Ask a running formation to stop, and wait for it to checkpoint.

    Called from the SIGINT handler and at shutdown. The worker only notices a
    cancellation between LLM calls, so this waits rather than returning
    immediately -- exiting while the engine is mid-memory is exactly how a run
    used to end up with no record of where it got to.
    """
    with RUN.lock:
        busy = RUN.status in ("running", "preparing", "stopping")
        mon = RUN.monitor
        thread = RUN.thread
    if not busy or not thread or not thread.is_alive():
        return False
    print(f"\n{reason}\n")
    if mon:
        print(f"Saving checkpoint… {mon.character_id}: {mon.done}/{mon.total} memories "
              f"safely completed.")
    else:
        print("Saving checkpoint…")
    RUN.stop_flag.set()
    with RUN.lock:
        if RUN.status in ("running", "preparing"):
            RUN.status = "stopping"
    deadline = time.time() + timeout
    while thread.is_alive() and time.time() < deadline:
        thread.join(timeout=1.0)
    with RUN.lock:
        mon = RUN.monitor
        done = RUN.stopped_at_memory
    if done is None and mon:
        done = mon.done
    if done is not None:
        print(f"\n{RUN.character}: {done} memories safely completed.")
        print(f"Next run will resume from memory {done + 1}.")
    elif thread.is_alive():
        print("\nWorker did not stop in time. The last checkpoint on disk is still "
              "valid — nothing written since is trusted.")
    return True


def _install_sigint(srv):
    """Ctrl+C stops the run cleanly the first time, and the server the second."""
    state = {"count": 0}

    def handler(signum, frame):
        state["count"] += 1
        if state["count"] > 1:
            print("\nSecond interrupt — exiting now.")
            os._exit(130)
        stopped = _drain_worker("Interrupt received.")
        if not stopped:
            print("\nstopped")
        else:
            print("\nStopping server.")
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    require_heart_bench()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"RUN HEART-BENCH  \u2192  {url}    (ctrl-c to stop)")
    cat = snapshots.catalogue()
    for c in cat:
        f = c.get("formation") or {}
        if c["prepared"]:
            print(f"  prepared: {c['character']}  {c['prepared_memories']}/"
                  f"{c['total_memories']} memories  \u00b7  {c['questions']} questions"
                  + ("" if c["full_protocol"] else "   [development depth]"))
        if f.get("resumable"):
            print(f"  RESUMABLE: {c['character']}  checkpoint at {f['done']}/{f['total']} "
                  f"memories \u2014 Full Protocol will resume from {f['resume_from']} "
                  f"(saved {f.get('updated_at')})")
        elif f.get("state") in ("corrupt", "stale"):
            print(f"  checkpoint for {c['character']} cannot be resumed: {f.get('detail')}")
    if not any(c["prepared"] for c in cat):
        print("  no prepared characters yet \u2014 Full Protocol Benchmark will form one first")
    prepared_ids = [c["character"] for c in cat if c["prepared"]]
    if prepared_ids:
        print(f"  Personality Fidelity (does MindForm's formed state match HEART's hidden "
              f"target?): python -m bench.heart.fidelity --character {prepared_ids[0]}")
    if args.open:
        webbrowser.open(url)
    _install_sigint(srv)
    try:
        srv.serve_forever()
    finally:
        _drain_worker("Shutting down.")


if __name__ == "__main__":
    main()
