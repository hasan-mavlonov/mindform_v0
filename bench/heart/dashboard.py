"""Local read-only dashboard for a HEART-Bench run.

    python -m bench.heart.dashboard --run <run_id>

Stdlib only, matching the project's own zero-dependency cockpit. It reads
``events.jsonl`` and nothing else -- it never imports the engine, never loads a
character, and never calls an LLM. Safe to open while a run is in flight; the
page repolls every few seconds.

Withheld-label discipline: HEART's ground-truth Big Five is served only once a
character's ingestion has finished, and the page labels it post-hoc analysis.
Per-question ground truth is only ever read from ``graded`` events, which the
runner writes after the answer is committed.
"""

import argparse
import json
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from bench.heart.config import RESULTS_ROOT

ARM_ORDER = ["naive_rag", "mindform_d1", "mindform_d2"]
ARM_LABEL = {"naive_rag": "Naive RAG (A)", "mindform_d1": "MindForm D1",
             "mindform_d2": "MindForm D2"}


def load_events(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def build_payload(run_dir, run_id):
    ev = load_events(run_dir)
    start = next((e for e in ev if e["event"] == "run_start"), {})
    ingest_steps = [e for e in ev if e["event"] == "ingest_step"]
    ingest_done = next((e for e in ev if e["event"] == "ingest_done"), None)
    committed = [e for e in ev if e["event"] == "answer_committed"]
    graded = [e for e in ev if e["event"] == "graded"]
    invalid = next((e for e in ev if e["event"] in ("run_invalid", "leak_detected")), None)
    ended = next((e for e in ev if e["event"] == "run_end"), None)

    by_q = {(g["arm"], g["question_id"], g.get("repeat", 0)): g for g in graded}

    # cumulative accuracy per arm, in grading order
    series = {a: [] for a in ARM_ORDER}
    tally = {a: [0, 0] for a in ARM_ORDER}
    for g in graded:
        a = g["arm"]
        if a not in tally:
            continue
        tally[a][1] += 1
        if g.get("correct"):
            tally[a][0] += 1
        series[a].append({"n": tally[a][1],
                          "acc": tally[a][0] / tally[a][1],
                          "question_id": g["question_id"],
                          "correct": bool(g.get("correct"))})

    # cost / perf, split by phase
    def _sum(rows, key):
        return sum(r.get(key) or 0 for r in rows)

    perf = {
        "ingest_calls_est": len(ingest_steps) * 5,   # 5 LLM calls per memory
        "ingest_steps": len(ingest_steps),
        "ingest_latency_ms_mean": (
            round(_sum(ingest_steps, "latency_ms") / len(ingest_steps), 1)
            if ingest_steps else None),
        "ingest_seconds": (ingest_done or {}).get("seconds"),
        "answer_calls": len(committed),
        "answer_input_tokens": _sum(committed, "input_tokens"),
        "answer_output_tokens": _sum(committed, "output_tokens"),
        "answer_latency_ms_mean": (
            round(_sum(committed, "latency_ms") / len(committed), 1) if committed else None),
        "answer_cost_usd": (round(_sum(committed, "cost_usd"), 4)
                            if any(c.get("cost_usd") is not None for c in committed) else None),
    }

    # trait trajectory over ingestion (from logged state; no LLM re-run)
    traj = [{"i": s.get("chrono_position", i), **s.get("traits_after", {})}
            for i, s in enumerate(ingest_steps)]
    # thin for the browser: at most ~600 points
    if len(traj) > 600:
        step = len(traj) // 600 + 1
        traj = traj[::step] + [traj[-1]]

    latest = committed[-1] if committed else None
    latest_grade = None
    if latest:
        latest_grade = by_q.get((latest["arm"], latest["question_id"], latest.get("repeat", 0)))

    integrity = {
        "leak_failures": sum(1 for g in graded if not g.get("leak_check_ok", True))
                         + (1 if (invalid or {}).get("event") == "leak_detected" else 0),
        "state_mutations": sum(1 for g in graded if g.get("state_mutated")),
        "invalid": (invalid or {}).get("reason") or (invalid or {}).get("detail"),
        "snapshot_id": (ingest_done or {}).get("snapshot_id"),
    }

    # post-hoc only: ground-truth Big Five once ingestion is complete
    withheld = None
    if ingest_done:
        try:
            from bench.heart import heartdata
            chars = heartdata.load_characters()
            cid = start.get("character")
            if cid in chars:
                withheld = heartdata.withheld_big_five(chars[cid])
        except Exception:
            withheld = None

    return {
        "run_id": run_id,
        "start": start,
        "ended": bool(ended),
        "ingest_done": bool(ingest_done),
        "ingest_total": start.get("n_memories"),
        "ingest_seen": len(ingest_steps),
        "traits_final": (ingest_done or {}).get("traits_final"),
        "elapsed_s": (ev[-1]["elapsed_s"] if ev else 0),
        "tallies": {a: tally[a] for a in ARM_ORDER},
        "series": series,
        "perf": perf,
        "trajectory": traj,
        "integrity": integrity,
        "withheld_big_five_posthoc": withheld,
        "latest": ({
            "arm": latest["arm"],
            "question_id": latest["question_id"],
            "scenario": latest["scenario"],
            "trigger_event": latest["trigger_event"],
            "options": latest["options"],
            "memories": latest["memories_retrieved"],
            "retrieval": latest["retrieval"],
            "state": latest["mindform_state"],
            "selected": latest["selected_answer"],
            "prompt_sha256": latest["prompt_sha256"],
            "model": latest["model"],
            "input_tokens": latest["input_tokens"],
            "output_tokens": latest["output_tokens"],
            "latency_ms": latest["latency_ms"],
            "ground_truth": (latest_grade or {}).get("ground_truth"),
            "correct": (latest_grade or {}).get("correct"),
        } if latest else None),
        "question_matrix": _matrix(graded),
    }


def _matrix(graded):
    qs, rows = [], {}
    for g in graded:
        qid = g["question_id"]
        if qid not in rows:
            rows[qid] = {}
            qs.append(qid)
        rows[qid].setdefault(g["arm"], []).append(
            {"sel": g.get("selected_answer"), "gt": g.get("ground_truth"),
             "ok": bool(g.get("correct")), "repeat": g.get("repeat", 0)})
    return [{"question_id": q, "arms": rows[q]} for q in qs]


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HEART-Bench run</title>
<style>
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --surface-2:#f2f1ee; --card:#ffffff;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#83817a;
  --rule:rgba(11,11,11,.12);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --good:#1a7f4b; --bad:#b3261e; --warn:#9a6206;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#111110; --card:#232322;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --rule:rgba(255,255,255,.15);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
  --good:#4ec98a; --bad:#ff8a7a; --warn:#f0b458;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-2);color:var(--text-primary);font-family:var(--sans);
     font-size:14px;line-height:1.5;padding:20px}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:16px}
h1{font-size:19px;margin:0;font-weight:600;letter-spacing:-.01em}
h2{font-size:13px;margin:0 0 10px;font-weight:600;text-transform:uppercase;
   letter-spacing:.08em;color:var(--text-secondary)}
.card{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--rule);border-radius:9px;padding:11px 13px}
.tile .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--text-muted)}
.tile .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:3px}
.tile .v.sm{font-size:14px;font-weight:500}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:840px){.row{grid-template-columns:1fr}}
.mono{font-family:var(--mono);font-size:12px}
.ok{color:var(--good)} .bad{color:var(--bad)} .warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--rule)}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted);font-weight:600}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
details{border:1px solid var(--rule);border-radius:8px;padding:9px 12px;background:var(--surface-1)}
summary{cursor:pointer;font-weight:500;font-size:12.5px}
.mem{font-family:var(--mono);font-size:11.5px;padding:4px 0;border-bottom:1px solid var(--rule);
     display:grid;grid-template-columns:34px 62px 1fr;gap:8px;align-items:baseline}
.opt{padding:6px 0;border-bottom:1px solid var(--rule);font-size:13px}
.opt b{display:inline-block;width:18px}
.pill{display:inline-block;font-family:var(--mono);font-size:10.5px;padding:2px 7px;
      border-radius:4px;background:var(--surface-2);border:1px solid var(--rule)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin-bottom:6px}
.legend span{display:inline-flex;align-items:center;gap:5px}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.muted{color:var(--text-muted)}
.banner{border-radius:9px;padding:10px 14px;font-weight:600;font-size:13px}
.banner.bad{background:rgba(179,38,30,.12);color:var(--bad);border:1px solid var(--bad)}
</style></head><body>
<div class="wrap" id="root"><div class="card">loading…</div></div>
<script>
const ARMS=[["naive_rag","Naive RAG (A)","var(--s1)"],["mindform_d1","MindForm D1","var(--s2)"],
            ["mindform_d2","MindForm D2","var(--s3)"]];
const OCEAN=[["O","Openness","var(--s1)"],["C","Conscientiousness","var(--s2)"],
             ["E","Extraversion","var(--s3)"],["A","Agreeableness","var(--s4)"],
             ["N","Neuroticism","var(--s5)"]];
const esc=s=>String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const pct=(a,b)=>b?(100*a/b).toFixed(1)+"%":"—";

function tile(k,v,cls){return `<div class="tile"><div class="k">${k}</div><div class="v ${cls||''}">${v}</div></div>`}

function accBars(d){
  const W=440,H=170,pad=34,bw=64,gap=44;
  let bars="",i=0;
  for(const [key,label,col] of ARMS){
    const [c,n]=d.tallies[key]||[0,0]; const acc=n?c/n:0;
    const x=pad+i*(bw+gap), h=(H-pad-26)*acc, y=H-26-h;
    bars+=`<rect x="${x}" y="${y}" width="${bw}" height="${Math.max(h,1)}" rx="4" fill="${col}"/>`;
    bars+=`<text x="${x+bw/2}" y="${y-7}" text-anchor="middle" font-size="12" font-weight="600"
            fill="var(--text-primary)">${pct(c,n)}</text>`;
    bars+=`<text x="${x+bw/2}" y="${H-11}" text-anchor="middle" font-size="10.5"
            fill="var(--text-secondary)">${label}</text>`;
    bars+=`<text x="${x+bw/2}" y="${H-1}" text-anchor="middle" font-size="10"
            fill="var(--text-muted)" font-family="var(--mono)">${c}/${n}</text>`;
    i++;
  }
  const cy=H-26-(H-pad-26)*0.25;
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
    <line x1="${pad-8}" y1="${cy}" x2="${W-8}" y2="${cy}" stroke="var(--text-muted)"
      stroke-dasharray="4 3" stroke-width="1"/>
    <text x="${W-10}" y="${cy-5}" text-anchor="end" font-size="10" fill="var(--text-muted)">chance 25%</text>
    <line x1="${pad-8}" y1="${H-26}" x2="${W-8}" y2="${H-26}" stroke="var(--rule)"/>
    ${bars}</svg>`;
}

function accLine(d){
  const W=440,H=170,l=34,r=10,t=12,b=26;
  const maxN=Math.max(1,...ARMS.map(([k])=>(d.series[k]||[]).length));
  const X=n=>l+(W-l-r)*((n-1)/Math.max(1,maxN-1));
  const Y=a=>t+(H-t-b)*(1-a);
  let g="";
  for(const gv of [0,.25,.5,.75,1]){
    g+=`<line x1="${l}" y1="${Y(gv)}" x2="${W-r}" y2="${Y(gv)}" stroke="var(--rule)" stroke-width="1"/>
        <text x="${l-6}" y="${Y(gv)+3}" text-anchor="end" font-size="9.5"
          fill="var(--text-muted)" font-family="var(--mono)">${(gv*100)|0}</text>`;
  }
  let paths="";
  for(const [key,label,col] of ARMS){
    const s=d.series[key]||[]; if(!s.length) continue;
    const pts=s.map(p=>`${X(p.n)},${Y(p.acc)}`).join(" ");
    paths+=`<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2"
             stroke-linejoin="round" stroke-linecap="round"/>`;
    const last=s[s.length-1];
    paths+=`<circle cx="${X(last.n)}" cy="${Y(last.acc)}" r="4" fill="${col}"
             stroke="var(--card)" stroke-width="2"/>`;
    paths+=`<text x="${X(last.n)-6}" y="${Y(last.acc)-9}" text-anchor="end" font-size="10"
             font-weight="600" fill="var(--text-primary)">${label}</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">${g}${paths}
    <text x="${W-r}" y="${H-6}" text-anchor="end" font-size="9.5" fill="var(--text-muted)">questions graded →</text></svg>`;
}

function traj(d){
  const s=d.trajectory||[]; const W=880,H=230,l=36,r=54,t=14,b=28;
  if(!s.length) return '<p class="muted">No ingestion steps logged yet.</p>';
  const n=s.length;
  const X=i=>l+(W-l-r)*(i/Math.max(1,n-1));
  const Y=v=>t+(H-t-b)*(1-(v+1)/2);
  let g=`<line x1="${l}" y1="${Y(0)}" x2="${W-r}" y2="${Y(0)}" stroke="var(--rule)" stroke-width="1.5"/>`;
  for(const gv of [-1,-.5,.5,1]){
    g+=`<line x1="${l}" y1="${Y(gv)}" x2="${W-r}" y2="${Y(gv)}" stroke="var(--rule)"
         stroke-width="1" stroke-dasharray="3 4"/>
        <text x="${l-6}" y="${Y(gv)+3}" text-anchor="end" font-size="9.5"
         fill="var(--text-muted)" font-family="var(--mono)">${gv>0?'+':''}${gv}</text>`;
  }
  g+=`<text x="${l-6}" y="${Y(0)+3}" text-anchor="end" font-size="9.5"
       fill="var(--text-muted)" font-family="var(--mono)">0</text>`;
  let paths="";
  // End labels de-collide: traits cluster near 0 early in a run, so lay the
  // labels out at a minimum vertical spacing instead of letting them overprint.
  const ends=OCEAN.map(([k,label,col])=>({k,col,v:s[s.length-1][k]||0}));
  ends.sort((a,b)=>a.v-b.v);
  const GAP=12; let placed=[];
  for(const e of ends){ let y=Y(e.v);
    while(placed.some(q=>Math.abs(q-y)<GAP)) y-=GAP/2;
    placed.push(y); e.labelY=y; }
  for(const [k,label,col] of OCEAN){
    const pts=s.map((p,i)=>`${X(i)},${Y(p[k]||0)}`).join(" ");
    paths+=`<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="2"
             stroke-linejoin="round"/>`;
  }
  for(const e of ends){
    const cy=Y(e.v);
    if(Math.abs(e.labelY-cy)>1)
      paths+=`<line x1="${X(n-1)+3}" y1="${cy}" x2="${X(n-1)+7}" y2="${e.labelY-3}"
               stroke="${e.col}" stroke-width="1" opacity=".55"/>`;
    paths+=`<circle cx="${X(n-1)}" cy="${cy}" r="3.5" fill="${e.col}"
             stroke="var(--card)" stroke-width="1.5"/>
            <text x="${X(n-1)+9}" y="${e.labelY}" font-size="10" font-weight="600"
             fill="${e.col}">${e.k} ${e.v>=0?'+':''}${e.v.toFixed(2)}</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">${g}${paths}
    <text x="${l}" y="${H-6}" font-size="9.5" fill="var(--text-muted)">experience 1</text>
    <text x="${W-r}" y="${H-6}" text-anchor="end" font-size="9.5" fill="var(--text-muted)">experience ${d.ingest_seen}</text></svg>`;
}

function stateBars(st){
  if(!st) return '<p class="muted">This arm does not expose MindForm state (naive RAG / D1).</p>';
  let h='<div style="display:flex;flex-direction:column;gap:5px">';
  for(const [k,label,col] of OCEAN){
    const v=st.traits[k]||0, w=Math.abs(v)*50;
    h+=`<div style="display:grid;grid-template-columns:120px 1fr 52px;gap:8px;align-items:center">
      <span style="font-size:12px">${label}</span>
      <span style="position:relative;height:13px;background:var(--surface-2);border-radius:3px">
        <span style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--rule)"></span>
        <span style="position:absolute;top:1px;bottom:1px;border-radius:2px;background:${col};
          ${v>=0?`left:50%;width:${w}%`:`right:50%;width:${w}%`}"></span></span>
      <span class="mono" style="text-align:right">${v>=0?'+':''}${v.toFixed(2)}</span></div>`;
  }
  h+='</div>';
  const bits=[];
  if(st.top_values&&st.top_values.length) bits.push("Values: "+st.top_values.map(v=>`${v[0]} ${v[1]>=0?'+':''}${v[1]}`).join(", "));
  if(st.top_moral&&st.top_moral.length) bits.push("Moral: "+st.top_moral.map(v=>`${v[0]} ${v[1]}`).join(", "));
  if(st.top_drives&&st.top_drives.length) bits.push("Needs: "+st.top_drives.map(v=>`${v[0]} ${v[1]}`).join(", "));
  if(st.beliefs&&st.beliefs.length) bits.push("Beliefs: "+st.beliefs.slice(0,3).join("; "));
  if(bits.length) h+=`<div class="mono muted" style="margin-top:9px;font-size:11.5px">${esc(bits.join(" · "))}</div>`;
  return h;
}

function render(d){
  const st=d.start||{}, integ=d.integrity||{}, perf=d.perf||{};
  const leakOK=!integ.leak_failures, snapOK=!integ.state_mutations;
  const stage = d.ended ? "complete" : (d.ingest_done ? "answering" : "ingesting");
  let h="";

  if(integ.invalid) h+=`<div class="banner bad">RUN INVALID — ${esc(integ.invalid)}</div>`;

  h+=`<div><h1>HEART-Bench × MindForm · <span class="mono">${esc(d.run_id)}</span></h1>
    <div class="muted" style="font-size:12.5px">${esc(st.character||'?')} ·
      model <span class="mono">${esc(st.model||'?')}</span> ·
      temp ${st.temperature} · top_k ${st.top_k} ·
      embedder <span class="mono">${esc(st.embedder||'?')}</span></div></div>`;

  h+=`<div class="tiles">
    ${tile("Stage",stage,"sm")}
    ${tile("Ingestion",`${d.ingest_seen}/${d.ingest_total??'?'}`)}
    ${tile("Questions graded",ARMS.reduce((a,[k])=>a+(d.tallies[k]||[0,0])[1],0))}
    ${tile("Elapsed",(d.elapsed_s/60).toFixed(1)+" min")}
    ${tile("Answer tokens",(perf.answer_input_tokens||0).toLocaleString()+" in")}
    ${tile("Est. cost",perf.answer_cost_usd==null?'<span class="muted">rates unset</span>':'$'+perf.answer_cost_usd,"sm")}
    ${tile("Leak check",leakOK?'<span class="ok">PASS</span>':'<span class="bad">FAIL</span>',"sm")}
    ${tile("Snapshot integrity",snapOK?'<span class="ok">STABLE</span>':'<span class="bad">MUTATED</span>',"sm")}
  </div>`;

  h+=`<div class="row">
    <div class="card"><h2>Accuracy by arm</h2>
      <div class="muted" style="font-size:11.5px;margin-bottom:6px">Smoke-test scale — not a
        significance claim.</div>${accBars(d)}</div>
    <div class="card"><h2>Cumulative accuracy</h2>${accLine(d)}</div></div>`;

  h+=`<div class="card"><h2>Trait formation over ingestion</h2>
    <div class="legend">${OCEAN.map(([k,l,c])=>`<span><i class="sw" style="background:${c}"></i>${l}</span>`).join("")}</div>
    ${traj(d)}
    <div class="muted" style="font-size:11.5px;margin-top:6px">Logged engine state — no model
      was called to draw this.</div></div>`;

  const L=d.latest;
  if(L){
    h+=`<div class="card"><h2>Current question · ${esc(L.question_id)} · ${esc(L.arm)}</h2>
      <div style="font-weight:600;margin-bottom:3px">${esc(L.scenario.name)}</div>
      <div class="muted" style="font-size:12.5px;margin-bottom:9px">${esc((L.scenario.context_text||"").slice(0,320))}…</div>
      <div class="mono" style="font-size:11.5px;margin-bottom:9px">Trigger — ${esc((L.trigger_event||{}).sender||'')}:
        ${esc(((L.trigger_event||{}).message_content||"").slice(0,180))}…</div>
      ${L.options.map(o=>`<div class="opt"><b>${o.label}</b>${esc(o.content.slice(0,220))}…</div>`).join("")}
      <div style="margin-top:11px;display:flex;gap:9px;align-items:center;flex-wrap:wrap">
        <span class="pill">chose ${esc(L.selected??'—')}</span>
        ${L.ground_truth?`<span class="pill">truth ${esc(L.ground_truth)}</span>
          <span class="pill ${L.correct?'ok':'bad'}">${L.correct?'CORRECT':'INCORRECT'}</span>`
          :'<span class="pill muted">truth withheld until commit</span>'}
        <span class="pill">${L.latency_ms} ms</span>
        <span class="pill">${L.input_tokens}→${L.output_tokens} tok</span>
        <span class="pill mono">sha ${esc((L.prompt_sha256||'').slice(0,10))}</span>
      </div></div>`;

    h+=`<div class="row">
      <div class="card"><h2>MindForm state used</h2>${stateBars(L.state)}</div>
      <div class="card"><h2>Retrieved memories</h2>
        <div class="muted" style="font-size:11.5px;margin-bottom:7px">
          ${esc(L.retrieval.retriever)} · k=${L.retrieval.k} ·
          drive_bias=${L.retrieval.drive_bias}</div>
        <details><summary>${L.memories.length} memories reached the model</summary>
          <div style="max-height:280px;overflow:auto;margin-top:8px">
          ${L.memories.map((m,i)=>`<div class="mem"><span class="muted">${i+1}</span>
            <span>${(m.score??0).toFixed(3)}</span>
            <span>${esc(m.anon_id)} · ${esc((m.text||"").slice(0,100))}…</span></div>`).join("")}
          </div></details></div></div>`;
  }

  const M=d.question_matrix||[];
  if(M.length){
    h+=`<div class="card"><h2>Per-question comparison</h2><table>
      <thead><tr><th>Question</th>${ARMS.map(([k,l])=>`<th>${l}</th>`).join("")}<th>truth</th></tr></thead>
      <tbody>${M.map(row=>{
        let gt="";
        const cells=ARMS.map(([k])=>{
          const rs=row.arms[k]; if(!rs) return '<td class="muted">—</td>';
          gt=rs[0].gt;
          return `<td class="num">${rs.map(r=>`${r.sel??'∅'} ${r.ok?'<span class="ok">✓</span>':'<span class="bad">✗</span>'}`).join(" ")}</td>`;
        }).join("");
        return `<tr><td class="mono">${esc(row.question_id.replace(/^Q_CHAR_\d+_/,''))}</td>${cells}<td class="num">${esc(gt)}</td></tr>`;
      }).join("")}</tbody></table></div>`;
  }

  h+=`<div class="card"><h2>Cost &amp; performance</h2><table>
    <thead><tr><th>Phase</th><th>Calls</th><th>Input tok</th><th>Output tok</th><th>Mean latency</th></tr></thead>
    <tbody>
      <tr><td>Ingestion (5 LLM calls/memory)</td><td class="num">${perf.ingest_calls_est}</td>
          <td class="num muted">not returned by engine path</td><td class="num muted">—</td>
          <td class="num">${perf.ingest_latency_ms_mean??'—'} ms/mem</td></tr>
      <tr><td>Embedding (local MiniLM)</td><td class="num">${perf.ingest_steps}</td>
          <td class="num">0</td><td class="num">0</td><td class="num muted">—</td></tr>
      <tr><td>Retrieval (local cosine)</td><td class="num">0</td><td class="num">0</td>
          <td class="num">0</td><td class="num muted">—</td></tr>
      <tr><td>MCQ answering</td><td class="num">${perf.answer_calls}</td>
          <td class="num">${(perf.answer_input_tokens||0).toLocaleString()}</td>
          <td class="num">${(perf.answer_output_tokens||0).toLocaleString()}</td>
          <td class="num">${perf.answer_latency_ms_mean??'—'} ms</td></tr>
    </tbody></table></div>`;

  if(d.withheld_big_five_posthoc && d.ingest_done){
    const w=d.withheld_big_five_posthoc, f=d.traits_final||{};
    const map={openness:"O",conscientiousness:"C",extraversion:"E",agreeableness:"A",neuroticism:"N"};
    h+=`<div class="card"><h2>Post-hoc: formed vs withheld Big Five</h2>
      <div class="muted" style="font-size:11.5px;margin-bottom:8px">Revealed only after ingestion
        finished. This is our own analysis, not a HEART metric, and it was never in any prompt.</div>
      <table><thead><tr><th>Trait</th><th>HEART (0–1)</th><th>HEART (−1…+1)</th>
        <th>MindForm formed</th><th>Same direction</th></tr></thead><tbody>
      ${Object.entries(map).map(([long,k])=>{
        const gt=w[long], gtc=gt==null?null:(gt*2-1), mf=f[k];
        // |withheld| < 0.15 is the scale midpoint -- no direction to agree with.
        const scored=(gtc!=null&&Math.abs(gtc)>=0.15);
        const agree=scored&&mf!=null&&((gtc>=0)===(mf>=0));
        return `<tr><td>${long}</td><td class="num">${gt??'—'}</td>
          <td class="num">${gtc==null?'—':(gtc>=0?'+':'')+gtc.toFixed(2)}</td>
          <td class="num">${mf==null?'—':(mf>=0?'+':'')+mf.toFixed(3)}</td>
          <td>${!scored?'<span class="muted">n/a — midpoint</span>'
                :(agree?'<span class="ok">yes</span>':'<span class="bad">no</span>')}</td></tr>`;
      }).join("")}</tbody></table></div>`;
  }

  document.getElementById("root").innerHTML=h;
}

async function poll(){
  try{ const r=await fetch("/api/data",{cache:"no-store"}); render(await r.json()); }
  catch(e){ /* keep the last good frame */ }
}
poll(); setInterval(poll,4000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    run_dir = None
    run_id = None

    def do_GET(self):
        if self.path.startswith("/api/data"):
            body = json.dumps(build_payload(self.run_dir, self.run_id)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--snapshot", metavar="PATH",
                    help="write a static HTML+JSON pair instead of serving")
    args = ap.parse_args()

    run_dir = os.path.join(RESULTS_ROOT, args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"no such run: {run_dir}")

    if args.snapshot:
        payload = build_payload(run_dir, args.run)
        html = PAGE.replace('poll(); setInterval(poll,4000);',
                            f"render({json.dumps(payload)});")
        with open(args.snapshot, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"wrote {args.snapshot}")
        return

    Handler.run_dir, Handler.run_id = run_dir, args.run
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"HEART-Bench dashboard for {args.run} -> {url}   (ctrl-c to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
