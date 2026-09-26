"""RUN PERSONALITY TEST -- a small, watchable BFI-2-S administration.

    python -m bench.personality.live            # then open http://127.0.0.1:8501

Pick a character, click Run, and watch the frozen MindForm character answer
a real personality inventory about itself -- one item at a time, self-report,
no HEART labels anywhere near the prompt. When it finishes: domain bars,
optional facet drill-down, and (behind its own "research/details" toggle) the
three-way comparison against MindForm's own internal state and HEART's hidden
target -- never presented as if they were one number.

This is a completely separate testing experience from bench/heart/live.py:
different port, different server, different question ("what personality did
MindForm form?" vs "does it behave like that person?"). Run either
independently, or both -- see bench/tests.py for a combined landing page.
"""

import argparse
import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bench.heart import snapshots
from bench.heart.config import require_heart_bench
from bench.personality import report, runner
from bench.personality.bfi2 import INSTRUMENT


class Run:
    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.reset()

    def reset(self):
        with self.lock:
            self.status = "idle"          # idle|running|done|error
            self.error = None
            self.character = None
            self.tier = None
            self.item_index = 0
            self.total_items = len(INSTRUMENT.items)
            self.current_item = None
            self.answered = []            # [{number, text, domain, facet, rating}]
            self.result = None
            self.comparison = None
            self.started_at = None
            self.finished_at = None

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def payload(self):
        with self.lock:
            elapsed = ((self.finished_at or time.time()) - self.started_at) \
                if self.started_at else 0
            return {
                "status": self.status, "error": self.error,
                "character": self.character, "tier": self.tier,
                "instrument": {"id": INSTRUMENT.id, "name": INSTRUMENT.name,
                              "full_name": INSTRUMENT.full_name,
                              "license_notice": INSTRUMENT.license_notice},
                "item_index": self.item_index, "total_items": self.total_items,
                "current_item": self.current_item,
                "answered": self.answered,
                "result": self.result, "comparison": self.comparison,
                "elapsed_s": round(elapsed, 1),
            }


RUN = Run()


def _worker(character, tier):
    try:
        RUN.set(status="running", character=character, tier=tier,
                item_index=0, answered=[], result=None, comparison=None,
                error=None, started_at=time.time())

        def on_item(i, total, item, rating):
            with RUN.lock:
                RUN.item_index = i
                RUN.current_item = {"number": item.number, "text": item.text,
                                    "domain": item.domain, "facet": item.facet}
                RUN.answered.append({"number": item.number, "text": item.text,
                                     "domain": item.domain, "facet": item.facet,
                                     "reverse_keyed": item.reverse_keyed,
                                     "rating": rating})

        result = runner.administer(character, INSTRUMENT,
                                   tier=(None if tier == "auto" else tier),
                                   on_item=on_item)
        comparison = report.build_comparison(result)
        with RUN.lock:
            RUN.result = result
            RUN.comparison = comparison
            RUN.status = "done"
            RUN.finished_at = time.time()
    except runner.NoSnapshot as exc:
        RUN.set(status="error", error=str(exc), finished_at=time.time())
    except Exception as exc:
        RUN.set(status="error", error=f"{type(exc).__name__}: {exc}",
                finished_at=time.time())
        traceback.print_exc()


def start_run(character, tier):
    with RUN.lock:
        if RUN.status == "running":
            return False, "a run is already in progress"
    RUN.reset()
    RUN.set(status="running")
    t = threading.Thread(target=_worker, args=(character, tier), daemon=True)
    RUN.thread = t
    t.start()
    return True, "started"


def _setup_payload():
    cat = snapshots.catalogue()
    return {"instrument": {"id": INSTRUMENT.id, "name": INSTRUMENT.name,
                           "full_name": INSTRUMENT.full_name,
                           "domain_order": list(INSTRUMENT.domain_order),
                           "license_notice": INSTRUMENT.license_notice},
            "characters": cat}


PAGE = r"""<!doctype html><meta charset="utf-8">
<title>Personality Test</title>
<style>
:root{color-scheme:light;--bg:#f2f1ee;--card:#fff;--ink:#0b0b0b;--sec:#52514e;--mut:#83817a;
 --rule:rgba(11,11,11,.12);--accent:#2a78d6;}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --bg:#111110;--card:#232322;--ink:#fff;--sec:#c3c2b7;--mut:#8f8e86;--rule:rgba(255,255,255,.15);
 --accent:#5b9eec;}}
body{margin:0;padding:22px;background:var(--bg);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,sans-serif}
.wrap{max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:14px}
h1{font-size:20px;margin:0} .sub{color:var(--sec);font-size:13px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:16px 18px}
select,button{font:inherit;padding:8px 10px;border-radius:8px;border:1px solid var(--rule);
 background:var(--card);color:var(--ink)}
button.primary{background:var(--accent);color:#fff;border:none;cursor:pointer;font-weight:600}
button.primary:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.bar{height:9px;background:rgba(128,128,128,.15);border-radius:99px;overflow:hidden}
.bar span{display:block;height:100%;background:var(--accent);transition:width .25s}
.tr{display:flex;align-items:center;gap:10px;padding:6px 0}
.tl{flex:1.6;font-size:13px} .num{font:12px/1 ui-monospace,monospace;width:70px;text-align:right}
.track{flex:3;height:11px;background:rgba(128,128,128,.15);border-radius:99px;overflow:hidden}
.fill{display:block;height:100%;background:var(--accent);border-radius:99px}
.small{font-size:12.5px;color:var(--mut)}
.notice{font-size:11.5px;color:var(--mut);background:rgba(128,128,128,.08);border-radius:8px;
 padding:8px 10px;margin-top:8px}
.feed{max-height:160px;overflow-y:auto;font:12px/1.7 ui-monospace,monospace;color:var(--sec);
 background:rgba(128,128,128,.06);border-radius:8px;padding:8px 10px;margin-top:8px}
details summary{cursor:pointer;font-size:13px;color:var(--sec)}
</style>
<div class="wrap" id="root"></div>
<script>
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let SETUP=null, STATE=null;
async function setup(){ SETUP=await (await fetch("/api/setup")).json(); }
// BUG (fixed here): render() replaces #root's whole innerHTML, which destroys
// every child node -- including a native <details> a viewer just opened. With
// poll() calling render() unconditionally every 600ms, a <details> could never
// stay open past one polling tick: nothing about the *data* had changed, but
// the DOM node holding "open" was torn down and rebuilt from a template that
// never encodes that transient DOM state, so it always came back closed. That
// reads as "the dropdown opens and immediately closes."
// Fix: skip the repaint entirely when the state is byte-for-byte the same as
// last time. This is not a timeout or a special-case for <details> -- once a
// run finishes, /api/state is genuinely static (elapsed_s freezes to
// finished_at, not time.time()), so there is nothing left to legitimately
// redraw, and any DOM state the viewer created (an open <details>, a scroll
// position) is simply left alone. Real changes -- a run progressing, a new
// run starting -- still repaint immediately, once, correctly.
let LAST_STATE_JSON=null;
async function poll(){
  try{
    const j=await (await fetch("/api/state",{cache:"no-store"})).json();
    const s=JSON.stringify(j);
    if(s===LAST_STATE_JSON) return;
    LAST_STATE_JSON=s; STATE=j; render();
  }catch(e){}
}
async function start(){
  const character=document.getElementById("char").value;
  const tier=document.getElementById("tier").value;
  const r=await fetch("/api/start",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({character,tier})});
  if(!r.ok){ const j=await r.json(); alert(j.message); }
  poll();
}
function controls(){
  const running=STATE?.status==="running";
  const chars=SETUP?.characters||[];
  const opts=chars.flatMap(c=>{
    const rows=[];
    if(c.dev_prepared) rows.push(`<option value="${c.character}::dev">${c.character} — dev (${c.dev_memories} mem)</option>`);
    if(c.full_prepared) rows.push(`<option value="${c.character}::full">${c.character} — full (${c.total_memories} mem)</option>`);
    return rows;
  }).join("");
  return `<div class="card">
    <div class="row">
      <select id="chartier" ${running?"disabled":""} onchange="onCharTier()">${opts}</select>
      <input type="hidden" id="char"><input type="hidden" id="tier">
      <button class="primary" onclick="start()" ${running?"disabled":""}>
        ${running?"Running…":"Run "+esc(SETUP?.instrument?.name||"Personality Test")}</button>
    </div>
    <p class="small" style="margin-top:8px">${esc(SETUP?.instrument?.full_name||"")} —
      MindForm's already-formed character answers a real personality questionnaire
      about itself. Nothing is retrained or re-formed, and HEART's hidden target
      personality is never shown to it.</p>
    ${licenseNotice(SETUP?.instrument?.license_notice)}
  </div>`;
}
function onCharTier(){
  const v=document.getElementById("chartier").value.split("::");
  document.getElementById("char").value=v[0];
  document.getElementById("tier").value=v[1];
}
// Licensing stays visible everywhere this test appears, but the long legal
// paragraph is one click away rather than always taking up space -- the short
// line is what an investor or partner actually needs to see by default.
function licenseNotice(full){
  if(!full) return "";
  return `<p class="notice">Internal research use only — not yet licensed for
    commercial use.
    <details style="display:inline"><summary style="display:inline;cursor:pointer;
      color:var(--accent)">Legal details ▾</summary>
      <div style="margin-top:6px">${esc(full)}</div></details></p>`;
}
function progress(){
  if(!STATE || STATE.status==="idle") return "";
  if(STATE.status==="error")
    return `<div class="card"><b>Error:</b> ${esc(STATE.error)}</div>`;
  const pct=100*STATE.item_index/Math.max(STATE.total_items,1);
  const cur=STATE.current_item;
  return `<div class="card">
    <div class="row" style="justify-content:space-between">
      <span>${STATE.item_index} / ${STATE.total_items} items</span>
      <span class="small">${STATE.elapsed_s}s</span></div>
    <div class="bar" style="margin-top:8px"><span style="width:${pct}%"></span></div>
    ${cur?`<p class="small" style="margin-top:8px">"${esc(cur.text)}" — ${esc(cur.domain)} /
      ${esc(cur.facet)}</p>`:""}
    <div class="feed">${STATE.answered.slice(-8).map(a=>
      `${a.number}. ${esc(a.text)} → ${a.rating}${a.reverse_keyed?" (reverse-keyed)":""}`
    ).join("<br>")}</div>
  </div>`;
}
function results(){
  const r=STATE?.result, c=STATE?.comparison;
  if(!r || STATE.status!=="done") return "";
  const scored=r.scored;
  const order=Object.keys(scored.domain_scores);
  const bars=order.map((d,i)=>{
    const pct=(scored.domain_scores[d]-1)/4*100;
    return `<div class="tr"><span class="tl">${esc(d)}</span>
      <span class="track"><span class="fill" style="width:${pct}%"></span></span>
      <span class="num">${scored.domain_scores[d].toFixed(2)} / 5</span></div>`;
  }).join("");
  const facetsByDomain={};
  r.items.forEach(it=>{ (facetsByDomain[it.domain]=facetsByDomain[it.domain]||new Set()).add(it.facet); });
  const facetHtml=order.map(d=>{
    const rows=[...facetsByDomain[d]].sort().map(f=>{
      const pct=(scored.facet_scores[f]-1)/4*100;
      return `<div class="tr" style="padding:3px 0"><span class="tl small">${esc(f)}</span>
        <span class="track" style="height:6px;flex:2.4"><span class="fill" style="width:${pct}%;opacity:.6"></span></span>
        <span class="num">${scored.facet_scores[f].toFixed(2)}</span></div>`;
    }).join("");
    return `<div style="margin-top:10px"><h3 style="font-size:12px;color:var(--sec);
      text-transform:uppercase;letter-spacing:.04em;margin:0 0 2px">${esc(d)}</h3>${rows}</div>`;
  }).join("");
  const compRows="OCEAN".split("").map(k=>{
    const row=c.rows[k];
    return `<div class="tr"><span class="tl">${esc(row.name)}</span>
      <span class="num">${row.internal.toFixed(3)}</span>
      <span class="num">${row.bfi2.toFixed(3)}</span>
      <span class="num">${row.hidden_target.toFixed(3)}</span>
      <span class="num">${row.bfi2_vs_hidden_pct.toFixed(0)}%</span></div>`;
  }).join("");
  return `<div class="card">
    <h1 style="font-size:16px">${esc(r.instrument.name)} PERSONALITY TEST</h1>
    <p class="sub">${esc(r.character)} (${esc(r.occupation)}) — ${esc(r.tier.toUpperCase())} tier</p>
    ${bars}
    <details style="margin-top:8px"><summary>View 15 facets ↓</summary>${facetHtml}
      <p class="notice">${esc(r.instrument.reliability_note)}</p></details>
    ${licenseNotice(r.instrument.license_notice)}
  </div>
  <details class="card">
    <summary><b>Research details ▾</b> — internal MindForm, self-report, and HEART's
      hidden target are three separate measurements, never averaged together.</summary>
    <p class="small" style="margin-top:8px">${esc(c.protocol_note)}</p>
    <div class="row small" style="justify-content:space-between;padding:4px 0">
      <span style="flex:1.6">trait</span><span style="width:70px;text-align:right">internal</span>
      <span style="width:70px;text-align:right">${esc(r.instrument.name.toLowerCase())}</span>
      <span style="width:70px;text-align:right">hidden</span>
      <span style="width:70px;text-align:right">match</span></div>
    ${compRows}
    <p class="notice">Overall: self-report vs hidden target
      <b>${c.summary.bfi2_vs_hidden_overall_pct}%</b> · internal vs hidden target
      <b>${c.summary.internal_vs_hidden_overall_pct}%</b> · internal vs self-report
      <b>${c.summary.internal_vs_bfi2_overall_pct}%</b></p>
    <p class="notice">${esc(c.ground_truth_disclosure)}</p>
  </details>`;
}
function render(){
  let h=`<h1>Personality Test</h1>
    <p class="sub">What personality did MindForm form? — MindForm takes the
    personality it formed and answers a standardized questionnaire about
    itself, in character.</p>`;
  h+=controls()+progress()+results();
  document.getElementById("root").innerHTML=h;
  const sel=document.getElementById("chartier");
  if(sel && !document.getElementById("char").value) onCharTier();
}
(async()=>{ await setup(); await poll(); setInterval(poll,600); })();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200, ctype="application/json"):
        body = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/setup"):
            return self._send(_setup_payload())
        if self.path.startswith("/api/state"):
            return self._send(RUN.payload())
        return self._send(PAGE.encode("utf-8"), ctype="text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if self.path.startswith("/api/start"):
            ok, msg = start_run(body.get("character", "CHAR_01"), body.get("tier", "auto"))
            return self._send({"ok": ok, "message": msg}, 200 if ok else 409)
        return self._send({"ok": False, "message": "unknown endpoint"}, 404)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    require_heart_bench()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"RUN PERSONALITY TEST  →  {url}    (ctrl-c to stop)")
    print(f"  instrument: {INSTRUMENT.name} ({INSTRUMENT.full_name})")
    print(f"  {INSTRUMENT.license_notice}")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
