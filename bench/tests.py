"""MINDFORM TESTS -- the landing page for both testing experiences.

    python -m bench.tests            # opens http://127.0.0.1:8000

Two completely separate testing experiences, never merged:

    PERSONALITY TEST  (BFI-2-S)   "What personality did MindForm form?"
    BEHAVIOR TEST     (HEART)     "Does the formed personality actually
                                   behave like the target person?"

This file starts both existing servers -- bench.heart.live's Handler on one
port and bench.personality.live's Handler on another -- inside this one
process, purely by reusing their already-existing, already-tested Handler
classes. It does not modify either module, does not share any state between
them, and does not know anything about what either one does internally; it
is a thin composition root, not a merge. Each sub-app also continues to work
exactly as before when launched on its own:

    python -m bench.heart.live
    python -m bench.personality.live

Clicking a card just navigates to that sub-app's own port; each runs
independently of the other from that point on.

The page below is fully static -- built once per request, no polling, no
client-side state machine -- so its native <details> "Learn more" sections
have nothing to fight: there's no rerender loop here to tear them down mid
click, unlike the two sub-apps (see their own module docstrings for the
dropdown-collapse bug fixed on the BFI-2-S page and the equivalent risk
avoided on HEART's page).
"""

import argparse
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STYLE = """
:root{color-scheme:light;--bg:#f2f1ee;--card:#fff;--ink:#0b0b0b;--sec:#52514e;--mut:#83817a;
 --rule:rgba(11,11,11,.12);--accent:#2a78d6;--accent2:#1baf7a;--notice-bg:rgba(128,128,128,.08);}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;
 --bg:#111110;--card:#232322;--ink:#fff;--sec:#c3c2b7;--mut:#8f8e86;--rule:rgba(255,255,255,.15);
 --accent:#5b9eec;--accent2:#4ec98a;--notice-bg:rgba(255,255,255,.06);}}
body{margin:0;padding:40px 22px;background:var(--bg);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,sans-serif}
.wrap{max-width:820px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
h1{font-size:23px;margin:0} .lede{color:var(--sec);font-size:14.5px;margin-top:4px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
@media(max-width:680px){.cards{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--rule);border-radius:14px;
 padding:22px 20px;display:flex;flex-direction:column;gap:10px}
.card.personality{border-top:3px solid var(--accent)}
.card.behavior{border-top:3px solid var(--accent2)}
.tag{font:600 10.5px/1 ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase}
.card.personality .tag{color:var(--accent)} .card.behavior .tag{color:var(--accent2)}
.card h2{font-size:18px;margin:2px 0 0;letter-spacing:-.01em}
.short{font-size:13.5px;color:var(--sec);margin:0}
.short b{color:var(--ink);font-weight:600}
details.learn{margin-top:2px}
details.learn>summary{cursor:pointer;font-size:13px;font-weight:600;color:var(--sec);
 list-style:none;display:flex;align-items:center;gap:5px}
details.learn>summary::-webkit-details-marker{display:none}
details.learn>summary::after{content:"▾";font-size:11px;transition:transform .15s}
details.learn[open]>summary::after{transform:rotate(180deg)}
.learn-body{font-size:13px;color:var(--sec);margin-top:8px;line-height:1.65}
.learn-body ul{margin:6px 0;padding-left:18px}
.learn-body li{margin:3px 0}
.learn-body h4{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);
 margin:12px 0 4px}
.tldr{background:var(--notice-bg);border-radius:8px;padding:8px 10px;margin:6px 0 2px;
 font-size:13px}
.tldr b{display:block;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
 color:var(--mut);margin-bottom:3px;font-weight:600}
details.legal{margin-top:6px} details.legal>summary{cursor:pointer;font-size:12px;color:var(--mut)}
.legal-body{font-size:11.5px;color:var(--mut);margin-top:4px;line-height:1.5}
a.run{display:block;text-align:center;margin-top:6px;padding:10px 14px;border-radius:9px;
 text-decoration:none;font-weight:600;font-size:13.5px;color:#fff}
.card.personality a.run{background:var(--accent)} .card.behavior a.run{background:var(--accent2)}
.foot{font-size:12px;color:var(--mut)}
"""

PERSONALITY_LEARN_MORE = """
<div class="tldr"><b>TL;DR</b>MindForm takes the personality it formed and answers a
  standardized Big Five questionnaire in-character.</div>
<div class="learn-body">
  <ul>
    <li>30 questions, answered once each</li>
    <li>5 Big Five domains, 15 more specific facets underneath them</li>
    <li>uses the character's already-frozen state -- nothing is retrained or reformed</li>
    <li>HEART's hidden target personality is never shown to it during the test</li>
  </ul>
  <h4>Three separate measurements</h4>
  <ul>
    <li><b>Internal MindForm personality</b> -- read directly off the formed character's own state</li>
    <li><b>BFI-2-S expressed / self-report personality</b> -- what it says about itself on the questionnaire</li>
    <li><b>Hidden target personality</b> -- the answer key, revealed only after the test, never shown to it</li>
  </ul>
  These are kept visibly separate on the results page -- they are not the same
  measurement and are never averaged together.
  <details class="legal"><summary>Licensing note ▾</summary>
    <div class="legal-body">BFI-2 items copyright 2015 by Oliver P. John and Christopher J.
      Soto. Reprinted with permission. Free for non-commercial research use; commercial use
      requires the authors' explicit permission and is not currently granted. This is the
      officially published 30-item short form (BFI-2-S), used because its exact item text and
      scoring key are independently verifiable; the full 60-item BFI-2 is gated behind the
      authors' own registration process.</div>
  </details>
</div>
"""

BEHAVIOR_LEARN_MORE = """
<div class="tldr"><b>TL;DR</b>HEART gives the formed character unseen behavioral situations
  and checks whether it makes the same choices as the intended person.</div>
<div class="learn-body">
  <ul>
    <li>every scenario is graded against a hidden, official ground-truth answer</li>
    <li>tests behavior in realistic situations, not just a personality label</li>
    <li>compares three ways of answering: Naive RAG (plain memory lookup), MindForm D1
      (MindForm's own retrieval), and MindForm D2 (D1 plus its persistent formed state)</li>
    <li>three depths to run it at: Quick Test (a handful of questions), Character Test
      (every question for one character), and Full Protocol (forms the character from
      all 1,000 memories first, then answers every question)</li>
    <li>Full Protocol's formation step is expensive -- typically hours -- so an already
      formed (or partly formed) character resumes instead of being re-run unnecessarily</li>
  </ul>
  <details class="legal"><summary>Research details ▾</summary>
    <div class="legal-body">Every prompt is scanned before it is sent so the hidden answer
      (and every other withheld label) can never leak in; the frozen snapshot is re-verified
      byte-for-byte after every question; and the ground truth is revealed only after an
      answer is already committed. The full research view -- question-level results,
      per-arm accuracy, retrieval details, snapshot integrity -- lives on HEART's own page.</div>
  </details>
</div>
"""

PAGE_HEAD = """<!doctype html><meta charset="utf-8">
<title>MindForm Tests</title>
<style>__STYLE__</style>
<div class="wrap">
  <div><h1>MindForm Tests</h1>
    <p class="lede">Choose what you want to evaluate.</p></div>
  <div class="cards">
    <div class="card personality">
      <div class="tag">Personality Test · __INSTRUMENT_NAME__</div>
      <h2>What personality did MindForm form?</h2>
      <p class="short">A standardized personality questionnaire. Shows <b>5 Big Five
        domains</b> and <b>15 facets</b>.</p>
      <details class="learn"><summary>Learn more</summary>__PERSONALITY_LEARN_MORE__</details>
      <a class="run" href="http://127.0.0.1:__PERSONALITY_PORT__/" target="_blank">
        Run Personality Test</a>
    </div>
    <div class="card behavior">
      <div class="tag">Behavior Test · HEART-Bench</div>
      <h2>Does MindForm behave like the target person?</h2>
      <p class="short">A behavioral benchmark using unseen real-life scenarios. Compares
        MindForm against baselines and a hidden ground truth.</p>
      <details class="learn"><summary>Learn more</summary>__BEHAVIOR_LEARN_MORE__</details>
      <a class="run" href="http://127.0.0.1:__HEART_PORT__/" target="_blank">
        Run Behavior Test</a>
    </div>
  </div>
  <p class="foot">Each opens in its own tab and runs independently -- starting one does not
    affect the other.</p>
</div>
"""


def render_page(heart_port, personality_port, instrument_name):
    html = PAGE_HEAD
    html = html.replace("__STYLE__", STYLE)
    html = html.replace("__INSTRUMENT_NAME__", instrument_name)
    html = html.replace("__PERSONALITY_LEARN_MORE__", PERSONALITY_LEARN_MORE)
    html = html.replace("__BEHAVIOR_LEARN_MORE__", BEHAVIOR_LEARN_MORE)
    html = html.replace("__PERSONALITY_PORT__", str(personality_port))
    html = html.replace("__HEART_PORT__", str(heart_port))
    return html


class LandingHandler(BaseHTTPRequestHandler):
    heart_port = 8500
    personality_port = 8501
    instrument_name = "BFI-2-S"

    def do_GET(self):
        body = render_page(self.heart_port, self.personality_port,
                           self.instrument_name).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _serve(server, name):
    try:
        server.serve_forever()
    except Exception as exc:
        print(f"{name} server stopped: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000, help="landing page port")
    ap.add_argument("--heart-port", type=int, default=8500)
    ap.add_argument("--personality-port", type=int, default=8501)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    # Import lazily so `python -m bench.tests --help` doesn't require a
    # HEART-Bench checkout just to print usage.
    from bench.heart import live as heart_live
    from bench.personality import live as personality_live
    from bench.heart.config import require_heart_bench
    require_heart_bench()

    heart_srv = ThreadingHTTPServer(("127.0.0.1", args.heart_port), heart_live.Handler)
    personality_srv = ThreadingHTTPServer(("127.0.0.1", args.personality_port),
                                          personality_live.Handler)
    threading.Thread(target=_serve, args=(heart_srv, "HEART"), daemon=True).start()
    threading.Thread(target=_serve, args=(personality_srv, "Personality"), daemon=True).start()

    LandingHandler.heart_port = args.heart_port
    LandingHandler.personality_port = args.personality_port
    LandingHandler.instrument_name = personality_live.INSTRUMENT.name
    landing_srv = ThreadingHTTPServer(("127.0.0.1", args.port), LandingHandler)

    url = f"http://127.0.0.1:{args.port}/"
    print(f"MINDFORM TESTS  →  {url}    (ctrl-c to stop everything)")
    print(f"  Personality Test ({personality_live.INSTRUMENT.name}) "
          f"→ http://127.0.0.1:{args.personality_port}/")
    print(f"  Behavior Test (HEART-Bench)         "
          f"→ http://127.0.0.1:{args.heart_port}/")
    if args.open:
        webbrowser.open(url)
    try:
        landing_srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
