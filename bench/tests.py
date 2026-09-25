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
"""

import argparse
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE_TEMPLATE = """<!doctype html><meta charset="utf-8">
<title>MindForm Tests</title>
<style>
:root{{color-scheme:light;--bg:#f2f1ee;--card:#fff;--ink:#0b0b0b;--sec:#52514e;--mut:#83817a;
 --rule:rgba(11,11,11,.12);--accent:#2a78d6;}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;
 --bg:#111110;--card:#232322;--ink:#fff;--sec:#c3c2b7;--mut:#8f8e86;--rule:rgba(255,255,255,.15);
 --accent:#5b9eec;}}}}
body{{margin:0;padding:40px 22px;background:var(--bg);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,sans-serif}}
.wrap{{max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:22px}}
h1{{font-size:22px;margin:0}} .sub{{color:var(--sec);font-size:14px;margin-top:4px}}
.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:640px){{.cards{{grid-template-columns:1fr}}}}
a.card{{display:block;background:var(--card);border:1px solid var(--rule);border-radius:14px;
 padding:22px 20px;text-decoration:none;color:inherit;transition:border-color .15s}}
a.card:hover{{border-color:var(--accent)}}
.tag{{font:600 10.5px/1 ui-monospace,monospace;letter-spacing:.06em;text-transform:uppercase;
 color:var(--accent)}}
.card h2{{font-size:17px;margin:8px 0 4px}}
.card p{{font-size:13px;color:var(--sec);margin:0}}
.foot{{font-size:12px;color:var(--mut)}}
</style>
<div class="wrap">
  <div><h1>MindForm Tests</h1>
    <p class="sub">Two independent ways to check what MindForm actually does.</p></div>
  <div class="cards">
    <a class="card" href="http://127.0.0.1:{personality_port}/" target="_blank">
      <div class="tag">Personality Test · {instrument_name}</div>
      <h2>What personality did MindForm form?</h2>
      <p>Give MindForm someone's life history, let it form a personality, then check whether
        it understood who they are.</p>
    </a>
    <a class="card" href="http://127.0.0.1:{heart_port}/" target="_blank">
      <div class="tag">Behavior Test · HEART-Bench</div>
      <h2>Does it behave like that person?</h2>
      <p>The deeper research benchmark: real dramatic scenarios, hidden ground-truth
        answers, naive-RAG / D1 / D2 comparison, full question-level results.</p>
    </a>
  </div>
  <p class="foot">Each opens in its own tab and runs independently -- starting one does not
    affect the other.</p>
</div>
"""


class LandingHandler(BaseHTTPRequestHandler):
    heart_port = 8500
    personality_port = 8501
    instrument_name = "BFI-2-S"

    def do_GET(self):
        body = PAGE_TEMPLATE.format(heart_port=self.heart_port,
                                    personality_port=self.personality_port,
                                    instrument_name=self.instrument_name).encode("utf-8")
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
