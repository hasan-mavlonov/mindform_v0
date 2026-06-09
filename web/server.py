"""Stdlib HTTP server for the MindForm Console -- no web framework, no new deps.

Serves the static cockpit (``web/static/``) and a small JSON API that calls the
engine through ``engine_bridge``. Threaded so a slow DeepSeek call on one turn
never blocks the rest of the UI.

    python console.py                 # from the repo root (recommended)
    python -m web.server              # equivalent
    PORT=9000 python console.py       # choose a port
    python console.py --port 9000     # ditto
"""

import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Run from the repo root so the engine modules import regardless of CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from web import engine_bridge as bridge  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".map": "application/json; charset=utf-8",
}

log = logging.getLogger("mindform.web.server")


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "MindFormConsole/1.0"

    # ---- helpers ------------------------------------------------------------
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=400):
        self._send_json({"error": message}, status=status)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _resolve_static(self, path):
        """Map a URL path to a file inside STATIC_DIR, or None if out of bounds."""
        rel = path.lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR):       # block path traversal
            return None
        if os.path.isdir(full):
            full = os.path.join(full, "index.html")
        return full if os.path.isfile(full) else None

    def _serve_static(self, path):
        full = self._resolve_static(path)
        if full is None:
            self._send_error_json("not found", status=404)
            return
        with open(full, "rb") as f:
            body = f.read()
        ext = os.path.splitext(full)[1].lower()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routing ------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/config":
                self._send_json(bridge.ui_config())
            elif path == "/api/characters":
                self._send_json({"characters": bridge.roster()})
            elif path == "/api/state":
                name = (parse_qs(parsed.query).get("name") or [""])[0]
                if not name:
                    self._send_error_json("name is required")
                else:
                    self._send_json(bridge.load_snapshot(name))
            elif path.startswith("/api/"):
                self._send_error_json("unknown endpoint", status=404)
            else:
                self._serve_static(path)
        except FileNotFoundError:
            self._send_error_json("character not found", status=404)
        except Exception as exc:                  # never leak a stack to the browser
            log.exception("GET %s failed", path)
            self._send_error_json(str(exc), status=500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_body()
            if path == "/api/turn":
                name = (body.get("name") or "").strip()
                message = (body.get("message") or "").strip()
                if not name or not message:
                    self._send_error_json("name and message are required")
                    return
                self._send_json(bridge.run_turn(name, message))
            elif path == "/api/create/genesis":
                bio = (body.get("bio") or "").strip()
                if not bio:
                    self._send_error_json("a biography is required")
                    return
                self._send_json(bridge.create_genesis(bio))
            elif path == "/api/create/manual":
                identity = body.get("identity") or {}
                if not (identity.get("name") or "").strip():
                    self._send_error_json("a name is required")
                    return
                self._send_json(bridge.create_manual(identity, body.get("levels") or {}))
            elif path == "/api/select":
                name = (body.get("name") or "").strip()
                if not name:
                    self._send_error_json("name is required")
                    return
                self._send_json(bridge.load_snapshot(name))
            else:
                self._send_error_json("unknown endpoint", status=404)
        except FileNotFoundError:
            self._send_error_json("character not found", status=404)
        except Exception as exc:
            log.exception("POST %s failed", path)
            self._send_error_json(str(exc), status=500)

    def log_message(self, fmt, *args):            # quieter, prefixed access log
        log.info("%s - %s", self.address_string(), fmt % args)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argv = argv if argv is not None else sys.argv[1:]

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    if "--host" in argv:
        host = argv[argv.index("--host") + 1]

    httpd = ThreadingHTTPServer((host, port), ConsoleHandler)
    httpd.daemon_threads = True
    url = f"http://{host}:{port}/"
    print("\n  MindForm Console")
    print(f"  -> open {url}")
    print("  (Ctrl+C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
