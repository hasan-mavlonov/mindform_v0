"""Run logging: machine-readable JSONL, a human run log, and console output.

Every benchmark event lands in ``events.jsonl`` (one JSON object per line, all
of them carrying ``run_id``/``event``/``ts``) so the dashboard and any later
analysis read files rather than re-running anything. ``run.log`` is the same
story in prose. The console renderer is what you watch while it runs.
"""

import json
import os
import sys
import time
import datetime

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"
BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"

_WIDTH = 76


def _color(s, c):
    return f"{c}{s}{RESET}" if sys.stdout.isatty() else str(s)


def new_run_id(prefix):
    return f"{prefix}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"


class Logbook:
    def __init__(self, run_id, root):
        self.run_id = run_id
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.events_path = os.path.join(self.dir, "events.jsonl")
        self.log_path = os.path.join(self.dir, "run.log")
        self.started = time.time()

    # ---- writing -----------------------------------------------------------
    def event(self, kind, **payload):
        rec = {
            "run_id": self.run_id,
            "event": kind,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - self.started, 2),
            **payload,
        }
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def note(self, text, level="INFO"):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} [{level}] {text}"
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return line

    def read_events(self, kind=None):
        if not os.path.exists(self.events_path):
            return []
        out = []
        with open(self.events_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind is None or rec.get("event") == kind:
                    out.append(rec)
        return out

    # ---- console -----------------------------------------------------------
    def banner(self, title, subtitle=None):
        print("\n" + "=" * _WIDTH)
        print(_color(title, BOLD))
        if subtitle:
            print(_color(subtitle, DIM))
        print("=" * _WIDTH)
        self.note(title + (f" | {subtitle}" if subtitle else ""))

    def say(self, text, level="INFO"):
        print(text)
        self.note(_strip_ansi(text), level)

    def ingest_tick(self, i, total, mem, traits, dt_ms):
        if i % 10 == 0 or i == total - 1 or i < 3:
            tv = " ".join(f"{k}{traits[k]:+.2f}" for k in "OCEAN")
            bar = _progress(i + 1, total, 22)
            print(f"\r  {bar} {i+1:>4}/{total}  [{mem['timeline'][:22]:<22}] "
                  f"{tv}  {dt_ms:>5.0f}ms", end="", flush=True)
        if i == total - 1:
            print()

    def question_header(self, arm, char_id, idx, total, scenario, trigger, options):
        print("\n" + "=" * _WIDTH)
        print(_color(f"HEART-Bench | {char_id} | Question {idx}/{total} | Arm: {arm}", BOLD))
        print("=" * _WIDTH)
        print(f"\n{_color('Situation:', BOLD)}")
        print(f"  {scenario.get('name','?')}  "
              f"{_color('(' + str((scenario.get('setting') or {}).get('location','?')) + ')', DIM)}")
        print(_wrap(scenario.get("context_text", ""), 2, 3))
        trig = trigger or {}
        print(f"\n{_color('Trigger:', BOLD)}")
        print(_wrap(f"{trig.get('sender','?')}: {trig.get('message_content','')}", 2, 2))
        print(f"\n{_color('Options:', BOLD)}")
        for o in options:
            print(_wrap(f"{o['label']}. {o['content']}", 2, 2))

    def retrieval_block(self, memories, embedder, k_shown=5):
        print(f"\n{_color(f'Retrieved memories ({len(memories)}, embedder={embedder}):', BOLD)}")
        for i, m in enumerate(memories[:k_shown], 1):
            score = m.get("score")
            s = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            print(f"  {i:>2}. [score={s}] [{m.get('anon_id','?')}] "
                  f"{_one_line(m.get('text',''), 62)}")
        if len(memories) > k_shown:
            print(_color(f"  … {len(memories)-k_shown} more (full list in events.jsonl)", DIM))

    def state_block(self, state):
        if not state:
            return
        print(f"\n{_color('MindForm state:', BOLD)}")
        for k in "OCEAN":
            v = state["traits"][k]
            print(f"  {k}: {_bar(v)} {v:+.2f}")
        if state.get("top_values"):
            print("  Values:  " + ", ".join(f"{n} {v:+.2f}" for n, v in state["top_values"]))
        if state.get("top_drives"):
            print("  Needs:   " + ", ".join(f"{n} {v:.2f}" for n, v in state["top_drives"]))
        if state.get("beliefs"):
            print("  Beliefs: " + "; ".join(state["beliefs"][:2]))

    def commit(self, choice):
        print(f"\n{_color('→ chose: ' + str(choice), BOLD + CYAN)}")
        print(_color("ANSWER COMMITTED", DIM))

    def reveal(self, truth, correct, tallies, latency_ms, usage, cost):
        print(f"\nGround truth: {truth}")
        print("Result: " + (_color("✅ CORRECT", GREEN) if correct else _color("❌ INCORRECT", RED)))
        print(f"\n{_color('Running accuracy:', BOLD)}")
        for arm, (c, n) in tallies.items():
            pct = (100.0 * c / n) if n else 0.0
            print(f"  {arm:<22} {c}/{n} = {pct:5.1f}%")
        cost_s = f"${cost:.4f}" if cost is not None else "n/a (rates unset)"
        print(f"\nLatency: {latency_ms:.0f} ms | tokens in/out: "
              f"{usage.get('input_tokens','?')}/{usage.get('output_tokens','?')} | cost: {cost_s}")
        print("=" * _WIDTH)

    def integrity(self, before, after, mutated, leak_ok, leak_detail=""):
        ok = _color("OK", GREEN); bad = _color("FAIL", RED)
        print(f"{_color('Snapshot:', DIM)} {before[:12]}… → {after[:12]}…  "
              f"state_mutated={'no ' + ok if not mutated else 'YES ' + bad}   "
              f"leak_check={ok if leak_ok else bad + ' ' + leak_detail}")


def _progress(done, total, width):
    filled = int(width * done / max(total, 1))
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def _bar(v, width=20):
    mid = width // 2
    pos = int(round(mid + v * mid))
    pos = max(0, min(width, pos))
    cells = ["·"] * (width + 1)
    cells[mid] = "|"
    cells[pos] = "█"
    return "".join(cells)


def _one_line(text, n):
    t = " ".join((text or "").split())
    return t[:n] + ("…" if len(t) > n else "")


def _wrap(text, indent, max_lines):
    words = " ".join((text or "").split()).split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > _WIDTH - indent:
            lines.append(cur); cur = w
            if len(lines) >= max_lines:
                break
        else:
            cur = f"{cur} {w}".strip()
    if cur and len(lines) < max_lines:
        lines.append(cur)
    pad = " " * indent
    out = "\n".join(pad + l for l in lines)
    if len(words) and len(lines) >= max_lines:
        out += " …"
    return out


def _strip_ansi(s):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)
