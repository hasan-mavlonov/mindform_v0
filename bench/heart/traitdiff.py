"""Trait-differentiation sanity diagnostic — NOT a HEART benchmark score.

    python -m bench.heart.traitdiff --characters CHAR_01,CHAR_08 --memories 50

Ingests two or more HEART characters from a neutral MindForm start, using only
the raw memories, and then asks three questions the benchmark cannot:

  * do the formed OCEAN vectors DIFFER between characters, or does everything
    collapse to the same state?
  * do they saturate at the ±1 boundary?
  * do they point the same way as HEART's withheld Big Five?

The withheld labels are read strictly after every character's formation has
finished, and never enter a prompt. If characters do not differentiate, D2 has
no signal to carry and the benchmark arm is moot — which is why this runs before
the expensive smoke test.
"""

import argparse
import json
import os
import time

from core.config import BASIS, BASIS_NAMES
from bench.heart import heartdata, mfadapter
from bench.heart.config import RESULTS_ROOT
from bench.heart.logbook import Logbook, new_run_id

LONG = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
        "A": "agreeableness", "N": "neuroticism"}
SATURATION = 0.95
# |withheld| below this is the scale midpoint: no direction to agree with.
DIRECTION_DEADBAND = 0.15


def form_character(log, cid, n_mem, run_dir, resume=True):
    char = heartdata.load_characters()[cid]
    name = f"traitdiff {cid}"
    label = f"frozen-{cid}"
    manifest_path = os.path.join(run_dir, "snapshots", label, "manifest.json")

    if resume and os.path.exists(manifest_path):
        sid = mfadapter.restore(name, run_dir, label)
        log.say(f"{cid}: resumed snapshot {sid[:16]}…")
        return name, mfadapter.state_summary(name), sid

    mems = heartdata.ingestible_memories(char)
    mems = mems if not n_mem else mems[:n_mem]
    mfadapter.register_id_map(mems)

    log.banner(f"FORMING {cid}", f"{len(mems)} memories, chronological, neutral start")
    mfadapter.create_neutral(name, char.get("occupation", "N/A"))
    log.event("traitdiff_ingest_start", character=cid, memories=len(mems),
              big_five_used=False)
    t0 = time.time()
    mfadapter.ingest_all(name, mems, log, on_step=log.ingest_tick)
    sid, _ = mfadapter.freeze(name, run_dir, label)
    state = mfadapter.state_summary(name)
    log.say(f"{cid}: formed in {(time.time()-t0)/60:.1f} min → snapshot {sid[:16]}…")
    log.say("  " + "  ".join(f"{k}{state['traits'][k]:+.3f}" for k in BASIS))
    log.event("traitdiff_formed", character=cid, snapshot_id=sid,
              traits=state["traits"], memories=len(mems))
    return name, state, sid


def analyse(formed):
    """formed: {cid: state}. Withheld labels are read HERE, after all formation."""
    chars = heartdata.load_characters()
    rows = []
    for cid, state in formed.items():
        gt = heartdata.withheld_big_five(chars[cid])          # post-hoc only
        row = {"character": cid, "formed": state["traits"], "withheld_01": gt,
               "withheld_signed": {k: round(gt.get(LONG[k], 0.5) * 2 - 1, 3) for k in BASIS}}
        # A withheld value at (or next to) the midpoint has no direction to
        # agree with -- CHAR_01's conscientiousness is exactly 0.50/0.00, so
        # scoring its sign would manufacture a miss. Those are scored "n/a".
        row["direction_agreement"] = {}
        for k in BASIS:
            gt = row["withheld_signed"][k]
            if abs(gt) < DIRECTION_DEADBAND:
                row["direction_agreement"][k] = None
            else:
                row["direction_agreement"][k] = (gt >= 0) == (row["formed"][k] >= 0)
        row["agreement_count"] = sum(1 for v in row["direction_agreement"].values() if v)
        row["agreement_scored"] = sum(1 for v in row["direction_agreement"].values()
                                      if v is not None)
        row["saturated"] = [k for k in BASIS if abs(row["formed"][k]) >= SATURATION]
        rows.append(row)

    # pairwise separation between formed vectors
    pairs = []
    ids = list(formed)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = formed[ids[i]]["traits"], formed[ids[j]]["traits"]
            l1 = sum(abs(a[k] - b[k]) for k in BASIS)
            pairs.append({"pair": f"{ids[i]} vs {ids[j]}",
                          "l1_distance": round(l1, 4),
                          "mean_abs_diff": round(l1 / len(BASIS), 4),
                          "per_trait": {k: round(a[k] - b[k], 4) for k in BASIS}})
    return rows, pairs


def _fill_css(v):
    """Bar geometry for a signed value. Built with concatenation, not %-format:
    the literal '%' in 'left:50%' is a format character and silently breaks it."""
    w = abs(v) * 50
    side = "left:50%" if v >= 0 else "right:50%"
    return f"{side};width:{w:.1f}%"


def render_html(rows, pairs, path, run_id):
    S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    SD = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]

    def bars(row):
        out = []
        for i, k in enumerate(BASIS):
            f, g = row["formed"][k], row["withheld_signed"][k]
            out.append(f"""
        <div class="tr">
          <span class="tl">{BASIS_NAMES[k]}</span>
          <span class="track">
            <span class="mid"></span>
            <span class="fill" style="--c:var(--s{i+1});{_fill_css(f)}"></span>
            <span class="gt" style="left:{50+g*50:.1f}%"></span>
          </span>
          <span class="num">{f:+.3f}</span>
          <span class="num muted">{g:+.2f}</span>
          <span class="{ {True:'ok',False:'bad',None:'mut'}[row['direction_agreement'][k]] }">{ {True:'✓',False:'✗',None:'–'}[row['direction_agreement'][k]] }</span>
        </div>""")
        return "".join(out)

    cards = "".join(f"""
      <div class="card">
        <h2>{r['character']}</h2>
        <div class="hdr"><span>trait</span><span></span><span>formed</span><span>withheld</span><span>dir</span></div>
        {bars(r)}
        <p class="foot">direction agreement <b>{r['agreement_count']}/{r['agreement_scored']}</b>
           <span class="mut">(traits whose withheld value sits at the midpoint are not scored)</span> ·
           saturated at ±{SATURATION}: <b>{', '.join(r['saturated']) or 'none'}</b></p>
      </div>""" for r in rows)

    prs = "".join(f"""<tr><td>{p['pair']}</td><td class="num">{p['l1_distance']}</td>
        <td class="num">{p['mean_abs_diff']}</td>
        <td class="num">{' '.join(f"{k}{v:+.2f}" for k,v in p['per_trait'].items())}</td></tr>"""
        for p in pairs)

    html = f"""<!doctype html><meta charset="utf-8"><title>Trait differentiation · {run_id}</title>
<style>
:root{{color-scheme:light;--bg:#f2f1ee;--card:#fff;--ink:#0b0b0b;--sec:#52514e;--mut:#83817a;
 --rule:rgba(11,11,11,.12);--ok:#1a7f4b;--bad:#b3261e;
 --s1:{S[0]};--s2:{S[1]};--s3:{S[2]};--s4:{S[3]};--s5:{S[4]};}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;
 --bg:#111110;--card:#232322;--ink:#fff;--sec:#c3c2b7;--mut:#8f8e86;--rule:rgba(255,255,255,.15);
 --ok:#4ec98a;--bad:#ff8a7a;--s1:{SD[0]};--s2:{SD[1]};--s3:{SD[2]};--s4:{SD[3]};--s5:{SD[4]};}}}}
body{{margin:0;padding:22px;background:var(--bg);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;display:flex;flex-direction:column;gap:14px}}
h1{{font-size:19px;margin:0}}
h2{{font-size:14px;margin:0 0 10px;font-family:ui-monospace,Menlo,monospace}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:15px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}}
.tr,.hdr{{display:grid;grid-template-columns:118px 1fr 56px 56px 20px;gap:8px;align-items:center}}
.hdr{{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);
 padding-bottom:5px;border-bottom:1px solid var(--rule);margin-bottom:6px}}
.tl{{font-size:12px}}
.track{{position:relative;height:14px;background:var(--bg);border-radius:3px}}
.mid{{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--rule)}}
.fill{{position:absolute;top:1px;bottom:1px;background:var(--c);border-radius:2px}}
.gt{{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink);opacity:.75}}
.num{{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;text-align:right;
 font-variant-numeric:tabular-nums}}
.muted,.mut{{color:var(--mut)}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}}
.foot{{font-size:11.5px;color:var(--sec);margin:10px 0 0;padding-top:8px;border-top:1px solid var(--rule)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule)}}
th{{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}}
.note{{font-size:12px;color:var(--sec)}}
.key{{font-size:11.5px;color:var(--mut);display:flex;gap:16px;flex-wrap:wrap}}
</style>
<div class="wrap">
  <div>
    <h1>Trait differentiation · <span style="font-family:ui-monospace">{run_id}</span></h1>
    <p class="note">Formed from raw episodic memories only, neutral start. HEART's Big Five was
      read <b>after</b> every character finished forming and never entered a prompt.
      Sanity diagnostic — not a benchmark score.</p>
    <div class="key"><span>coloured bar = MindForm formed value</span>
      <span>│ vertical tick = HEART withheld value</span></div>
  </div>
  <div class="grid">{cards}</div>
  <div class="card"><h2>separation between formed vectors</h2>
    <table><thead><tr><th>pair</th><th>L1 distance</th><th>mean |Δ| per trait</th>
      <th>per-trait difference</th></tr></thead><tbody>{prs}</tbody></table>
    <p class="foot">Collapse to one state would show as an L1 distance near 0. Differentiation
      is a precondition for D2 carrying any information at all.</p>
  </div>
</div>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--characters", default="CHAR_01,CHAR_08")
    ap.add_argument("--memories", type=int, default=50, help="0 = all")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cids = [c.strip() for c in args.characters.split(",") if c.strip()]
    run_id = args.run_id or new_run_id("traitdiff")
    log = Logbook(run_id, RESULTS_ROOT)
    log.banner(f"TRAIT DIFFERENTIATION · {run_id}",
               f"{', '.join(cids)} · {args.memories or 'all'} memories each · "
               f"withheld labels read only after formation")

    formed = {}
    for cid in cids:
        _, state, _ = form_character(log, cid, args.memories, log.dir,
                                     resume=not args.no_resume)
        formed[cid] = state

    rows, pairs = analyse(formed)

    log.banner("RESULT", "post-hoc comparison against withheld Big Five")
    hdr = f"{'trait':<20}" + "".join(f"{r['character']+' formed':>18}"
                                     f"{r['character']+' withheld':>20}" for r in rows)
    log.say(hdr)
    for k in BASIS:
        line = f"{BASIS_NAMES[k]:<20}"
        for r in rows:
            da = r["direction_agreement"][k]
            mark = "--" if da is None else ("OK" if da else "XX")
            gt = r["withheld_signed"][k]
            line += f"{r['formed'][k]:>+18.3f}" + f"{gt:+.2f} {mark}".rjust(20)
        log.say(line)
    log.say("")
    for r in rows:
        log.say(f"  {r['character']}: direction agreement "
                f"{r['agreement_count']}/{r['agreement_scored']} scored · "
                f"saturated: {', '.join(r['saturated']) or 'none'}")
    log.say("")
    for p in pairs:
        log.say(f"  {p['pair']}: L1={p['l1_distance']}  mean|Δ|={p['mean_abs_diff']}  "
                + " ".join(f"{k}{v:+.2f}" for k, v in p["per_trait"].items()))

    collapse = all(p["mean_abs_diff"] < 0.05 for p in pairs) if pairs else False
    log.say("")
    log.say("  VERDICT: characters COLLAPSED to the same state — D2 has no signal to carry."
            if collapse else
            "  VERDICT: characters differentiated — D2 has something to carry.")

    out = {"run_id": run_id, "characters": rows, "pairs": pairs,
           "memories_per_character": args.memories, "collapsed": collapse}
    with open(os.path.join(log.dir, "traitdiff.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    html = render_html(rows, pairs, os.path.join(log.dir, "traitdiff.html"), run_id)
    log.event("traitdiff_result", **out)
    log.say(f"\njson : {os.path.join(log.dir, 'traitdiff.json')}")
    log.say(f"html : {html}")


if __name__ == "__main__":
    main()
