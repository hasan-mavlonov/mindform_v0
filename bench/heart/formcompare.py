"""Does disabling the model's reasoning change how MindForm forms personality?

    python -m bench.heart.formcompare --character CHAR_01 --memories 50

`core/llm.py` calls a thinking model without ``reasoning_effort``, so every
ingestion call burns ~865 invisible reasoning tokens: 6.5 s/call against 1.0 s
with reasoning off. Over 1,000 memories that is ~9 hours versus ~1.4. Before
buying that speedup we need to know whether it changes the formed character.

Three ingestions of the SAME memories in the SAME order:

    think_on_1   production configuration          (the reference)
    think_off    reasoning_effort="none"           (the treatment)
    think_on_2   production configuration again    (the noise floor)

The ingestion calls run at temperature 0.1-0.2, so two identical configurations
do NOT produce identical characters. Without ``think_on_2`` a difference between
on and off is uninterpretable — it could be sampling jitter. The question this
script answers is therefore not "is there a difference" but:

    is  |think_on_1 - think_off|  larger than  |think_on_1 - think_on_2| ?

Only if the treatment moves the character further than re-running the same
configuration does disabling reasoning actually cost us anything.
"""

import argparse
import json
import os
import time

from core.config import BASIS, BASIS_NAMES
from bench.heart import heartdata, mfadapter
from bench.heart.config import RESULTS_ROOT, require_heart_bench
from bench.heart.logbook import Logbook, new_run_id

CONFIGS = [
    ("think_on_1", False, "production config (reference)"),
    ("think_off", True, 'reasoning_effort="none" (treatment)'),
    ("think_on_2", False, "production config again (noise floor)"),
]


def form(log, cid, n_mem, run_dir, label, no_thinking, resume=True):
    char = heartdata.load_characters()[cid]
    name = f"formcmp {cid} {label}"
    snap_label = f"frozen-{label}"
    manifest = os.path.join(run_dir, "snapshots", snap_label, "manifest.json")

    if resume and os.path.exists(manifest):
        mfadapter.restore(name, run_dir, snap_label)
        log.say(f"{label}: resumed")
        with open(manifest, encoding="utf-8") as fh:
            meta = json.load(fh)
        return mfadapter.state_summary(name), meta.get("seconds")

    mems = heartdata.ingestible_memories(char)[: n_mem or None]
    mfadapter.register_id_map(mems)
    log.banner(f"FORMING · {label}", f"{cid} · {len(mems)} memories · no_thinking={no_thinking}")
    mfadapter.create_neutral(name, char.get("occupation", "N/A"))

    t0 = time.time()
    if no_thinking:
        with mfadapter.NoThinkingPatch():
            mfadapter.ingest_all(name, mems, log, on_step=log.ingest_tick)
    else:
        mfadapter.ingest_all(name, mems, log, on_step=log.ingest_tick)
    secs = time.time() - t0

    sid, man = mfadapter.freeze(name, run_dir, snap_label)
    man["seconds"] = round(secs, 1)
    with open(os.path.join(run_dir, "snapshots", snap_label, "manifest.json"),
              "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)

    state = mfadapter.state_summary(name)
    log.say(f"\n{label}: {secs/60:.1f} min ({secs/len(mems):.1f} s/memory) → {sid[:16]}…")
    log.say("  " + "  ".join(f"{k}{state['traits'][k]:+.3f}" for k in BASIS))
    log.event("formcompare_formed", label=label, character=cid, no_thinking=no_thinking,
              seconds=round(secs, 1), traits=state["traits"], snapshot_id=sid)
    return state, round(secs, 1)


def l1(a, b):
    return round(sum(abs(a[k] - b[k]) for k in BASIS), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="CHAR_01")
    ap.add_argument("--memories", type=int, default=50)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    require_heart_bench()

    run_id = args.run_id or new_run_id("formcompare")
    log = Logbook(run_id, RESULTS_ROOT)
    log.banner(f"FORMATION COMPARISON · {run_id}",
               f"{args.character} · {args.memories} memories · "
               f"treatment vs a same-config replicate")

    states, timings = {}, {}
    for label, no_think, _desc in CONFIGS:
        states[label], timings[label] = form(log, args.character, args.memories,
                                             log.dir, label, no_think,
                                             resume=not args.no_resume)

    ref = states["think_on_1"]["traits"]
    treat = states["think_off"]["traits"]
    noise = states["think_on_2"]["traits"]

    d_treatment = l1(ref, treat)
    d_noise = l1(ref, noise)

    log.banner("RESULT", "trait vectors after identical memories")
    log.say(f"{'trait':<20}{'think_on_1':>13}{'think_off':>13}{'think_on_2':>13}"
            f"{'|off-on1|':>12}{'|on2-on1|':>12}")
    for k in BASIS:
        log.say(f"{BASIS_NAMES[k]:<20}{ref[k]:>+13.3f}{treat[k]:>+13.3f}{noise[k]:>+13.3f}"
                f"{abs(treat[k]-ref[k]):>12.3f}{abs(noise[k]-ref[k]):>12.3f}")
    log.say("")
    log.say(f"  L1 distance, treatment (think_off vs think_on_1) : {d_treatment}")
    log.say(f"  L1 distance, noise floor (think_on_2 vs think_on_1): {d_noise}")
    ratio = (d_treatment / d_noise) if d_noise else float("inf")
    log.say(f"  ratio treatment/noise                             : {ratio:.2f}")
    log.say("")
    for label, _, desc in CONFIGS:
        s = timings[label]
        log.say(f"  {label:<12} {s/60:>5.1f} min  ({s/args.memories:>5.1f} s/memory)  {desc}")

    on_rate = (timings["think_on_1"] + timings["think_on_2"]) / 2 / args.memories
    off_rate = timings["think_off"] / args.memories
    log.say("")
    log.say(f"  speedup: {on_rate/off_rate:.1f}× "
            f"({on_rate:.1f} → {off_rate:.1f} s/memory)")
    log.say(f"  projected for 1,000 memories: "
            f"{on_rate*1000/3600:.1f} h → {off_rate*1000/3600:.1f} h")

    if ratio <= 1.0:
        verdict = ("SAFE — disabling reasoning moved the character no further than "
                   "re-running the same config. The speedup is free.")
    elif ratio <= 2.0:
        verdict = ("MARGINAL — the treatment moved the character somewhat more than "
                   "sampling noise. Usable with the deviation recorded.")
    else:
        verdict = ("COSTLY — disabling reasoning changes formation well beyond the "
                   "noise floor. Run Stage 1 in the production configuration.")
    log.say(f"\n  VERDICT: {verdict}")

    out = {
        "run_id": run_id, "character": args.character, "memories": args.memories,
        "traits": {k: states[k]["traits"] for k in states},
        "values": {k: states[k]["top_values"] for k in states},
        "drives": {k: states[k]["top_drives"] for k in states},
        "esteem": {k: states[k]["esteem"] for k in states},
        "beliefs": {k: states[k]["beliefs"] for k in states},
        "seconds": timings,
        "l1_treatment": d_treatment, "l1_noise": d_noise, "ratio": round(ratio, 3),
        "s_per_memory": {"thinking_on": round(on_rate, 2), "thinking_off": round(off_rate, 2)},
        "verdict": verdict,
    }
    with open(os.path.join(log.dir, "formcompare.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.event("formcompare_result", **out)
    log.say(f"\njson: {os.path.join(log.dir, 'formcompare.json')}")


if __name__ == "__main__":
    main()
