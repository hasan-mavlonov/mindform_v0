"""Personality Fidelity — did MindForm understand who this person is?

    python -m bench.heart.fidelity --character CHAR_01

Complements HEART's Behavioral Fidelity (does MindForm act like this person?)
with a cheaper, more direct question: read MindForm's frozen OCEAN state and
compare it straight to HEART's hidden Big Five target. No LLM call. No text is
generated or read. That absence is the point.

Why not administer a real personality inventory (BFI-2 / IPIP-NEO-120 / etc.)
instead of reading the state directly: having the model answer a questionnaire
in character would require generating text from the formed state -- the same
state-to-text step HEART's D2 arm already tests. A bad questionnaire score
could then mean "formation is wrong" OR "self-report generation is wrong",
which destroys the one thing this benchmark exists to isolate. Reading the
five numbers directly has no generation step to blame, so a bad score here
means one thing: the formed representation itself doesn't match the target.
(IPIP-NEO-120 remains a reasonable Phase 2 EXPLORATORY probe -- public domain,
facet-level, no licensing friction -- but it answers a different question
(self-report fidelity) and is deliberately not implemented here.)

This never forms or re-forms a character. It reads whichever frozen snapshot
already exists (bench.heart.snapshots) and fails with instructions if none
does. Use bench.heart.prepare or the live UI's Full Protocol Benchmark to
build one first.

Ground-truth discipline: HEART's big_five is read from the character record
only AFTER the frozen state has been restored and its traits read -- mirroring
HEART's own reveal-after-commit rule, even though nothing here could leak
backward into formation (the read is a pure function of files already on
disk). The frozen snapshot directory is hashed before and after and the two
must match: this tool is read-only by construction, not just by convention.
"""

import argparse
import math
import os

from core.config import BASIS, BASIS_NAMES
from bench.heart import heartdata, mfadapter, snapshots
from bench.heart.config import RESULTS_ROOT, require_heart_bench
from bench.heart.dashboard import ARM_LABEL, ARM_ORDER, load_events

LONG = {"O": "openness", "C": "conscientiousness", "E": "extraversion",
        "A": "agreeableness", "N": "neuroticism"}

# Secondary/debug only -- never the headline metric. Matches traitdiff.py's
# convention so the two diagnostics never disagree on what counts as "no
# direction to agree with".
DIRECTION_DEADBAND = 0.15
SATURATION = 0.95


class NoSnapshot(Exception):
    """No frozen snapshot exists for this character at this tier."""


# ---------------------------------------------------------------------------
# math -- five numbers each, nothing more
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    """cos(angle) between two same-length vectors. 1 = identical direction and
    magnitude ratio, 0 = orthogonal, -1 = opposite. Undefined (returns None)
    for a zero vector -- a character with every trait at exactly 0.0 has no
    direction to compare."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


def mean_absolute_error(a, b):
    """Average |difference| per dimension, on the same -1..+1 scale both live on."""
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def signed_target(char):
    """HEART's hidden Big Five, rescaled from its native 0..1 to MindForm's
    -1..+1 so the two are on the same axis before any arithmetic touches them."""
    gt = heartdata.withheld_big_five(char)
    return {k: round(gt.get(LONG[k], 0.5) * 2 - 1, 4) for k in BASIS}


def direction_agreement(formed, target_signed):
    """Per-trait sign match, secondary/debug only. A target inside the deadband
    (HEART's own scale midpoint) has no direction to agree with and scores
    ``None`` rather than manufacturing a miss."""
    out = {}
    for k in BASIS:
        t = target_signed[k]
        out[k] = None if abs(t) < DIRECTION_DEADBAND else (t >= 0) == (formed[k] >= 0)
    return out


# ---------------------------------------------------------------------------
# reading a frozen snapshot -- never forming one
# ---------------------------------------------------------------------------
def _snapshot_digest(run_dir, label):
    import hashlib
    d = os.path.join(run_dir, "snapshots", label)
    h = hashlib.sha256()
    for fname in sorted(os.listdir(d)):
        h.update(fname.encode())
        with open(os.path.join(d, fname), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def compute(character_id, tier=None):
    """Personality Fidelity for one character's already-frozen snapshot.

    ``tier``: "full" or "dev" to require that specific tier, or None to prefer
    a full snapshot and fall back to dev. Never ingests, never creates a
    character, never calls freeze() -- only restore() over an already-frozen
    snapshot, then a read.
    """
    chars = heartdata.load_characters()
    if character_id not in chars:
        raise NoSnapshot(f"{character_id} is not a HEART character")
    char = chars[character_id]

    if tier is not None:
        snap = snapshots.best_for_tier(character_id, tier)
        if not snap:
            raise NoSnapshot(
                f"{character_id} has no {tier} snapshot prepared. "
                f"{'Run a Full Protocol Benchmark first.' if tier == 'full' else 'Run bench.heart.runner --stage 0 first.'}")
    else:
        snap = snapshots.best_for_tier(character_id, "full") or snapshots.best_for(character_id)
        if not snap:
            raise NoSnapshot(
                f"{character_id} has no frozen snapshot at all. Prepare one with "
                f"bench.heart.prepare (full) or bench.heart.runner --stage 0 (dev) first.")

    name, run_dir, label = snap["bench_name"], snap["run_dir"], snap["label"]
    digest_before = _snapshot_digest(run_dir, label)

    # Read the frozen OCEAN state. restore() copies the FROZEN files over the
    # live path and returns their hash; it never touches the frozen copy in
    # run_dir/snapshots/label, which is what digest_before/after verifies.
    restored_id = mfadapter.restore(name, run_dir, label)
    formed = mfadapter.trait_vector(mfadapter.load_character(name))

    digest_after = _snapshot_digest(run_dir, label)
    if digest_after != digest_before:
        # Should be impossible -- restore() only ever writes to the live path --
        # but this tool's entire premise is "read-only", so it checks rather
        # than assumes.
        raise RuntimeError(
            f"frozen snapshot at {run_dir}/snapshots/{label} changed while reading "
            f"it ({digest_before[:12]}… -> {digest_after[:12]}…) -- refusing the result")

    # HEART's ground truth, read only now -- after the formed state above was
    # already captured. Nothing downstream of this line can feed back into
    # ``formed``; the ordering exists to mirror HEART's own reveal-after-commit
    # discipline, not because it changes what gets measured.
    target_signed = signed_target(char)
    target_raw = heartdata.withheld_big_five(char)

    mf_vec = [formed[k] for k in BASIS]
    ht_vec = [target_signed[k] for k in BASIS]

    is_full = bool(snap["full_protocol"])
    total_memories = len(char.get("episodic_memory_set") or [])

    return {
        "character": character_id,
        "occupation": char.get("occupation"),
        "tier": "full" if is_full else "dev",
        "protocol": {
            "faithful": is_full,
            "label": "FULL PROTOCOL" if is_full else "DEVELOPMENT TEST",
            "note": ("Formed from the complete memory set." if is_full else
                     f"Formed from {snap['memories']} of {total_memories} memories -- "
                     f"not a protocol-faithful fidelity score."),
        },
        "snapshot": {"memories": snap["memories"], "total_memories": total_memories,
                     "snapshot_id": restored_id, "run_id": snap["run_id"],
                     "run_dir": run_dir, "label": label},
        "trait_similarity": (round(cosine_similarity(mf_vec, ht_vec), 4)
                             if cosine_similarity(mf_vec, ht_vec) is not None else None),
        "average_trait_error": round(mean_absolute_error(mf_vec, ht_vec), 4),
        "per_trait": {
            k: {
                "name": BASIS_NAMES[k],
                "mindform": formed[k],
                "heart_raw_0_1": target_raw.get(LONG[k]),
                "heart_signed": target_signed[k],
                "abs_error": round(abs(formed[k] - target_signed[k]), 4),
                "saturated": abs(formed[k]) >= SATURATION,
            } for k in BASIS
        },
        "direction_agreement": {
            "per_trait": direction_agreement(formed, target_signed),
            "note": "secondary/debug only -- not a validated metric. A target "
                    f"within ±{DIRECTION_DEADBAND} of HEART's own scale midpoint "
                    "has no direction to agree with and scores n/a.",
        },
        "ground_truth_disclosure": (
            "HEART's Big Five was hidden from MindForm for the entire formation "
            "process and read here only after the frozen state above was captured."),
    }


# ---------------------------------------------------------------------------
# the paired Behavioral Fidelity result (HEART), read from real run logs only
# ---------------------------------------------------------------------------
def find_behavioral_result(character_id, protocol_faithful=None):
    """The most recent COMPLETED HEART run for this character, or None.

    Never fabricates or hardcodes a number: every figure here is counted fresh
    from that run's own events.jsonl. A run that aborted (leak, invalid state,
    an unreachable model) is excluded outright -- a benchmark that didn't
    finish has no accuracy to report. When ``protocol_faithful`` is given, only
    runs formed at that tier are considered, so a dev-tier fidelity score is
    never paired with a full-tier behavioral score or vice versa.
    """
    candidates = []
    if not os.path.isdir(RESULTS_ROOT):
        return None
    for run_id in os.listdir(RESULTS_ROOT):
        run_dir = os.path.join(RESULTS_ROOT, run_id)
        events = load_events(run_dir)
        if not events:
            continue
        start = next((e for e in events if e.get("event") == "run_start"), None)
        if not start or start.get("character") != character_id:
            continue
        if start.get("mode") not in ("quick", "character", "full"):
            continue
        if any(e.get("event") in ("run_invalid", "run_error", "leak_detected")
               for e in events):
            continue
        if protocol_faithful is not None and bool(
                start.get("protocol_faithful")) != protocol_faithful:
            continue
        graded = [e for e in events if e.get("event") == "graded"]
        if not graded:
            continue
        mtime = max((e.get("elapsed_s", 0) for e in events), default=0)
        candidates.append((os.path.getmtime(os.path.join(run_dir, "events.jsonl")),
                           run_id, start, graded))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, run_id, start, graded = candidates[0]

    tallies = {}
    for g in graded:
        arm = g.get("arm")
        c, n = tallies.get(arm, (0, 0))
        tallies[arm] = (c + (1 if g.get("correct") else 0), n + 1)

    return {
        "run_id": run_id,
        "mode": start.get("mode"),
        "protocol_faithful": bool(start.get("protocol_faithful")),
        "snapshot_memories": start.get("snapshot_memories"),
        "n_questions": start.get("n_questions"),
        "per_arm": {
            arm: {
                "label": ARM_LABEL.get(arm, arm),
                "correct": tallies.get(arm, (0, 0))[0],
                "total": tallies.get(arm, (0, 0))[1],
                "pct": (round(100.0 * tallies[arm][0] / tallies[arm][1], 1)
                       if tallies.get(arm, (0, 0))[1] else None),
            } for arm in ARM_ORDER if arm in tallies
        },
    }


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------
def render_text(fid, behavioral):
    W = 72
    L = []
    a = L.append
    a("=" * W)
    a(f"PERSONALITY FIDELITY — Did MindForm understand who this person is?")
    a("=" * W)
    a(f"character      {fid['character']}  ({fid['occupation']})")
    a(f"protocol       {fid['protocol']['label']} -- {fid['protocol']['note']}")
    a(f"snapshot       {fid['snapshot']['memories']}/{fid['snapshot']['total_memories']} "
      f"memories, id {fid['snapshot']['snapshot_id'][:16]}…")
    a("")
    ts = fid["trait_similarity"]
    a(f"Trait Similarity:     {ts:+.2f}" if ts is not None else "Trait Similarity:     n/a (zero vector)")
    a(f"Average Trait Error:  {fid['average_trait_error']:.2f}   (scale: 0 = exact, 2 = maximum)")
    a("")
    a(f"{fid['ground_truth_disclosure']}")
    a("")
    a("-" * W)
    a(f"{'trait':<18}{'MindForm':>10}{'HEART target':>14}{'abs err':>10}{'dir':>6}")
    a("-" * W)
    da = fid["direction_agreement"]["per_trait"]
    for k in BASIS:
        r = fid["per_trait"][k]
        mark = {True: "✓", False: "✗", None: "n/a"}[da[k]]
        sat = " (sat)" if r["saturated"] else ""
        a(f"{r['name']:<18}{r['mindform']:>+10.3f}{r['heart_signed']:>+14.3f}"
          f"{r['abs_error']:>10.3f}{mark:>6}{sat}")
    a("-" * W)
    a(f"direction agreement: {fid['direction_agreement']['note']}")
    a("")
    a("=" * W)
    a("BEHAVIORAL FIDELITY (HEART) — Does MindForm act like this person?")
    a("=" * W)
    if not behavioral:
        a("No completed HEART run found for this character/tier yet.")
        a("Run the live UI or bench.heart.runner to produce one.")
    else:
        tier_note = "FULL PROTOCOL" if behavioral["protocol_faithful"] else "DEVELOPMENT TEST"
        a(f"run            {behavioral['run_id']}  ({tier_note}, "
          f"{behavioral['n_questions']} questions)")
        a("")
        for arm in ARM_ORDER:
            row = behavioral["per_arm"].get(arm)
            if not row:
                continue
            a(f"  {row['label']:<16} {row['correct']}/{row['total']} = {row['pct']}%")
    a("=" * W)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# standalone HTML report -- same generated-report pattern as traitdiff.py
# ---------------------------------------------------------------------------
def _fill_css(v):
    w = abs(v) * 50
    side = "left:50%" if v >= 0 else "right:50%"
    return f"{side};width:{w:.1f}%"


def render_html(fid, behavioral, path):
    S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    SD = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
    da = fid["direction_agreement"]["per_trait"]

    def bars():
        out = []
        for i, k in enumerate(BASIS):
            r = fid["per_trait"][k]
            out.append(f"""
        <div class="tr">
          <span class="tl">{r['name']}</span>
          <span class="track">
            <span class="mid"></span>
            <span class="fill" style="--c:var(--s{i+1});{_fill_css(r['mindform'])}"></span>
            <span class="gt" style="left:{50+r['heart_signed']*50:.1f}%"></span>
          </span>
          <span class="num">{r['mindform']:+.3f}</span>
          <span class="num muted">{r['heart_signed']:+.2f}</span>
          <span class="{ {True:'ok',False:'bad',None:'mut'}[da[k]] }">{ {True:'✓',False:'✗',None:'–'}[da[k]] }</span>
        </div>""")
        return "".join(out)

    ts = fid["trait_similarity"]
    ts_s = f"{ts:+.2f}" if ts is not None else "n/a"

    behav_rows = ""
    if behavioral:
        for arm in ARM_ORDER:
            row = behavioral["per_arm"].get(arm)
            if not row:
                continue
            behav_rows += (f'<div class="tr"><span class="tl">{row["label"]}</span>'
                          f'<span class="num" style="flex:1;text-align:right">'
                          f'{row["correct"]}/{row["total"]} = {row["pct"]}%</span></div>')
        behav_note = (f"run {behavioral['run_id']} · "
                     f"{'FULL PROTOCOL' if behavioral['protocol_faithful'] else 'DEVELOPMENT TEST'} · "
                     f"{behavioral['n_questions']} questions")
    else:
        behav_rows = '<p class="mut">No completed HEART run found yet for this character/tier.</p>'
        behav_note = ""

    html = f"""<!doctype html><meta charset="utf-8">
<title>Personality Fidelity · {fid['character']}</title>
<style>
:root{{color-scheme:light;--bg:#f2f1ee;--card:#fff;--ink:#0b0b0b;--sec:#52514e;--mut:#83817a;
 --rule:rgba(11,11,11,.12);--ok:#1a7f4b;--bad:#b3261e;
 --s1:{S[0]};--s2:{S[1]};--s3:{S[2]};--s4:{S[3]};--s5:{S[4]};}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;
 --bg:#111110;--card:#232322;--ink:#fff;--sec:#c3c2b7;--mut:#8f8e86;--rule:rgba(255,255,255,.15);
 --ok:#4ec98a;--bad:#ff8a7a;--s1:{SD[0]};--s2:{SD[1]};--s3:{SD[2]};--s4:{SD[3]};--s5:{SD[4]};}}}}
body{{margin:0;padding:22px;background:var(--bg);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,sans-serif}}
.wrap{{max-width:760px;margin:0 auto;display:flex;flex-direction:column;gap:14px}}
h1{{font-size:19px;margin:0}}
h2{{font-size:15px;margin:0 0 4px}}
.mut{{color:var(--mut)}} .sub{{color:var(--sec);font-size:13px;margin-top:2px}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:16px 18px}}
.stats{{display:flex;gap:22px;margin:10px 0}}
.stat .v{{font-size:26px;font-weight:700}} .stat .k{{font-size:11px;color:var(--mut);
 text-transform:uppercase;letter-spacing:.05em}}
.hdr{{display:flex;gap:8px;font-size:11px;color:var(--mut);text-transform:uppercase;
 letter-spacing:.04em;padding:0 0 6px;border-bottom:1px solid var(--rule)}}
.hdr span{{flex:1}} .hdr span:first-child{{flex:1.4}}
.tr{{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid var(--rule)}}
.tr:last-child{{border-bottom:none}}
.tl{{flex:1.4;font-size:13px}}
.track{{flex:2.2;position:relative;height:10px;background:rgba(128,128,128,.15);
 border-radius:99px;overflow:visible}}
.mid{{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--rule)}}
.fill{{position:absolute;top:0;bottom:0;background:var(--c);border-radius:99px}}
.gt{{position:absolute;top:-4px;width:2px;height:18px;background:var(--ink);
 transform:translateX(-1px)}}
.num{{font:12px/1 ui-monospace,monospace;width:52px;text-align:right}}
.ok{{color:var(--ok)}} .bad{{color:var(--bad)}}
.foot{{font-size:12px;color:var(--mut);margin-top:10px}}
.disclosure{{font-size:12px;color:var(--sec);background:rgba(128,128,128,.08);
 border-radius:8px;padding:8px 10px;margin-top:10px}}
</style>
<div class="wrap">
  <div>
    <h1>PERSONALITY FIDELITY</h1>
    <div class="sub">Did MindForm understand who this person is? — {fid['character']}
      ({fid['occupation']})</div>
  </div>
  <div class="card">
    <div class="stats">
      <div class="stat"><div class="v">{ts_s}</div><div class="k">Trait Similarity</div></div>
      <div class="stat"><div class="v">{fid['average_trait_error']:.2f}</div>
        <div class="k">Average Trait Error</div></div>
    </div>
    <div class="hdr"><span>trait</span><span></span><span>MindForm</span>
      <span>HEART target</span><span>dir</span></div>
    {bars()}
    <p class="foot">direction agreement is secondary/debug only — traits within
      ±{DIRECTION_DEADBAND} of HEART's scale midpoint have no direction to score (–).
      The vertical tick on each bar marks HEART's target.</p>
    <p class="disclosure">{fid['ground_truth_disclosure']}</p>
  </div>
  <div class="card">
    <h2>BEHAVIORAL FIDELITY (HEART)</h2>
    <div class="sub">Does MindForm act like this person? {behav_note}</div>
    <div style="margin-top:8px">{behav_rows}</div>
  </div>
</div>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--character", default="CHAR_01")
    ap.add_argument("--tier", choices=["full", "dev"], default=None,
                    help="require this tier; default: prefer full, fall back to dev")
    ap.add_argument("--save", action="store_true", help="also write an HTML report")
    args = ap.parse_args()
    require_heart_bench()

    try:
        fid = compute(args.character, tier=args.tier)
    except NoSnapshot as exc:
        raise SystemExit(f"\n{exc}\n")
    behavioral = find_behavioral_result(args.character,
                                        protocol_faithful=fid["protocol"]["faithful"])
    print(render_text(fid, behavioral))

    if args.save:
        out_dir = os.path.join(RESULTS_ROOT, "fidelity")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{args.character.lower()}-fidelity.html")
        render_html(fid, behavioral, path)
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
