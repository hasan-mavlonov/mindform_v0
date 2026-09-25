"""Render a BFI-2-S result: the public screen, and the research comparison view.

Three measurements, kept explicitly separate everywhere below (never averaged
together, never presented as if they were the same number):

  1. Internal MindForm OCEAN     -- read directly off the frozen state file
  2. BFI-2-S expressed OCEAN     -- scored from the model's self-report answers
  3. HEART hidden OCEAN target   -- external ground truth, revealed only here,
                                    after both 1 and 2 already exist

(1) vs (3) is exactly bench.heart.fidelity's own comparison, reused rather
than recomputed. (2) vs (3) and (1) vs (2) are new here. The public screen
shows only the BFI-2-S result (2) plus its own facet breakdown; the
comparison table combining all three is a separate, clearly-labeled
research/details section.
"""

from bench.heart import fidelity as heart_fidelity


def trait_match_pct(a, b):
    """The one, defensible public transform: closeness on the -1..+1 scale,
    as a percentage of the maximum possible gap (2). Same formula used
    everywhere a percentage is shown -- see bench.heart.fidelity's own
    decision memo for why this and not an invented "match %"."""
    return round(100.0 * (1.0 - abs(a - b) / 2.0), 1)


def build_comparison(result):
    """Combine a BFI-2 administer() result with HEART's existing Personality
    Fidelity comparison, at the SAME tier the BFI-2 run used -- a dev-tier
    self-report is never compared against a full-tier internal reading or
    vice versa."""
    character = result["character"]
    tier = result["tier"]
    fid = heart_fidelity.compute(character, tier=tier)   # (1) vs (3), already built/tested

    ocean = "OCEAN"
    rows = {}
    for k in ocean:
        internal = fid["per_trait"][k]["mindform"]
        bfi2 = result["scored"]["ocean_signed"][k]
        hidden = fid["per_trait"][k]["heart_signed"]
        rows[k] = {
            "name": fid["per_trait"][k]["name"],
            "internal": internal,
            "bfi2": bfi2,
            "hidden_target": hidden,
            "bfi2_vs_hidden_pct": trait_match_pct(bfi2, hidden),
            "internal_vs_hidden_pct": trait_match_pct(internal, hidden),
            "internal_vs_bfi2_pct": trait_match_pct(internal, bfi2),
        }

    def mae(key_a, key_b):
        return sum(abs(rows[k][key_a] - rows[k][key_b]) for k in ocean) / len(ocean)

    return {
        "character": character,
        "tier": tier,
        "protocol_note": fid["protocol"]["note"],
        "rows": rows,
        "summary": {
            "bfi2_vs_hidden_overall_pct": round(
                100.0 * (1.0 - mae("bfi2", "hidden_target") / 2.0), 1),
            "internal_vs_hidden_overall_pct": round(
                100.0 * (1.0 - mae("internal", "hidden_target") / 2.0), 1),
            "internal_vs_bfi2_overall_pct": round(
                100.0 * (1.0 - mae("internal", "bfi2") / 2.0), 1),
        },
        "ground_truth_disclosure": fid["ground_truth_disclosure"],
    }


def render_text(result, comparison):
    W = 76
    L = []
    a = L.append
    inst = result["instrument"]
    scored = result["scored"]
    # Official domain order, taken from the instrument itself -- OCEAN letter
    # order is used only in the comparison table below, never here, so the
    # public listing always matches the instrument's own published structure.
    order = list(scored["domain_scores"].keys())
    facets_by_domain = {}
    for row in result["items"]:
        facets_by_domain.setdefault(row["domain"], set()).add(row["facet"])

    a("=" * W)
    a(f"{inst['name']} PERSONALITY TEST — {result['character']} ({result['occupation']})")
    a("=" * W)
    a(f"tier            {result['tier'].upper()}  ({result['snapshot']['memories']} memories, "
      f"snapshot {result['snapshot']['snapshot_id'][:16]}…)")
    a(f"instrument      {inst['full_name']}")
    a(f"license         {inst['license_notice']}")
    a("")
    a("DOMAIN SCORES (self-report, this instrument's own 1-5 scale)")
    a("-" * W)
    for d in order:
        bar_len = round((scored["domain_scores"][d] - 1) / 4 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        a(f"  {d:<24}{bar}  {scored['domain_scores'][d]:.2f} / 5")
    a("")
    if inst["facets_are_exploratory"]:
        a(f"Facets below are EXPLORATORY, descriptive output only — "
          f"{inst['reliability_note']}")
        a("")
    for d in order:
        a(f"{d}")
        for f in sorted(facets_by_domain.get(d, ())):
            a(f"  {f:<24}{scored['facet_scores'][f]:.2f} / 5")
    a("")
    a("=" * W)
    a("COMPARISON (research/details — three separate measurements, not one)")
    a("=" * W)
    a(comparison["protocol_note"])
    a("")
    a(f"{'trait':<20}{'internal':>10}{'bfi2-s':>10}{'hidden':>10}"
      f"{'bfi2 vs hidden':>16}")
    a("-" * W)
    for k in "OCEAN":
        r = comparison["rows"][k]
        a(f"{r['name']:<20}{r['internal']:>+10.3f}{r['bfi2']:>+10.3f}"
          f"{r['hidden_target']:>+10.3f}{r['bfi2_vs_hidden_pct']:>15.1f}%")
    a("-" * W)
    s = comparison["summary"]
    a(f"Overall: BFI-2-S vs hidden target {s['bfi2_vs_hidden_overall_pct']}%  |  "
      f"internal vs hidden target {s['internal_vs_hidden_overall_pct']}%  |  "
      f"internal vs BFI-2-S self-report {s['internal_vs_bfi2_overall_pct']}%")
    a("")
    a(comparison["ground_truth_disclosure"])
    a("=" * W)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# standalone HTML report -- same generated-report pattern as fidelity.py
# ---------------------------------------------------------------------------
def render_html(result, comparison, path):
    S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    inst = result["instrument"]
    scored = result["scored"]
    order = list(scored["domain_scores"].keys())
    facets_by_domain = {}
    for row in result["items"]:
        facets_by_domain.setdefault(row["domain"], set()).add(row["facet"])

    def domain_bar(d, i):
        pct = (scored["domain_scores"][d] - 1) / 4 * 100
        return f"""
      <div class="tr">
        <span class="tl">{d}</span>
        <span class="track"><span class="fill" style="--c:var(--s{i+1});width:{pct:.1f}%"></span></span>
        <span class="num">{scored["domain_scores"][d]:.2f} / 5</span>
      </div>"""

    def facet_rows(d):
        out = []
        for f in sorted(facets_by_domain.get(d, ())):
            pct = (scored["facet_scores"][f] - 1) / 4 * 100
            out.append(f"""
        <div class="tr sub">
          <span class="tl">{f}</span>
          <span class="track small"><span class="fill mut" style="width:{pct:.1f}%"></span></span>
          <span class="num">{scored['facet_scores'][f]:.2f}</span>
        </div>""")
        return "".join(out)

    domain_html = "".join(domain_bar(d, i) for i, d in enumerate(order))
    facet_html = "".join(f'<div class="facet-group"><h3>{d}</h3>{facet_rows(d)}</div>'
                         for d in order)

    comp_rows = ""
    for k in "OCEAN":
        r = comparison["rows"][k]
        comp_rows += (f'<div class="tr"><span class="tl">{r["name"]}</span>'
                     f'<span class="num">{r["internal"]:+.3f}</span>'
                     f'<span class="num">{r["bfi2"]:+.3f}</span>'
                     f'<span class="num">{r["hidden_target"]:+.3f}</span>'
                     f'<span class="num">{r["bfi2_vs_hidden_pct"]:.0f}%</span></div>')

    s = comparison["summary"]
    html = f"""<!doctype html><meta charset="utf-8">
<title>{inst['name']} · {result['character']}</title>
<style>
:root{{color-scheme:light;--bg:#f2f1ee;--card:#fff;--ink:#0b0b0b;--sec:#52514e;--mut:#83817a;
 --rule:rgba(11,11,11,.12);--s1:{S[0]};--s2:{S[1]};--s3:{S[2]};--s4:{S[3]};--s5:{S[4]};}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;
 --bg:#111110;--card:#232322;--ink:#fff;--sec:#c3c2b7;--mut:#8f8e86;--rule:rgba(255,255,255,.15);}}}}
body{{margin:0;padding:22px;background:var(--bg);color:var(--ink);
 font:14px/1.55 system-ui,-apple-system,sans-serif}}
.wrap{{max-width:760px;margin:0 auto;display:flex;flex-direction:column;gap:14px}}
h1{{font-size:19px;margin:0}} h2{{font-size:15px;margin:0 0 4px}}
h3{{font-size:12px;margin:14px 0 4px;color:var(--sec);text-transform:uppercase;letter-spacing:.04em}}
.mut{{color:var(--mut)}} .sub{{color:var(--sec);font-size:13px;margin-top:2px}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:16px 18px}}
.tr{{display:flex;align-items:center;gap:10px;padding:6px 0}}
.tr.sub{{padding:3px 0}}
.tl{{flex:1.6;font-size:13px}} .tr.sub .tl{{font-size:12px;color:var(--sec)}}
.track{{flex:3;height:11px;background:rgba(128,128,128,.15);border-radius:99px;overflow:hidden}}
.track.small{{height:6px;flex:2.4}}
.fill{{display:block;height:100%;background:var(--c);border-radius:99px}}
.fill.mut{{background:var(--mut)}}
.num{{font:12px/1 ui-monospace,monospace;width:70px;text-align:right;flex:none}}
.notice{{font-size:11.5px;color:var(--mut);background:rgba(128,128,128,.08);
 border-radius:8px;padding:8px 10px;margin-top:8px}}
details{{margin-top:4px}} summary{{cursor:pointer;font-size:13px;color:var(--sec);padding:4px 0}}
.hdr{{display:flex;gap:10px;font-size:11px;color:var(--mut);text-transform:uppercase;
 letter-spacing:.04em;padding:2px 0 6px}}
.hdr span{{flex:1;text-align:right}} .hdr span:first-child{{flex:1.6;text-align:left}}
.facet-group:first-child h3{{margin-top:2px}}
</style>
<div class="wrap">
  <div>
    <h1>{inst['name']} PERSONALITY TEST</h1>
    <div class="sub">What personality did MindForm form? — {result['character']}
      ({result['occupation']})</div>
  </div>
  <div class="card">
    {domain_html}
    <details>
      <summary>View {sum(len(v) for v in facets_by_domain.values())} facets ↓</summary>
      {facet_html}
      <p class="notice">{inst['reliability_note']}</p>
    </details>
    <p class="notice">{inst['license_notice']}</p>
  </div>
  <details class="card">
    <summary><b>Research / details</b> — Internal MindForm, {inst['name']} self-report,
      and HEART's hidden target are three separate measurements, shown separately.</summary>
    <p class="sub" style="margin-top:10px">{comparison['protocol_note']}</p>
    <div class="hdr"><span>trait</span><span>internal</span><span>{inst['name'].lower()}</span>
      <span>hidden target</span><span>match</span></div>
    {comp_rows}
    <p class="notice">Overall: {inst['name']} vs hidden target
      <b>{s['bfi2_vs_hidden_overall_pct']}%</b> · internal vs hidden target
      <b>{s['internal_vs_hidden_overall_pct']}%</b> · internal vs {inst['name']} self-report
      <b>{s['internal_vs_bfi2_overall_pct']}%</b></p>
    <p class="notice">{comparison['ground_truth_disclosure']}</p>
  </details>
</div>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
