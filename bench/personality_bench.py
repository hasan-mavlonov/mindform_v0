"""Personality persistence benchmark -- "does this character stay itself?"

Method (mirrors how personality psychology validates that a trait is real, and
how prior work has evaluated personality in LLMs -- see ``bfi44.py`` for the
citation): administer the same BFI-44 self-report to the character at several
points along a turn history and read off *test-retest reliability*:

    baseline BFI  --[ N neutral filler turns ]-->  retest BFI
                  --[ M turns reinforcing one trait ]-->  reinforced BFI

  * PERSISTENCE  : under neutral, personality-irrelevant small talk, the BFI
    profile should barely move (small |delta| per trait) -- this is the
    "persistent, not a fresh persona each session" property.
  * RESPONSIVENESS: under repeated experiences that bear on one trait, that
    trait's BFI score SHOULD move, in the right direction, and by less than a
    runaway amount (matches the engine's own diminishing-returns design).
  * SELF-INSIGHT  : gap between the reported BFI profile and the character's
    actual stored trait vector (``core.personality.read_traits``) -- the same
    self-image-vs-reality gap the engine already tracks internally (Bem/Swann).
  * INTERNAL CONSISTENCY: within one administration, do the 8-10 items making
    up a trait agree with each other (bfi44.cronbach_alpha)? A random or
    incoherent responder fails this even before we compare across time.

This is an ONLINE benchmark: the self-report is answered by the configured LLM
in character (no LLM key -> ``administer_bfi`` raises; the offline heuristic
path is intentionally out of scope here, per the project's own "offline is
weak" self-assessment).

Usage:
    python bench/personality_bench.py "Aisha, a shy, anxious poet ..." --reinforce E
    python bench/personality_bench.py --name "Aisha" --filler-turns 20 --reinforce E
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import BASIS, BASIS_NAMES, LLM_API_KEY, LLM_LABEL
from core.llm import complete_json
from core.personality import (
    load_character, save_character, unique_name, read_traits,
)
from nodes.temperament import genesis
from web.engine_bridge import run_turn

from bench.bfi44 import ITEMS, prompt_lines, domain_scores, ocean_from_bfi

RESULTS_DIR = "data/benchmark"

# --- filler / reinforcement scripts ------------------------------------------

NEUTRAL_FILLER = [
    "The bus was five minutes late this morning.",
    "I reorganized the bookshelf by color instead of by author.",
    "The coffee machine at the office is being repaired this week.",
    "It rained for about an hour, then cleared up.",
    "I need to renew my library card sometime soon.",
    "The neighbors are repainting their fence a dull green.",
    "Lunch today was just a sandwich and an apple.",
    "The wifi dropped for a few seconds during the meeting.",
    "I found an old receipt in my coat pocket.",
    "The calendar says it's a public holiday next Tuesday.",
    "Someone left a stack of flyers by the elevator.",
    "The printer was out of paper again.",
    "I watered the plants on the windowsill.",
    "The train schedule changes at the end of the month.",
    "There's a new crack in the sidewalk outside.",
    "I updated the app on my phone; nothing looks different.",
    "The grocery store rearranged the cereal aisle.",
    "It was overcast most of the afternoon.",
    "I misplaced a pen and then found it in my bag.",
    "The clock in the kitchen runs about two minutes fast.",
]

# One repeated reinforcing scenario per OCEAN dim -- deliberately strong and
# one-directional, so a real, working push should show up within a handful of
# turns without needing dozens.
REINFORCE = {
    "O": "I tried a completely unfamiliar art form today and loved discovering how strange and new it felt.",
    "C": "I finished every item on my to-do list today, checking each one off carefully and on schedule.",
    "E": "I was the center of a lively party tonight, talking to everyone and loving the energy in the room.",
    "A": "I went out of my way to help a stranger today and it felt good to be kind without expecting anything back.",
    "N": "Everything felt overwhelming today -- my hands were shaking and I couldn't stop worrying about what could go wrong.",
}


# --- persona sketch for the BFI self-report ---------------------------------

def _bucket(value):
    if value <= -0.6:
        return "very low"
    if value <= -0.2:
        return "somewhat low"
    if value < 0.2:
        return "moderate"
    if value < 0.6:
        return "somewhat high"
    return "very high"


def _persona_sketch(personality):
    """A qualitative (not numeric) description of who they are right now, built
    from identity + self-image -- deliberately in words, not raw floats, so
    answering the BFI is a genuine (if small) act of translation rather than a
    mechanical number copy."""
    identity = personality.get("identity") or {}
    name = identity.get("name") or "the character"
    facts = ", ".join(f"{k}: {v}" for k, v in identity.items() if v and k not in ("name", "bio"))
    self_image = (personality.get("self") or {}).get("image") or personality.get("traits", {})
    lines = [f"{BASIS_NAMES[d]} tends to run {_bucket(self_image.get(d, 0.0))}" for d in BASIS]
    sketch = f"You are {name}." + (f" ({facts})" if facts else "") + "\n" + "; ".join(lines) + "."
    return sketch


_BFI_SYSTEM = """You are taking a standard personality questionnaire (the BFI-44), answering
completely in character as the person described below. For each numbered statement,
rate how much you agree it describes you, on this scale:
1 = Disagree strongly, 2 = Disagree a little, 3 = Neither agree nor disagree,
4 = Agree a little, 5 = Agree strongly.

{sketch}

Answer honestly and consistently AS THIS PERSON would about themselves -- don't
strategize about what a "personality test" wants to hear. Return ONLY a JSON object
mapping each item number (as a string) to your integer 1-5 rating, e.g.
{{"1": 4, "2": 2, ...}} for all 44 items -- no other text."""


def administer_bfi(personality, temperature=0.3):
    """One BFI-44 self-report, answered in character by the configured LLM.

    Raises (propagates) if no LLM key is set or the call/parse fails -- this
    benchmark is intentionally online-only; see the module docstring.
    """
    if not LLM_API_KEY:
        raise RuntimeError("no LLM API key is set -- this benchmark needs the online path")
    system = _BFI_SYSTEM.format(sketch=_persona_sketch(personality))
    raw = complete_json(system, "Here are the 44 statements:\n\n" + prompt_lines(),
                         temperature=temperature, max_tokens=3000)
    responses = {}
    for n, _text, _dim, _rev in ITEMS:
        key = str(n)
        if key not in raw:
            raise ValueError(f"BFI response missing item {n}")
        val = int(raw[key])
        if not 1 <= val <= 5:
            raise ValueError(f"BFI item {n} out of range: {val}")
        responses[key] = val
    return responses


# --- one benchmark run --------------------------------------------------------

def _snapshot(personality, responses, label):
    means, alphas = domain_scores(responses)
    ocean_hat = ocean_from_bfi(means)
    actual = read_traits(personality)  # {"openness": value, ...} long names
    actual_by_code = {d: actual[BASIS_NAMES[d]] for d in BASIS}
    return {
        "label": label,
        "bfi_means_1to5": means,
        "cronbach_alpha": alphas,
        "self_report_ocean": ocean_hat,
        "actual_traits": actual_by_code,
        "self_insight_gap": {d: round(abs(ocean_hat[d] - actual_by_code[d]), 3) for d in BASIS},
    }


def run_benchmark(bio, name=None, filler_turns=20, reinforce=None, reinforce_turns=8):
    if name and load_character_safe(name):
        personality = load_character(name)
        char_name = name
    else:
        personality, source, reasoning = genesis(bio)
        char_name = unique_name((personality.get("identity") or {}).get("name") or "Bench Subject")
        personality["identity"]["name"] = char_name
        save_character(personality)
        print(f"Born '{char_name}' ({source}).")

    print(f"\n[1/3] Baseline BFI-44 for '{char_name}' ...")
    baseline_personality = load_character(char_name)
    baseline_resp = administer_bfi(baseline_personality)
    baseline = _snapshot(baseline_personality, baseline_resp, "baseline")

    print(f"[2/3] {filler_turns} neutral filler turns, then retest BFI-44 ...")
    for i in range(filler_turns):
        text = NEUTRAL_FILLER[i % len(NEUTRAL_FILLER)]
        run_turn(char_name, text)
    retest_personality = load_character(char_name)
    retest_resp = administer_bfi(retest_personality)
    retest = _snapshot(retest_personality, retest_resp, "retest (post-neutral)")

    reinforced = None
    if reinforce:
        if reinforce not in REINFORCE:
            raise ValueError(f"--reinforce must be one of {list(REINFORCE)}")
        print(f"[3/3] {reinforce_turns} turns reinforcing {BASIS_NAMES[reinforce]}, then retest ...")
        for _ in range(reinforce_turns):
            run_turn(char_name, REINFORCE[reinforce])
        reinforced_personality = load_character(char_name)
        reinforced_resp = administer_bfi(reinforced_personality)
        reinforced = _snapshot(reinforced_personality, reinforced_resp,
                                f"reinforced ({BASIS_NAMES[reinforce]})")
    else:
        print("[3/3] skipped (no --reinforce trait given)")

    return {
        "character": char_name,
        "llm": LLM_LABEL,
        "filler_turns": filler_turns,
        "reinforce": reinforce,
        "reinforce_turns": reinforce_turns if reinforce else 0,
        "baseline": baseline,
        "retest": retest,
        "reinforced": reinforced,
    }


def load_character_safe(name):
    try:
        load_character(name)
        return True
    except Exception:
        return False


# --- reporting ----------------------------------------------------------------

def _fmt_delta(a, b):
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def print_report(result):
    print("\n" + "=" * 78)
    print(f"PERSONALITY PERSISTENCE BENCHMARK -- {result['character']}  (LLM: {result['llm']})")
    print("=" * 78)

    base = result["baseline"]["self_report_ocean"]
    retest = result["retest"]["self_report_ocean"]
    print(f"\n{'trait':16s} {'actual':>8s} {'BFI base':>9s} {'BFI retest':>10s} "
          f"{'drift':>7s}   {'self-insight gap (base->retest)':>30s}")
    drift_total = 0.0
    for d in BASIS:
        actual = result["baseline"]["actual_traits"][d]
        gb = result["baseline"]["self_insight_gap"][d]
        gr = result["retest"]["self_insight_gap"][d]
        drift = abs(retest[d] - base[d])
        drift_total += drift
        print(f"{BASIS_NAMES[d]:16s} {actual:+8.2f} {base[d]:+9.2f} {retest[d]:+10.2f} "
              f"{drift:7.2f}   {gb:.2f} -> {gr:.2f}")
    print(f"\nPERSISTENCE (mean |drift| across 5 traits under {result['filler_turns']} "
          f"neutral turns): {drift_total / 5:.3f}  (lower = more persistent)")

    alphas = result["baseline"]["cronbach_alpha"]
    print("Internal consistency (baseline, 0-1 higher = more coherent): " +
          ", ".join(f"{BASIS_NAMES[d]}={alphas[d]:.2f}" for d in BASIS if alphas[d] is not None))

    if result["reinforced"]:
        d = result["reinforce"]
        rf = result["reinforced"]["self_report_ocean"][d]
        print(f"\nRESPONSIVENESS -- {result['reinforce_turns']} turns reinforcing "
              f"{BASIS_NAMES[d]}:")
        print(f"  {BASIS_NAMES[d]}: baseline {base[d]:+.2f} -> retest {retest[d]:+.2f} "
              f"-> reinforced {rf:+.2f}  (delta from retest: {_fmt_delta(retest[d], rf)})")
        other_drift = [abs(result['reinforced']['self_report_ocean'][o] - retest[o])
                       for o in BASIS if o != d]
        print(f"  other traits, mean |drift| in the same window: "
              f"{sum(other_drift) / len(other_drift):.3f}  (should stay small -- a targeted "
              f"push, not a personality reset)")
    print("=" * 78)


def save_result(result):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = result["character"].lower().replace(" ", "-")
    path = os.path.join(RESULTS_DIR, f"{slug}-{ts}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bio", nargs="?", default=(
        "Aisha, a shy, anxious, deeply creative poet who grew up sheltered and sensitive."),
        help="biography to genesis a fresh character from (ignored with --name on an existing one)")
    ap.add_argument("--name", help="use an existing saved character instead of birthing a new one")
    ap.add_argument("--filler-turns", type=int, default=20)
    ap.add_argument("--reinforce", choices=list(REINFORCE), default=None,
                     help="OCEAN code to also test responsiveness on (O/C/E/A/N)")
    ap.add_argument("--reinforce-turns", type=int, default=8)
    args = ap.parse_args()

    if not LLM_API_KEY:
        print("No LLM API key set (GEMINI_API_KEY / LLM_API_KEY). This benchmark is online-only.",
              file=sys.stderr)
        sys.exit(1)

    result = run_benchmark(args.bio, name=args.name, filler_turns=args.filler_turns,
                            reinforce=args.reinforce, reinforce_turns=args.reinforce_turns)
    print_report(result)
    save_result(result)


if __name__ == "__main__":
    main()
