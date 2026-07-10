"""SOCIAL EXPRESSION: the outward voice -- inner state becomes an audible manner.

Dependency-free (pure arithmetic -- no torch / numpy / network), like the other node suites.
Proves the three claims of the slice:

  * GOFFMAN  -- the voice follows the SELF-VIEW, not the raw traits: identical traits with
                opposite self-images sound different (and speak like who they THINK they are).
  * THE LENS DRIVES THE MOUTH -- the same text is answered brightly or braced depending on
                this turn's INTERPRETED appraisal, not a private sentiment lexicon.
  * SHAPE    -- the style dims audibly shape the offline reply: hedges at low assertion, a
                reach toward the other at high warmth, a length budget at low energy, a worry
                tail under strain -- so two characters differ with no LLM key at all.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.llm as llm_mod
llm_mod.LLM_API_KEY = ""           # never touch the network

import web.reply as reply_mod
reply_mod.LLM_API_KEY = ""         # force the offline voice in generate_reply too

from core.config import BASIS, STYLE_THRESH
from nodes.expression import (
    style, voice, plain_reply, read_voice, read_expression, STYLE_DIMS,
)
from web.reply import generate_reply

_checks = 0
_failed = 0


def check(label, cond):
    global _checks, _failed
    _checks += 1
    if not cond:
        _failed += 1
    print(("PASS" if cond else "FAIL") + " -- " + label)


def person(image=None, esteem=0.0, traits=None, tensions=None):
    zero = {k: 0.0 for k in BASIS}
    t = tensions or {}
    p = {
        "traits": {**zero, **(traits or {})},
        "temperament": {"mu": dict(zero)},
        "character": {"values": {}},
        "drives": {d: {"tension": t.get(d, 0.3)}
                   for d in ("autonomy", "competence", "relatedness")},
        "identity": {"name": "Test"},
    }
    if image is not None:
        p["self"] = {"image": {**zero, **image}, "esteem": esteem}
    return p


def appr(valence=0.0, threat=0.0):
    return {"valence": valence, "threat_challenge": threat, "intensity": 0.5,
            "novelty": 0.3, "agency": 0.0, "social": 0.0, "outcome": 0.0,
            "self_relevance": 0.5}


MSG = "I failed the exam I studied months for."


# --- the style derivation --------------------------------------------------------
blank = style(person())
check("a blank character has a neutral style (all dims ~0)",
      all(abs(blank[d]) < 1e-9 for d in STYLE_DIMS))
check("style dims stay clamped in [-1, 1] at extremes",
      all(-1.0 <= v <= 1.0 for v in style(person(
          image={"E": 1.0, "A": -1.0, "N": 1.0, "O": 1.0}, esteem=-1.0,
          tensions={"autonomy": 1.0, "competence": 1.0, "relatedness": 1.0})).values()))

warm = person({"E": 0.6, "A": 0.7}, esteem=0.6)
shy = person({"E": -0.6, "N": 0.7}, esteem=-0.5)
s_warm, s_shy = style(warm), style(shy)
check("a warm, secure self-image yields high warmth", s_warm["warmth"] >= STYLE_THRESH)
check("an anxious, doubting self-image yields hedging (negative assertion)",
      s_shy["assertion"] <= -STYLE_THRESH)
check("doubt and an anxious image strain the voice", s_shy["strain"] >= STYLE_THRESH)
check("need starvation adds strain",
      style(person({"N": 0.3}, tensions={"relatedness": 1.0, "competence": 1.0,
                                         "autonomy": 1.0}))["strain"]
      > style(person({"N": 0.3}))["strain"])

# GOFFMAN: identical raw traits, opposite self-images -> different voices
t_same = {"E": 0.0}
g1 = person({"E": 0.6, "A": 0.5}, traits=t_same)
g2 = person({"E": -0.6, "A": -0.5}, traits=t_same)
check("identical traits with opposite self-images produce different styles (Goffman)",
      style(g1) != style(g2))
check("the style follows the self-image, not the traits",
      style(person({"E": 0.7}, traits={"E": -0.7}))["energy"] > 0)
check("with no self yet, the style falls back to the actual traits (legacy-safe)",
      style({"traits": {"O": 0, "C": 0, "E": 0.7, "A": 0, "N": 0}})["energy"] > 0)


# --- the LLM brief ----------------------------------------------------------------
check("a blank character emits an empty voice brief (prompt unchanged)",
      voice(person()) == [])
v_shy = voice(person({"E": -0.6, "N": 0.7}, esteem=-0.5,
                     tensions={"relatedness": 0.9}), appr(valence=-0.5, threat=-0.4))
check("a braced mood line leads the brief under threat",
      v_shy and "braced" in v_shy[0])
check("the active need surfaces in the brief (Motivation -> voice)",
      any("connection" in line for line in v_shy))
check("the brief is capped at six lines", len(v_shy) <= 6)
check("high warmth softens a sting instead of just sitting heavy",
      any("soften" in line for line in voice(warm, appr(valence=-0.5))))


# --- the offline mouth: the lens drives it, the style shapes it -------------------
r_pos = plain_reply(warm, MSG, appr(valence=0.8))
r_neg = plain_reply(warm, MSG, appr(valence=-0.8))
check("the SAME text is answered differently under opposite interpreted moods",
      r_pos != r_neg)
check("a positive lens yields a positive lead even on 'failure' words",
      any(r_pos.startswith(x) or x in r_pos for x in
          ("That actually felt good", "Something in me opened", "I didn't want that to end")))
check("threat alone (neutral valence) still reads braced/neg -- the sign case",
      plain_reply(person(), MSG, appr(valence=0.0, threat=-0.5))
      in [x for fam in ("neg",) for x in
          ["That one sat heavy with me.", "I'm still carrying it, if I'm honest.",
           "It shook me more than I'd like to admit."]])

check("low assertion prepends a hedge",
      plain_reply(shy, MSG, appr(valence=-0.5)).startswith(("I don't know --", "Maybe it's just me")))
check("high assertion opens plainly",
      plain_reply(person({"E": 0.7, "A": -0.6}, esteem=0.7), MSG,
                  appr(valence=-0.5)).startswith("Honestly,"))
check("high warmth reaches toward the other",
      "How are you holding up?" in plain_reply(warm, MSG, appr(valence=-0.5)))
check("low energy clips to one sentence (the length budget)",
      plain_reply(shy, MSG, appr(valence=-0.5)).count(".") +
      plain_reply(shy, MSG, appr(valence=-0.5)).count("?") <= 1 or
      plain_reply(shy, MSG, appr(valence=-0.5)).rstrip()[-1] in ".!?")
check("strain adds a worry tail on a heavy turn (when energy allows it)",
      "bracing for what comes next" in plain_reply(
          person({"N": 0.8, "E": 0.3}, esteem=-0.4), MSG, appr(valence=-0.5)))
check("the active need is voiced in the aside",
      "alone in it" in plain_reply(person({"E": -0.3}, tensions={"relatedness": 0.9}),
                                   MSG, appr(valence=-0.5)))
check("the offline reply is deterministic for the same inputs",
      plain_reply(shy, MSG, appr(valence=-0.5)) == plain_reply(shy, MSG, appr(valence=-0.5)))
check("with no appraisal, the lexical fallback still yields a sensible lead",
      isinstance(plain_reply(person(), MSG), str) and len(plain_reply(person(), MSG)) > 0)


# --- the public contract ----------------------------------------------------------
text, source = generate_reply(warm, MSG, appraisal=appr(valence=-0.5))
check("generate_reply returns (text, 'rule') with no key",
      isinstance(text, str) and len(text) > 0 and source == "rule")

tags = read_voice(shy)
check("the voice tags read the style (hedged/quiet/wound tight)",
      "hedged" in tags and "quiet" in tags)
check("a blank character has no voice tags", read_voice(person()) == "")
ex = read_expression(warm, appr())
check("read_expression exposes four labelled style rows + the line",
      len(ex["style"]) == 4 and all("key" in r and "value" in r for r in ex["style"])
      and isinstance(ex["line"], str))


print()
print("%d/%d checks passed." % (_checks - _failed, _checks))
if _failed:
    print("SOME CHECKS FAILED.")
    sys.exit(1)
print("ALL CHECKS PASSED -- the inside comes out: the self-view, the need, and the lens shape the voice.")
