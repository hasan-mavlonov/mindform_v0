"""BEHAVIOR & LIFE OUTCOME: the enacted stance -- the loop, closed.

Dependency-free (pure arithmetic -- no torch / numpy / network; one optional memory-stamp
check skips without numpy). Proves the four claims of the slice:

  * TWO SYSTEMS -- approach (BAS) and inhibition (BIS) are independent formed sensitivities
                   anchored to the traits, so conflict (both loud) and apathy (both quiet)
                   exist, and dispositions are earned around trait set-points.
  * THE GATE    -- the carried stance scales how hard the next experience lands (intensity),
                   leaning in exposing, holding back muffling -- exactly attributable.
  * THE CREDIT  -- the world's answer trains the systems by the law of effect done right:
                   rewarded OWN action teaches approach hardest, threat trains inhibition
                   faster than mastery extinguishes it, kindness never entrenches
                   withdrawal, and a carried lean-in earns reception credit.
  * BOUNDED     -- the spiral damps (shrinking increments), a bright streak recovers a
                   withdrawn character (no lock-in), and everything stays in bounds.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.llm as llm_mod
llm_mod.LLM_API_KEY = ""           # never touch the network

from core.config import BEHAV_ACTIVE_THRESH, BEHAV_EXPOSURE
import nodes.behavior as behavior
from nodes.cognition import interpret, lens
from nodes.expression import plain_reply, voice
from core.personality import default_personality, migrate

_checks = 0
_failed = 0


def check(label, cond):
    global _checks, _failed
    _checks += 1
    if not cond:
        _failed += 1
    print(("PASS" if cond else "FAIL") + " -- " + label)


def person(E=0.0, N=0.0, O=0.0):
    p = default_personality()
    for k, v in (("E", E), ("N", N), ("O", O)):
        p["traits"][k] = v
        p["temperament"]["mu"][k] = v
    p["behavior"] = behavior.default_behavior(p)
    return p


def appr(v=0.0, tc=0.0, inten=0.6, ag=0.0, out=0.0, s=0.0):
    return {"valence": v, "threat_challenge": tc, "intensity": inten, "agency": ag,
            "outcome": out, "social": s, "self_relevance": 0.5, "novelty": 0.3}


def carry(p, tendency, mode, as_act=True):
    """Set the carried stance (and optionally the frozen act) on a personality."""
    p["behavior"]["set"] = {"tendency": tendency, "mode": mode}
    if as_act:
        p["behavior"]["last"] = {"tendency": tendency, "mode": mode, "style": {}, "turn": 0}
    return p


# --- birth & seeding -------------------------------------------------------------
bold, shy = person(E=0.6, N=-0.3), person(E=-0.6, N=0.7)
check("extraversion seeds an approach disposition", bold["behavior"]["approach"] > 0.2)
check("neuroticism seeds an inhibition disposition", shy["behavior"]["inhibition"] > 0.3)
check("a newborn's sensitivities sit AT their set-points (divergence is earned)",
      bold["behavior"]["approach"] == behavior.set_points(bold)["approach"])
check("a newborn carries no act and a steady stance",
      bold["behavior"]["last"] is None and bold["behavior"]["set"]["mode"] == "steady")
legacy = migrate({"traits": {"O": 0, "C": 0, "E": 0.5, "A": 0, "N": 0}})
check("migrate backfills behavior onto a pre-behavior save at its set-points",
      "behavior" in legacy and abs(legacy["behavior"]["approach"] - 0.275) < 1e-9)

# refresh: a pushed-away sensitivity relaxes back toward its set-point from both sides
hi = person(E=0.6); hi["behavior"]["approach"] = 0.9
lo = person(E=0.6); lo["behavior"]["approach"] = -0.5
check("refresh relaxes a sensitivity toward its set-point from both sides",
      behavior.refresh(hi)["behavior"]["approach"] < 0.9
      and behavior.refresh(lo)["behavior"]["approach"] > -0.5)


# --- the intake gate ---------------------------------------------------------------
lean = carry(person(), 0.8, "approach")
hold = carry(person(), -0.8, "withdraw")
a_raw = appr(inten=0.6)
check("leaning in makes the same event land harder (intensity up)",
      interpret(a_raw, lean)["intensity"] > 0.6)
check("holding back muffles the same event (intensity down)",
      interpret(a_raw, hold)["intensity"] < 0.6)
check("a steady stance leaves intensity untouched",
      interpret(a_raw, person())["intensity"] == 0.6)
check("the gate factor equals the interpreted/raw intensity ratio (exact attribution)",
      abs(interpret(a_raw, hold)["intensity"] / 0.6 - behavior.intake(hold)) < 1e-9)
check("the gate is bounded", 1 - BEHAV_EXPOSURE <= behavior.intake(hold) <= 1 + BEHAV_EXPOSURE
      and 0.0 <= interpret(appr(inten=1.0), lean)["intensity"] <= 1.0)


# --- operant credit: the law of effect, done right ----------------------------------
base = person()
own = behavior.apply_event(base, appr(v=0.5, out=0.8, ag=0.8))["behavior"]["approach"]
windfall = behavior.apply_event(base, appr(v=0.5, out=0.8, ag=0.0))["behavior"]["approach"]
check("rewarded OWN action teaches approach hardest (contingency gate)", own > windfall > 0)
inh_threat = behavior.apply_event(base, appr(v=-0.4, tc=-0.7))["behavior"]["inhibition"]
inh_mastery = behavior.apply_event(base, appr(v=0.6, tc=0.7))["behavior"]["inhibition"]
check("threat trains inhibition; safe mastered challenge extinguishes it",
      inh_threat > 0 and inh_mastery < 0)
check("threat trains harder than mastery extinguishes (bad is stronger)",
      inh_threat > -inh_mastery)

# kindness must NOT entrench withdrawal: no signed stance*reception product
w = carry(person(), -0.7, "withdraw")
w_after = behavior.apply_event(w, appr(v=0.6, s=0.8))["behavior"]["approach"]
check("a warm world never deepens a withdrawn character's withdrawal",
      w_after >= w["behavior"]["approach"])

# reception: None until a first act; credit only on a carried lean-IN
first = behavior.apply_event(person(), appr(v=0.7, s=0.8))["behavior"]
check("no act yet -> no reception reading", first["reception"] is None)
leaner = carry(person(), 0.7, "approach")
met = behavior.apply_event(leaner, appr(v=0.7, s=0.8))["behavior"]
check("a carried lean-in met warmly reads a positive reception", met["reception"] > 0.3)
no_credit = behavior.apply_event(carry(person(), -0.7, "withdraw"),
                                 appr(v=0.7, s=0.8))["behavior"]["approach"]
with_credit = behavior.apply_event(carry(person(), 0.7, "approach"),
                                   appr(v=0.7, s=0.8))["behavior"]["approach"]
check("only the carried lean-in earns the reception credit", with_credit > no_credit)


# --- bounded loop: the spiral damps, the bright streak recovers ---------------------
p, prev, increments = person(E=-0.6, N=0.7), None, []
prev = p["behavior"]["inhibition"]
for _ in range(5):
    p = behavior.refresh(p)
    gated = min(1.0, 0.7 * behavior.intake(p))       # the gate feeding back into training
    p = behavior.apply_event(p, appr(v=-0.4, tc=-0.6, inten=gated))
    increments.append(p["behavior"]["inhibition"] - prev)
    prev = p["behavior"]["inhibition"]
check("the withdrawal spiral damps (shrinking inhibition increments)",
      all(increments[i] < increments[i - 1] + 1e-9 for i in range(1, len(increments))))
check("under sustained threat the character withdraws", p["behavior"]["set"]["mode"] == "withdraw")
check("inhibition stays off the wall under bombardment", p["behavior"]["inhibition"] < 0.9)

rec = p
for _ in range(6):
    rec = behavior.refresh(rec)
    rec = behavior.apply_event(rec, appr(v=0.7, tc=0.5, ag=0.7, out=0.8, inten=0.7))
check("a bright streak of rewarded action recovers the withdrawn character (no lock-in)",
      rec["behavior"]["set"]["mode"] == "approach"
      and rec["behavior"]["approach"] > p["behavior"]["approach"])


# --- modes & the enacted voice -------------------------------------------------------
c = person(E=0.7, N=0.8)
c["behavior"]["approach"] = 0.6; c["behavior"]["inhibition"] = 0.7
conflicted = behavior.apply_event(c, appr(v=0.8, tc=-0.6, inten=0.8))
check("a bright-but-scary event with both systems loud reads CONFLICTED",
      conflicted["behavior"]["set"]["mode"] == "conflicted")
check("the conflicted voice speaks both pulls",
      "dive in" in plain_reply(conflicted, "They asked me to lead the project.")
      and "disappear" in plain_reply(conflicted, "They asked me to lead the project."))
quiet_withdrawn = carry(person(E=-0.6, N=0.7), -0.6, "withdraw")
quiet_withdrawn["self"] = {"image": {"O": 0, "C": 0, "E": -0.6, "A": 0, "N": 0.7}, "esteem": -0.4}
reply = plain_reply(quiet_withdrawn, "There is another meeting tomorrow.")
check("the inclination survives a quiet character's length budget (the act is always voiced)",
      "keep my distance" in reply)
check("a steady character's reply carries no inclination",
      "keep my distance" not in plain_reply(person(), "Nothing much happened today.")
      and "go back at it" not in plain_reply(person(), "Nothing much happened today."))
check("the carried stance surfaces in the lens brief for the LLM push",
      "holding back" in lens(carry(person(), -0.6, "withdraw")))
check("a steady stance adds nothing to the lens (prompt-preserving)",
      lens(person()) == "")
check("the inclination joins the reply voice brief",
      any("leaning in" in l for l in voice(carry(person(), 0.7, "approach"))))

# note_act freezes the act the reply performs
acted = behavior.note_act(carry(person(), 0.5, "approach", as_act=False), style={"warmth": 0.4})
check("note_act freezes stance, mode, and the style spoken with",
      acted["behavior"]["last"]["tendency"] == 0.5
      and acted["behavior"]["last"]["mode"] == "approach"
      and acted["behavior"]["last"]["style"]["warmth"] == 0.4)

# offline/legacy safety
check("interpret() on a personality with no behavior key does not crash",
      interpret(appr(), {"traits": {"O": 0, "C": 0, "E": 0, "A": 0, "N": 0}})["intensity"] == 0.6)

# the memory stamp (needs numpy; skipped cleanly without it)
try:
    import numpy as _np  # noqa: F401
    from core.memory import create_memory, load_memories
    _name = "BehaviorStampTest"
    for suffix in (".memories.json", ".memories.embeddings.npy"):
        _p = f"data/characters/behaviorstamptest{suffix}"
        if os.path.exists(_p):
            os.remove(_p)
    stamped = carry(person(), -0.5, "withdraw")
    stamped["identity"] = {"name": _name}
    create_memory("a quiet day", [0.0, 1.0], appr(), {}, stamped, name=_name)
    record = load_memories(_name)[0]
    check("each memory records the stance it was met with (the act history in the hub)",
          record.get("stance", {}).get("mode") == "withdraw")
    for suffix in (".memories.json", ".memories.embeddings.npy"):
        _p = f"data/characters/behaviorstamptest{suffix}"
        if os.path.exists(_p):
            os.remove(_p)
except ImportError:
    print("SKIP -- memory stance stamp (numpy not installed)")


print()
print("%d/%d checks passed." % (_checks - _failed, _checks))
if _failed:
    print("SOME CHECKS FAILED.")
    sys.exit(1)
print("ALL CHECKS PASSED -- the character acts, the world answers, and the loop closes bounded.")
