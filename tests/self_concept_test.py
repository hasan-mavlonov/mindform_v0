"""SELF-CONCEPT & IDENTITY: the reflective self -- a self-image + a self-esteem gauge.

Dependency-free (pure arithmetic -- no torch / numpy / network), like the other node suites.
Proves the three claims of the slice:

  * SEED     -- a newborn's self-image mirrors its birth traits; self-regard rests at a
                dispositional baseline from temperament + achievement.
  * FORMS    -- self-esteem rides success/acceptance (sociometer) and relaxes to its baseline;
                the self-image drifts toward the actual traits (Bem) but RESISTS disconfirming
                moves (Swann), so it lags -- the "still thinks they're the shy one" effect.
  * ACTS     -- an event that CONTRADICTS the self-view reads darker + more threatening + more
                self-relevant; high self-regard buffers threat, low regard amplifies it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.llm as llm_mod
llm_mod.LLM_API_KEY = ""           # never touch the network

from core.config import BASIS, SCHEMA_LEARN
import nodes.self_concept as sc
from nodes.cognition import interpret, read_lens
from core.personality import default_personality, migrate

_checks = 0
_failed = 0


def check(label, cond):
    global _checks, _failed
    _checks += 1
    if not cond:
        _failed += 1
    print(("PASS" if cond else "FAIL") + " -- " + label)


def person(image=None, esteem=0.0, traits=None, mu=None, values=None):
    """A minimal personality with an explicit self-image / esteem."""
    zero = {k: 0.0 for k in BASIS}
    img = {**zero, **(image or {})}
    return {
        "traits": {**zero, **(traits or {})},
        "temperament": {"mu": {**zero, **(mu or {})}},
        "character": {"values": values or {}},
        "self": {"image": img, "esteem": esteem},
    }


# --- SEED ----------------------------------------------------------------------
blank = default_personality()
check("a newborn's self-image mirrors its birth traits",
      blank["self"]["image"] == blank["traits"])
check("a blank character rests at neutral self-regard", abs(blank["self"]["esteem"]) < 1e-9)

check("neuroticism lowers the dispositional esteem baseline",
      sc.seed_base(person(mu={"N": 0.8})) < 0)
check("extraversion + achievement raise the baseline",
      sc.seed_base(person(mu={"E": 0.8}, values={"AC": 0.8})) > 0)
check("the baseline stays clamped in [-1, 1]",
      -1.0 <= sc.seed_base(person(mu={"N": -1.0, "E": 1.0}, values={"AC": 1.0})) <= 1.0)

legacy = migrate({"traits": {"O": 0, "C": 0, "E": -0.7, "A": 0, "N": 0.8}, "character": {"values": {}}})
check("migrate backfills the self (image = current traits, esteem at baseline)",
      "self" in legacy and legacy["self"]["image"]["E"] == -0.7 and legacy["self"]["esteem"] < 0)


# --- ESTEEM: sociometer + relax-to-baseline ------------------------------------
success = {"outcome": 0.8, "agency": 0.6, "social": 0.5, "valence": 0.7, "self_relevance": 0.8}
failure = {"outcome": -0.8, "agency": -0.5, "social": -0.4, "valence": -0.7, "self_relevance": 0.8}
up = sc.apply_event(person(esteem=0.0), success, person()["traits"])["self"]["esteem"]
down = sc.apply_event(person(esteem=0.0), failure, person()["traits"])["self"]["esteem"]
check("success / acceptance raises self-esteem", up > 0)
check("failure / rejection lowers self-esteem", down < 0)
check("self-esteem stays within [-1, 1] under a strong event",
      -1.0 <= sc.apply_event(person(esteem=0.9), success, person()["traits"])["self"]["esteem"] <= 1.0)

irrelevant = {"outcome": 0.8, "agency": 0.6, "social": 0.5, "valence": 0.7, "self_relevance": 0.0}
gated = sc.apply_event(person(esteem=0.0), irrelevant, person()["traits"])["self"]["esteem"]
check("an event with no self-relevance barely moves self-worth (the self-relevance gate)",
      abs(gated) < up)

# esteem relaxes toward its baseline (here base > 0 from E+AC); an above-base esteem falls
p_hi = person(esteem=0.9, mu={"E": 0.5}, values={"AC": 0.5})
check("self-esteem relaxes down toward a lower baseline",
      sc.refresh(p_hi)["self"]["esteem"] < 0.9)
p_lo = person(esteem=-0.9, mu={"E": 0.5}, values={"AC": 0.5})
check("self-esteem relaxes up toward a higher baseline",
      sc.refresh(p_lo)["self"]["esteem"] > -0.9)


# --- SELF-IMAGE: Bem drift with Swann resistance -------------------------------
neutral = {"outcome": 0.0, "agency": 0.0, "social": 0.0, "valence": 0.0, "self_relevance": 0.0}
# Bem: with the actual trait fixed, the self-image converges toward it over turns
p = person(image={"E": 0.0}, traits={"E": 0.6})
gaps = []
for _ in range(5):
    p = sc.apply_event(p, neutral, p["traits"])
    gaps.append(abs(p["self"]["image"]["E"] - 0.6))
check("the self-image drifts toward the actual trait over turns (Bem convergence)",
      all(gaps[i] < gaps[i - 1] for i in range(1, len(gaps))))

# Swann: from the SAME self-view and the SAME gap size, a DISCONFIRMING move is resisted
confirm = sc.apply_event(person(image={"E": 0.5}, traits={"E": 0.9}), neutral,
                         {"O": 0, "C": 0, "E": 0.9, "A": 0, "N": 0})["self"]["image"]["E"]
disconfirm = sc.apply_event(person(image={"E": 0.5}, traits={"E": 0.1}), neutral,
                            {"O": 0, "C": 0, "E": 0.1, "A": 0, "N": 0})["self"]["image"]["E"]
check("a confirming self-image move is larger than a disconfirming one of equal gap (Swann)",
      abs(confirm - 0.5) > abs(disconfirm - 0.5))
check("the self-image stays clamped in [-1, 1]",
      -1.0 <= sc.apply_event(person(image={"E": 0.9}, traits={"E": -1.0}), neutral,
                             {"O": 0, "C": 0, "E": -1.0, "A": 0, "N": 0})["self"]["image"]["E"] <= 1.0)


# --- ACTS: the self bends perception -------------------------------------------
party = {"valence": 0.2, "intensity": 0.6, "novelty": 0.4, "agency": 0.1, "social": 1.0,
         "outcome": 0.0, "self_relevance": 0.4, "threat_challenge": 0.2}
intro = interpret(party, person(image={"E": -0.7}))    # "I'm reserved" -> the party contradicts it
extra = interpret(party, person(image={"E": 0.7}))     # "I'm outgoing" -> the party affirms it
check("a self-CONTRADICTING event reads darker than a self-affirming one",
      intro["valence"] < extra["valence"])
check("a self-CONTRADICTING event reads as more threatening",
      intro["threat_challenge"] < extra["threat_challenge"])
check("a self-relevant event (touching a trait they identify with) reads as more self-relevant",
      intro["self_relevance"] > party["self_relevance"])

setback = {"valence": -0.3, "intensity": 0.6, "novelty": 0.2, "agency": -0.4, "social": 0.0,
           "outcome": -0.5, "self_relevance": 0.4, "threat_challenge": -0.2}
hi = interpret(setback, person(esteem=0.8))
lo = interpret(setback, person(esteem=-0.8))
check("high self-regard buffers threat; low regard amplifies it",
      hi["threat_challenge"] > lo["threat_challenge"])

flat = interpret(party, person())      # image all 0, esteem 0 -> no self
check("no self yet -> the self lens leaves the appraisal unchanged",
      flat["valence"] == party["valence"] and flat["threat_challenge"] == party["threat_challenge"]
      and flat["self_relevance"] == party["self_relevance"])

no_self = {"traits": {k: 0.0 for k in BASIS}}      # a personality with no "self" key at all
try:
    interpret(party, no_self)
    ok = True
except Exception:
    ok = False
check("interpret() on a personality with no self state does not crash (offline/legacy safe)", ok)


# --- Stability & lens ----------------------------------------------------------
p = person(esteem=-0.3, mu={"N": 0.6})
for _ in range(10):
    p = sc.refresh(p)
    p = sc.apply_event(p, setback, p["traits"])
    check_bounded = -1.0 <= p["self"]["esteem"] <= 1.0
check("self-esteem stays bounded and does not collapse under repeated setbacks",
      -1.0 < p["self"]["esteem"] <= 1.0)

check("low self-regard surfaces a self tag in the lens",
      "self" in read_lens(person(esteem=-0.8)).lower())
check("a large self-image-vs-actual gap surfaces a 'lags' tag",
      "lag" in read_lens(person(image={"E": 0.7}, traits={"E": -0.6})).lower())
check("a neutral, accurate self -> no self tag",
      read_lens(person()) == "")


print()
print("%d/%d checks passed." % (_checks - _failed, _checks))
if _failed:
    print("SOME CHECKS FAILED.")
    sys.exit(1)
print("ALL CHECKS PASSED -- the self forms, lags reality, and bends how the world is read.")
