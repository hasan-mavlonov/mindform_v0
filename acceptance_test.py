"""Acceptance test: two-timescale personality formation (stylized facts).

Dependency-free (no torch / numpy / sentence-transformers). This is also the first
"stylized-facts" ruler: it checks that the dynamics reproduce qualitative truths of
personality development, not just that code runs. An empty push {} stands for a
quiet period (time passing with no salient experience), isolating mood decay and
set-point return.

  1. one experience moves MOOD (state) a lot but DISPOSITION (trait) barely
  2. repeating it accumulates the slow trait as a bounded S-curve (ramp -> diminish)
  3. after a spike, mood fades back toward 0 but a small trait residue persists
  4. an unreinforced built-up trait partially relaxes back toward its set-point
  5. affective sign is immediate (mood); the lasting trait forms with repetition
  6. maturation: a sustained stream of mastery raises C and lowers N over time
  8. density (Whole Trait Theory): a volatile life has wider within-person state
     dispersion than a consistent one, even when the disposition is comparable

Run: python3 acceptance_test.py
"""

import math

from personality import default_personality, new_agent
from appraisal import appraise, blend_appraisal, bias_appraisal
from impact import impact
from updater import update_personality

QUIET = {}   # empty push: a quiet period -> only mood decay + set-point return


def state(p, d):
    return p["traits"][d]["state"]


def trait(p, d):
    return p["traits"][d]["trait"]


def disp(p, d):
    """Within-person dispersion (Whole Trait Theory SD) of an axis."""
    return math.sqrt(p["traits"][d]["trait_var"])


def run(p, appraisal, n=1):
    push = impact(appraisal) if appraisal is not QUIET else QUIET
    for _ in range(n):
        p = update_personality(p, push)
    return p


party = appraise("I went to a party and had fun.")
terror = appraise("I am terrified of everything.")
mastery = appraise("I faced my fear at the party and it went fine.")

# --- 1. mood reacts fast, disposition barely moves on a single experience ---
p1 = run(default_personality(), party)
s_E, t_E = state(p1, "E"), trait(p1, "E")

# --- 2. repetition accumulates the slow trait: bounded S-curve ---
p = default_personality()
traj = []
for _ in range(40):
    p = run(p, party)
    traj.append(trait(p, "E"))
incr = [traj[i] - (traj[i - 1] if i else 0.0) for i in range(len(traj))]

# --- 3. spike then quiet: mood fades, trait residue persists ---
p3 = run(run(default_personality(), party), QUIET, n=10)

# --- 4. set-point return: build E up, then a long quiet period ---
p4 = run(default_personality(), party, n=40)
peak_E = trait(p4, "E")
p4 = run(p4, QUIET, n=100)
relaxed_E = trait(p4, "E")

# --- 5. affective sign is immediate; trait forms with repetition ---
imm_terror = state(run(default_personality(), terror), "N")
imm_mastery = state(run(default_personality(), mastery), "N")
form_terror = trait(run(default_personality(), terror, n=30), "N")
form_mastery = trait(run(default_personality(), mastery, n=30), "N")

# --- 6. maturation: sustained mastery raises C, lowers N ---
p6 = run(default_personality(), mastery, n=40)

# --- 7. memory feedback: recurrence damps mood (habituation) but boosts the slow
#        trait (chronicity); retrieval blends a new appraisal toward memory ---
fresh = update_personality(default_personality(), impact(party), recurrence=0)
recurred = update_personality(default_personality(), impact(party), recurrence=4)
neg = {k: -v for k, v in party.items()}
blended = blend_appraisal(party, [neg])

# --- 8. density layer (Whole Trait Theory): a volatile life (alternating terror and
#        mastery) has wider within-person N dispersion than a consistent mastery one ---
consistent = run(default_personality(), mastery, n=30)
volatile = default_personality()
for _ in range(15):
    volatile = run(volatile, terror)
    volatile = run(volatile, mastery)
disp_consistent_N = disp(consistent, "N")
disp_volatile_N = disp(volatile, "N")

# --- 9. state-dependence: the SAME text nudges two agents differently, because each
#        reads and reacts through who it already is ("changed based on who he was") ---
adverse = appraise("I felt sad about the news.")   # mildly negative, NOT saturated
anxious0 = new_agent({"N": 0.6})        # an emotionally reactive agent
calm0 = new_agent({"N": -0.6})          # a steady agent
read_anx = bias_appraisal(adverse, anxious0)    # same text, read through each agent
read_calm = bias_appraisal(adverse, calm0)
push_anx_N = impact(read_anx, anxious0)["N"]
push_calm_N = impact(read_calm, calm0)["N"]

# --- 10. crystallization: a young agent forms faster than a seasoned one (same event) ---
young = default_personality()
seasoned = default_personality()
seasoned["experience_count"] = 300
young_dE = trait(update_personality(young, impact(party)), "E")
seasoned_dE = trait(update_personality(seasoned, impact(party)), "E")

# --- 11. divergence: under an IDENTICAL adverse life, the already-anxious agent ends
#         up more neurotic than the steady one (the corresponsive principle) ---
def live(p, text, n):
    for _ in range(n):
        a = bias_appraisal(appraise(text), p)
        p = update_personality(p, impact(a, p))
    return p

adverse_text = "I failed and felt afraid and helpless."
anx_end_N = trait(live(new_agent({"N": 0.6}), adverse_text, 20), "N")
calm_end_N = trait(live(new_agent({"N": -0.6}), adverse_text, 20), "N")

print(f"1) one party:      mood_E={s_E:+.3f}  trait_E={t_E:+.3f}")
print(f"2) trait_E (1,5,10,20,40): {[round(traj[i],3) for i in (0,4,9,19,39)]}")
print(f"3) spike->quiet:   mood_E={state(p3,'E'):+.3f}  trait_E residue={trait(p3,'E'):+.4f}")
print(f"4) set-point:      peak_E={peak_E:+.3f} -> relaxed_E={relaxed_E:+.3f}")
print(f"5) N immediate:    terror={imm_terror:+.3f} mastery={imm_mastery:+.3f} | formed: terror={form_terror:+.3f} mastery={form_mastery:+.3f}")
print(f"6) maturation:     trait_C={trait(p6,'C'):+.3f}  trait_N={trait(p6,'N'):+.3f}")
print(f"7) memory feedback: mood fresh={state(fresh,'E'):+.3f} recurred={state(recurred,'E'):+.3f} | "
      f"trait fresh={trait(fresh,'E'):+.4f} recurred={trait(recurred,'E'):+.4f} | blend_valence={blended['valence']:+.2f}")
print(f"8) dispersion N:   consistent_sd={disp_consistent_N:.3f}  volatile_sd={disp_volatile_N:.3f}")
print(f"9) who-he-was:     same text -> push_N anxious={push_anx_N:+.3f} calm={push_calm_N:+.3f} | "
      f"read threat anx={read_anx['threat_challenge']:+.2f} calm={read_calm['threat_challenge']:+.2f}")
print(f"10) crystallize:   young dE={young_dE:+.4f}  seasoned dE={seasoned_dE:+.4f}")
print(f"11) divergence:    same adverse life -> N anxious={anx_end_N:+.3f}  calm={calm_end_N:+.3f}")

checks = {
    "mood moves on first experience (state_E > 0.15)": s_E > 0.15,
    "disposition barely moves on first experience (trait_E < 0.05)": t_E < 0.05,
    "mood is faster than disposition (|state| > |trait|)": abs(s_E) > abs(t_E),
    "repetition accumulates the trait (monotone up)": all(traj[i] < traj[i + 1] for i in range(len(traj) - 1)),
    "trait stays bounded (< 1)": traj[-1] < 1.0,
    "trait formation is substantial after repetition (> 0.3)": traj[-1] > 0.3,
    "diminishing returns eventually (late increment < early-peak)": incr[-1] < incr[7],
    "after a spike mood fades back toward 0 (|state_E| < 0.05)": abs(state(p3, "E")) < 0.05,
    "a spike leaves a lasting trait residue (> 0)": trait(p3, "E") > 0,
    "set-point return: unreinforced trait relaxes back": relaxed_E < peak_E,
    "set-point return is partial, not erased/overshot (0 < relaxed < peak)": 0 < relaxed_E < peak_E,
    "affective sign immediate: terror raises mood_N, mastery lowers it": imm_terror > 0 > imm_mastery,
    "lasting trait forms with repetition: terror_N > mastery_N": form_terror > 0 > form_mastery,
    "maturation: sustained mastery raises C": trait(p6, "C") > 0.2,
    "maturation: sustained mastery lowers N": trait(p6, "N") < -0.2,
    "recurrence damps mood (habituation)": abs(state(recurred, "E")) < abs(state(fresh, "E")),
    "recurrence boosts consolidation (chronicity)": trait(recurred, "E") > trait(fresh, "E"),
    "retrieval blends appraisal toward memory": blended["valence"] < party["valence"],
    "blend with no memory is identity": blend_appraisal(party, []) == party,
    "density: a volatile life has wider state dispersion than a consistent one":
        disp_volatile_N > disp_consistent_N,
    "density: a consistent life keeps dispersion small (< 0.05)": disp_consistent_N < 0.05,
    "who-he-was: same text reads as more threat for the anxious agent":
        read_anx["threat_challenge"] < read_calm["threat_challenge"],
    "who-he-was: same text pushes N harder for the anxious agent":
        push_anx_N > push_calm_N,
    "crystallization: a young agent forms faster than a seasoned one":
        young_dE > seasoned_dE,
    "divergence: identical adverse life leaves the anxious agent more neurotic":
        anx_end_N > calm_end_N,
}

print("\nRESULTS:")
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

assert all(checks.values()), "acceptance test FAILED"
print("\nALL CHECKS PASSED -- two-timescale formation reproduces the stylized facts.")
