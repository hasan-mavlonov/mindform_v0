"""Evaluation for a personality-*formation* model.

This replaces the old Pandora ``readout`` (a ``text -> OCEAN`` MLP). That model was
removed because, in a simulator, **the personality is a state variable we set and
observe** -- estimating it from text solves a problem we do not have, and the
PANDORA model in any case learned Reddit *author attribution*, not personality
state, and was never validated (MSE only, no correlation). See
``docs/RESEARCH_REVIEW.md``.

The right instruments for a formation model are:

  1. Report the **observed** formed personality directly (no estimation).
  2. Score a **trajectory** against developmental stylized facts (construct
     validity) -- e.g. the maturity principle, set-point relaxation, mood faster
     than disposition.
  3. For a *real generative agent* only, an external-validity probe: administer a
     validated inventory and score with its published key (see ``ExpressionProbe``).

Dependency-free: imports nothing that needs torch / sentence-transformers / network.
"""

from math import sqrt
from typing import Protocol

from config import BASIS, BASIS_NAMES
from personality import read_traits, read_state


# --------------------------------------------------------------------------- #
# 1. Report the observed state (no estimation -- the state is known).
# --------------------------------------------------------------------------- #
def observed_dispersion(personality):
    """Within-person dispersion per trait (Whole Trait Theory SD).

    sqrt of the tracked EWMA of squared successive swings in momentary expression
    (MSSD-style instability). Falls back to 0.0 for pre-dispersion personality structs.
    """
    return {
        BASIS_NAMES[d]: sqrt(max(0.0, layers.get("trait_var", 0.0)))
        for d, layers in personality["traits"].items()
    }


def observe(personality):
    """The full observed personality: disposition (mean), mood (state), dispersion."""
    return {
        "disposition": read_traits(personality),   # slow central tendency
        "mood": read_state(personality),            # fast momentary state
        "dispersion": observed_dispersion(personality),
        "experience_count": personality.get("experience_count", 0),
    }


def format_observe(personality):
    obs = observe(personality)
    lines = [f"observed personality after {obs['experience_count']} experiences:",
             f"  {'trait':18s} {'disposition':>12s} {'mood':>8s} {'dispersion':>11s}"]
    for d in BASIS:
        name = BASIS_NAMES[d]
        lines.append(f"  {name:18s} {obs['disposition'][name]:+12.3f}"
                     f" {obs['mood'][name]:+8.3f} {obs['dispersion'][name]:11.3f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2. Construct-validity over a trajectory (a reusable "ruler").
#    A trajectory is a list of personality snapshots in time order.
# --------------------------------------------------------------------------- #
def trait_path(history, trait_key):
    """Disposition of one Big Five axis over a trajectory (by single-letter key)."""
    return [snap["traits"][trait_key]["trait"] for snap in history]


def maturity_index(history):
    """The maturity principle as one number: dC + dA - dN over the trajectory.

    Roberts et al.: with maturation people tend to rise in Conscientiousness and
    Agreeableness and fall in Neuroticism. A maturing stream of experience should
    push this index positive; a destabilizing stream, negative.
    """
    if len(history) < 2:
        return 0.0
    first, last = history[0]["traits"], history[-1]["traits"]
    d = {k: last[k]["trait"] - first[k]["trait"] for k in BASIS}
    return d["C"] + d["A"] - d["N"]


def is_bounded(history, limit=1.0):
    """No disposition ever leaves [-limit, limit] along the trajectory."""
    return all(abs(snap["traits"][k]["trait"]) <= limit
               for snap in history for k in BASIS)


def mood_faster_than_disposition(history):
    """Across the trajectory, mood moves more than disposition (two timescales)."""
    smax = max((abs(snap["traits"][k]["state"]) for snap in history for k in BASIS),
               default=0.0)
    tmax = max((abs(snap["traits"][k]["trait"]) for snap in history for k in BASIS),
               default=0.0)
    return smax >= tmax


# --------------------------------------------------------------------------- #
# 3. External-validity probe (for a REAL generative agent only).
# --------------------------------------------------------------------------- #
class ExpressionProbe(Protocol):
    """Estimate expressed traits from an agent's *behavior* -- external validity.

    This exists only to answer: "does an agent whose disposition we formed toward,
    say, high Extraversion actually *behave* more extraverted?" It is meaningful
    only once a real generator replaces the canned ``response.py``.

    The PANDORA ``text -> OCEAN`` MLP was deliberately NOT kept as the default
    implementation: it learned which Reddit author writes like what (attribution),
    not expressed state, and was never validated. The correct implementation
    administers a *validated inventory* (e.g. BFI-2 / IPIP items) to the generator
    in-character and scores the answers with the inventory's published key -- the
    same method used to measure personality in humans and, increasingly, in LLMs.
    """

    def measure(self, agent) -> dict:  # {trait_name: score}
        ...


class NullProbe:
    """Default: no external probe is configured (no real generator exists yet)."""

    def measure(self, agent) -> dict:
        raise NotImplementedError(
            "No ExpressionProbe configured. A behavioral probe is meaningful only "
            "with a real generative agent; wire one up and score it with a validated "
            "inventory's published key (see ExpressionProbe)."
        )


if __name__ == "__main__":
    # Dependency-free demo: form a personality over a maturing stream and report.
    from personality import default_personality
    from appraisal import appraise
    from impact import impact
    from updater import update_personality

    mastery = appraise("I faced my fear at the party and it went fine.")
    push = impact(mastery)

    p = default_personality()
    history = [p]
    for _ in range(40):
        p = update_personality(p, push)
        history.append(p)

    print(format_observe(p))
    print(f"\nmaturity_index (dC + dA - dN): {maturity_index(history):+.3f}  (expect > 0)")
    print(f"bounded: {is_bounded(history)}   mood_faster_than_disposition: "
          f"{mood_faster_than_disposition(history)}")
