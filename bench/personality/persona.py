"""What a personality test's prompt is allowed to know about a character.

Self-contained on purpose: bench/heart already has its own leak checker
(bench.heart.arms.leak_check), built for a different job -- checking a
scenario/MCQ prompt that may also carry a D2 "Formed Disposition" state block
whose own numbers need provenance-checking rather than blanket forbidding.
None of that applies here. A BFI-2 item prompt is much simpler (a persona
summary plus one statement to rate) and this module checks it against exactly
the same forbidden material HEART is careful about, independently, so this
package's safety does not depend on importing HEART's methodology-specific
code.
"""

from bench.heart import heartdata, mfadapter

# Only these two fields, and only this shape of state summary, may ever reach
# a personality-test prompt. Nothing here is more than heartdata.character_public
# and mfadapter.state_summary already independently guarantee -- this module
# re-affirms rather than replaces those guarantees.
ALLOWED_CHARACTER_KEYS = ("id", "occupation")


def character_context(char):
    """The only character-identity fields a prompt may use. No HEART labels."""
    pub = heartdata.character_public(char)
    return {k: pub[k] for k in ALLOWED_CHARACTER_KEYS if k in pub}


def state_context(bench_name):
    """The frozen character's own formed state -- MindForm's read of itself.

    Exactly bench.heart.mfadapter.state_summary(): traits, values, moral
    weighting, drives, self-image, beliefs. Never HEART's hidden big_five --
    that field is never read by anything in this module.
    """
    return mfadapter.state_summary(bench_name)


def leak_check(prompt, char):
    """Hard check: no HEART-withheld label reached a personality-test prompt.

    Returns (ok, detail). Deliberately independent of bench.heart.arms.leak_check
    (see module docstring) -- it checks the same forbidden material, by its own
    code path, so this package's safety guarantee does not depend on HEART's
    methodology-specific checker continuing to exist or behave the same way.
    """
    hits = []
    low = prompt.lower()

    forbidden = heartdata.forbidden_strings(char, [])   # no questions here, so no
                                                        # source_character check needed
    for s in forbidden:
        if s and len(s) > 3 and s.lower() in low:
            hits.append(s[:60])

    for field in ("description", "self_value_logic", "core_patterns"):
        v = char.get(field)
        if isinstance(v, str) and len(v) > 20 and v[:40].lower() in low:
            hits.append(f"field:{field}")

    # The withheld numeric Big Five itself: HEART's own hidden ground truth
    # must never appear near a trait word, at all, anywhere in this prompt --
    # there is no legitimate reason (unlike D2's derived state block) for a
    # personality-test prompt to state HEART's numbers.
    import re
    gt = char.get("big_five") or {}
    for trait, val in gt.items():
        try:
            want = float(val)
        except (TypeError, ValueError):
            continue
        for m in re.finditer(re.escape(trait), prompt, re.I):
            window = prompt[m.end(): m.end() + 16]
            for tok in re.findall(r"[-+]?\d*\.?\d+", window):
                try:
                    if float(tok) == want:
                        hits.append(f"big_five:{trait}={val}")
                except ValueError:
                    pass

    return (len(hits) == 0), "; ".join(sorted(set(hits))[:5])
