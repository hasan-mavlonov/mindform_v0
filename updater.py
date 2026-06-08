"""Apply an experience's signed push to the five traits, with diminishing returns.

    trait <- clamp(trait + push * (1 - |trait|))

The (1 - |trait|) factor is the diminishing-returns term: a push moves a trait by
its full amount near 0 and by progressively less as the trait approaches +/-1, so
the trait asymptotes toward the extreme instead of jumping past it.

    Example, a party with push_E = 0.3:
        0.00 -> 0.30 -> 0.51 -> 0.66 -> 0.76 -> ...

The push is signed, so experiences can lower a trait as well as raise it.

Identity and temperament ride through untouched here. The temperament dynamics --
pulling the current trait back toward its baseline ``x += tau*(mu - x)`` and the
slow baseline drift ``mu += eta*(x - mu)`` -- are Slice 2; today this is the pure
experience-forms step.
"""


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def update_personality(personality, push):
    """Return a new personality with each trait moved by its push (input unchanged)."""
    traits = dict(personality["traits"])
    for dim, value in traits.items():
        traits[dim] = clamp(value + push.get(dim, 0.0) * (1 - abs(value)))

    return {
        **personality,
        "traits": traits,
        "experience_count": personality.get("experience_count", 0) + 1,
    }
