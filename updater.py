"""Gradually update personality state toward predicted traits.

The personality is never overwritten. Instead each trait drifts toward its
predicted value by a fraction of the gap, with diminishing returns near the
extremes so that traits become progressively harder to push as they approach
+/-1.
"""

LEARNING_RATE = 0.1


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def update_personality(personality, prediction, learning_rate=LEARNING_RATE):
    """Move each trait toward its predicted value.

        effective_change = (prediction - current) * learning_rate * (1 - abs(current))
        new_value        = clamp(current + effective_change, -1, 1)

    Returns a new personality dict; the input is not mutated.
    """
    updated = dict(personality)

    for trait, predicted in prediction.items():
        current = updated.get(trait, 0.0)

        effective_change = (
            (predicted - current) * learning_rate * (1 - abs(current))
        )

        updated[trait] = clamp(current + effective_change)

    return updated
