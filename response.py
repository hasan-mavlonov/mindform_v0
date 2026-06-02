"""Generate a simple natural-language response reflecting personality state.

This is a lightweight, rule-based stand-in that surfaces the agent's current
OCEAN personality as a short sentence.
"""


def generate_response(personality):
    openness = personality.get("openness", 0.0)
    extraversion = personality.get("extraversion", 0.0)
    agreeableness = personality.get("agreeableness", 0.0)
    neuroticism = personality.get("neuroticism", 0.0)

    if neuroticism > 0.5:
        return "I feel a little anxious right now, but I'm trying to stay grounded."

    if extraversion > 0.5:
        return "I feel energized and happy to connect with others."

    if openness > 0.5:
        return "I'm curious and eager to explore new ideas."

    if agreeableness > 0.5:
        return "I feel warm and willing to help."

    return "I'm still learning how I feel."
