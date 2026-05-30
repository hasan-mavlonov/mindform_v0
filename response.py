def generate_response(personality):

    trust = personality["trust"]
    openness = personality["openness"]

    if trust > 0.5:

        return (
            "I feel comfortable opening up. "
            "I think I can trust this interaction."
        )

    elif trust < -0.5:

        return (
            "I feel guarded right now. "
            "I don't fully trust this situation."
        )

    elif openness > 0.5:

        return (
            "I'm curious and willing to explore new ideas."
        )

    return (
        "I'm still learning how I feel."
    )