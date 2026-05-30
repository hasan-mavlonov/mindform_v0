from personality import clamp


def apply_memory(memory, personality):

    emotion = memory["emotion"]
    intensity = memory["intensity"]

    if emotion == "positive_social":

        personality["trust"] += 0.15 * intensity
        personality["openness"] += 0.10 * intensity

    elif emotion == "rejection":

        personality["trust"] -= 0.20 * intensity
        personality["openness"] -= 0.05 * intensity

    personality["trust"] = clamp(personality["trust"])
    personality["openness"] = clamp(personality["openness"])

    return personality