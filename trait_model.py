from encoder import encode_text


def predict_trait_changes(text):

    embedding = encode_text(text)

    text = text.lower()

    changes = {
        "openness": 0.0,
        "conscientiousness": 0.0,
        "extraversion": 0.0,
        "agreeableness": 0.0,
        "neuroticism": 0.0,
    }

    if "party" in text:
        changes["extraversion"] += 0.15

    if "people" in text:
        changes["extraversion"] += 0.10

    if "book" in text:
        changes["openness"] += 0.10

    if "study" in text:
        changes["conscientiousness"] += 0.10

    if "help" in text:
        changes["agreeableness"] += 0.10

    if "anxious" in text:
        changes["neuroticism"] += 0.15

    return changes