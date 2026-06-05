from trait_model import predict

TESTS = [
    # Extraversion
    "I love meeting new people.",
    "I avoid social gatherings whenever possible.",
    "I enjoy being the center of attention.",
    "I hate talking to people.",
    "I spend most weekends alone.",
    "I love large parties.",

    # Neuroticism
    "I am constantly worried.",
    "I rarely feel anxious.",
    "Small problems keep me awake at night.",
    "I stay calm under pressure.",
    "I panic easily.",
    "I am emotionally stable.",

    # Conscientiousness
    "I always finish my work on time.",
    "My room is extremely organized.",
    "I procrastinate on everything.",
    "I often miss deadlines.",
    "I carefully plan my day.",
    "I am very disciplined.",

    # Openness
    "I love exploring new ideas.",
    "I enjoy abstract philosophy.",
    "I dislike change.",
    "I prefer familiar routines.",
    "I love art museums.",
    "I am fascinated by science.",

    # Agreeableness
    "I enjoy helping others.",
    "I often volunteer my time.",
    "I am rude to strangers.",
    "I frequently insult people.",
    "I try to understand everyone's perspective.",
    "I am compassionate.",

    # Mixed
    "I love meeting new people and helping them.",
    "I am anxious but very organized.",
    "I hate crowds but enjoy deep conversations.",
    "I am constantly worried about failing.",
    "I enjoy debating strangers.",
    "I spend all day playing video games alone.",

    # Extreme cases
    "I never talk to anyone.",
    "I am terrified of social interaction.",
    "I love every person I meet.",
    "I am obsessed with planning.",
    "I have zero interest in new experiences.",
    "I am endlessly curious about everything.",

    # MindForm examples
    "I love riding bikes.",
    "I love sleeping all day and don't talk to my friends.",
    "I hate talking to people.",
    "I faced my fear at a party.",
    "I spent the week alone and sad.",
    "I just discovered quantum physics and it is fascinating.",
    "I am terrified of everything.",
    "I enjoy making new friends.",
    "I prefer staying alone.",
    "I love meeting new people.",
]

for text in TESTS:
    print("\n" + "=" * 80)
    print(text)
    print(predict(text))