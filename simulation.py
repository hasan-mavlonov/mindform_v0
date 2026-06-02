from updater import (
    load_personality,
    save_personality
)

from trait_model import (
    predict_trait_changes
)

from updater import (
    apply_trait_changes
)

from memory import (
    create_memory
)


def run_interaction(text):
    personality = load_personality()

    trait_changes = predict_trait_changes(
        text
    )

    create_memory(
        text=text,
        trait_changes=trait_changes
    )

    personality = apply_trait_changes(
        personality,
        trait_changes
    )

    save_personality(personality)

    print("\nTRAIT CHANGES:")
    print(trait_changes)

    print("\nPERSONALITY:")
    print(personality)

if __name__ == "__main__":
    for _ in range(5):
        run_interaction(
            "I love meeting new people at parties"
        )