from personality import (
    load_personality,
    save_personality
)

from memory import create_memory

from updater import apply_memory

from response import generate_response


def run_interaction(text, emotion, intensity):

    personality = load_personality()

    memory = create_memory(
        text=text,
        emotion=emotion,
        intensity=intensity
    )

    updated_personality = apply_memory(
        memory,
        personality
    )

    save_personality(updated_personality)

    response = generate_response(
        updated_personality
    )

    print("\nMEMORY:")
    print(memory)

    print("\nUPDATED PERSONALITY:")
    print(updated_personality)

    print("\nRESPONSE:")
    print(response)