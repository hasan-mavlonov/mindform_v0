"""End-to-end demonstration of the MindForm interaction loop.

    text -> embedding -> trait prediction -> personality update -> memory storage

Run with: python simulation.py
"""

from personality import load_personality, save_personality
from trait_model import predict
from updater import update_personality
from memory import create_memory
from response import generate_response


def run_interaction(text):
    # 1. Load the current personality.
    personality = load_personality()

    # 2. Predict traits from the text.
    prediction = predict(text)

    # 3. Store the interaction in memory.
    create_memory(text=text, traits=prediction)

    # 4. Update the personality gradually toward the prediction.
    personality = update_personality(personality, prediction)

    # 5. Persist the updated personality.
    save_personality(personality)

    # 6. Print the results.
    print(f"\nINPUT: {text}")

    print("\nPREDICTED TRAITS:")
    for trait, value in prediction.items():
        print(f"  {trait:18s} {value:+.3f}")

    print("\nUPDATED PERSONALITY:")
    for trait, value in personality.items():
        print(f"  {trait:18s} {value:+.3f}")

    print(f"\nRESPONSE:\n  {generate_response(personality)}")

    return personality


if __name__ == "__main__":
    run_interaction("I love meeting new people.")
