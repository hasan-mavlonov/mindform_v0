"""
MindForm Interactive Shell

Type experiences and watch personality evolve in real time.

Commands:
  /exit   -> quit
  /reset  -> reset personality to zero
  /show   -> show current traits
"""

from personality import (
    load_personality, save_personality, default_personality, read_traits, read_state
)
from encoder import encode_text
from appraisal import appraise, blend_appraisal, bias_appraisal
from impact import impact
from updater import update_personality
from memory import create_memory, recurrence, retrieve_similar


def print_traits(personality):
    traits = read_traits(personality)
    mood = read_state(personality)
    print("\nCURRENT PERSONALITY (disposition / mood):")
    for k in traits:
        print(f"  {k:18s}: trait {traits[k]:+.3f}   mood {mood[k]:+.3f}")
    print()


def run():
    personality = load_personality()

    print("\n=== MindForm Interactive Mode ===")
    print("Type experiences. Commands: /exit /reset /show\n")

    while True:
        text = input(">>> ").strip()

        if not text:
            continue

        if text == "/exit":
            break

        if text == "/show":
            print_traits(personality)
            continue

        if text == "/reset":
            personality = default_personality()
            save_personality(personality)
            print("\nPersonality reset.\n")
            print_traits(personality)
            continue

        # --- CORE PIPELINE ---
        embedding = encode_text(text)[0]
        neighbors = retrieve_similar(embedding)
        appraisal = blend_appraisal(appraise(text), [m["appraisal"] for m in neighbors])  # memory tint
        appraisal = bias_appraisal(appraisal, personality)                                # who-he-was tint
        seen = recurrence(embedding)

        push = impact(appraisal, personality)
        personality = update_personality(personality, push, recurrence=seen)

        create_memory(text, embedding, appraisal, push, personality)
        save_personality(personality)

        print("\n--- RESULT ---")
        print(f"Seen before: {seen}")

        print("\nPush:")
        for k, v in push.items():
            print(f"  {k:10s}: {v:+.3f}")

        print_traits(personality)


if __name__ == "__main__":
    run()