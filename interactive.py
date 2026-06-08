"""
MindForm Interactive Shell

Type experiences and watch personality evolve in real time.

Commands:
  /genesis <bio>  -> birth a character from a short biography (seeds temperament)
  /show           -> show identity, current traits, and temperament baseline
  /reset          -> reset to a blank, neutral character
  /exit           -> quit
"""

from personality import (
    load_personality, save_personality, default_personality,
    read_traits, read_temperament,
)
from temperament import genesis
from encoder import encode_text
from appraisal import appraise
from llm_impact import push_from_text
from updater import update_personality
from memory import create_memory, recurrence


def print_state(personality):
    identity = personality.get("identity") or {}
    facts = ", ".join(f"{k}={v}" for k, v in identity.items() if v and k != "bio")
    if facts:
        print(f"\nIDENTITY: {facts}")

    traits = read_traits(personality)
    temperament = read_temperament(personality)
    print("\nCURRENT TRAIT        (<- baseline mu, stickiness tau):")
    for name, value in traits.items():
        mu = temperament[name]["mu"]
        tau = temperament[name]["tau"]
        print(f"  {name:18s}: {value:+.3f}   (<- mu {mu:+.2f}, tau {tau:.2f})")
    print()


def run():
    personality = load_personality()

    print("\n=== MindForm Interactive Mode ===")
    print("Type experiences, or: /genesis <bio>  /show  /reset  /exit\n")

    while True:
        text = input(">>> ").strip()

        if not text:
            continue

        if text == "/exit":
            break

        if text == "/show":
            print_state(personality)
            continue

        if text.startswith("/genesis"):
            bio = text[len("/genesis"):].strip()
            if not bio:
                print("usage: /genesis <a short biography>")
                continue
            personality, source, reasoning = genesis(bio)
            save_personality(personality)
            print(f"\nBorn a character (via {source})."
                  + (f"  --  {reasoning}" if reasoning else ""))
            print_state(personality)
            continue

        if text == "/reset":
            personality = default_personality()
            save_personality(personality)
            print("\nReset to a blank, neutral character.")
            print_state(personality)
            continue

        # --- CORE PIPELINE ---
        embedding = encode_text(text)
        appraisal = appraise(text)
        seen = recurrence(embedding)

        push, source, reasoning = push_from_text(text, appraisal)
        personality = update_personality(personality, push)

        create_memory(text, embedding, appraisal, push, personality)
        save_personality(personality)

        print("\n--- RESULT ---")
        print(f"Seen before: {seen}")
        print(f"Push via: {source}" + (f"  --  {reasoning}" if reasoning else ""))

        print("\nPush:")
        for k, v in push.items():
            print(f"  {k:10s}: {v:+.3f}")

        print_state(personality)


if __name__ == "__main__":
    run()
