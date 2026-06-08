"""
MindForm Interactive Shell

Start a new character (guided fields) or continue an existing one, then type
experiences and watch the personality evolve.

Commands:
  /new            -> create a new character via the field form
  /genesis <bio>  -> create a new character from a one-line biography
  /show           -> show identity, current traits, and temperament baseline
  /reset          -> reset to a blank, neutral character
  /exit           -> quit
"""

import os

from config import IDENTITY_FIELDS
from personality import (
    load_personality, save_personality, default_personality,
    read_traits, read_temperament, PERSONALITY_FILE,
)
from temperament import genesis, create_character
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


def prompt_fields():
    """Collect immutable identity fields + a free-text background for temperament."""
    print("\n-- New character (press Enter to skip a field) --")
    fields = {}
    for key, label in IDENTITY_FIELDS:
        value = input(f"  {label}: ").strip()
        if value:
            fields[key] = value
    background = input("  Background (a few words about their nature): ").strip()
    if background:
        fields["background"] = background
    return fields


def create_new():
    personality, source, reasoning = create_character(prompt_fields())
    save_personality(personality)
    print(f"\nCharacter created (temperament via {source})."
          + (f"  --  {reasoning}" if reasoning else ""))
    print_state(personality)
    return personality


def start():
    """Startup menu: continue the saved character, or create a new one."""
    print("\n=== MindForm ===")
    if os.path.exists(PERSONALITY_FILE):
        saved = load_personality()
        name = (saved.get("identity") or {}).get("name") or "unnamed character"
        print("  1) New character")
        print(f"  2) Continue: {name}")
        if input("> ").strip() == "2":
            print_state(saved)
            return saved
    return create_new()


def run():
    personality = start()

    print("Type experiences, or: /new  /genesis <bio>  /show  /reset  /exit\n")

    while True:
        text = input(">>> ").strip()

        if not text:
            continue

        if text == "/exit":
            break

        if text == "/show":
            print_state(personality)
            continue

        if text == "/new":
            personality = create_new()
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
