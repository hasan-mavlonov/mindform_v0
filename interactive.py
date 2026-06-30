"""
MindForm Interactive Shell

Pick an existing character or create a new one, then talk -- each line is an
experience that nudges the personality.

Commands (while talking):
  /switch         -> pick a different existing character
  /new            -> create a new character (questionnaire)
  /genesis <bio>  -> create a character from a one-line biography (LLM/heuristic)
  /show           -> show identity, current traits, and temperament baseline
  /exit           -> quit
"""

from core.config import IDENTITY_FIELDS, TRAIT_QUESTIONS, TRAIT_LEVELS
from core.personality import (
    save_character, list_characters, read_traits, read_temperament,
)
from nodes.temperament import build_character, genesis
from core.encoder import encode_text
from core.appraisal import appraise
from nodes.cognition import interpret, lens
from nodes.llm_impact import push_from_text
from nodes.values import values_push_from_text
from nodes.moral import moral_push_from_text
from core.updater import update_personality
from nodes.character import (
    default_character, update_values, update_moral, note_habit, form_beliefs,
    read_values, read_moral, read_beliefs,
)
from core.memory import create_memory, recurrence, recall, load_memories


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

    character = personality.get("character") or {}
    strong = [(n, v) for n, v in read_values(character).items() if abs(v) > 0.005]
    if strong:
        print("\nCHARACTER VALUES    (Schwartz, strongest first):")
        for name, value in strong[:5]:
            print(f"  {name:18s}: {value:+.3f}")
    strong_moral = [(n, v) for n, v in read_moral(character).items() if abs(v) > 0.005]
    if strong_moral:
        print("\nMORAL OUTLOOK       (Moral Foundations, strongest first):")
        for name, value in strong_moral[:6]:
            print(f"  {name:18s}: {value:+.3f}")
    beliefs = read_beliefs(character)
    if beliefs:
        print("\nBELIEFS (conviction, strongest first):")
        for b in beliefs[:5]:
            print(f"  {b['confidence']:+.2f}  {b['statement']}  (x{b.get('count', 1)})")
    habits = character.get("habits") or []
    if habits:
        print("\nHABITS: " + ", ".join(f"{h['text']} (x{h['count']})" for h in habits))
    print()


# --- Creation questionnaire -------------------------------------------------
def ask_identity():
    print("\n-- Who are they? (press Enter to skip a field) --")
    fields = {}
    while not fields.get("name"):
        name = input("  Name: ").strip()
        if name:
            fields["name"] = name
    for key, label in IDENTITY_FIELDS:
        if key == "name":
            continue
        value = input(f"  {label}: ").strip()
        if value:
            fields[key] = value
    return fields


def ask_traits():
    print("\n-- What are they like? (1-5, Enter = 3 = balanced) --")
    mu = {}
    for key, name, low, high in TRAIT_QUESTIONS:
        answer = input(f"  {name:18s} 1={low}  ...  5={high}: ").strip()
        level = int(answer) if answer.isdigit() and 1 <= int(answer) <= 5 else 3
        mu[key] = TRAIT_LEVELS[level]
    return mu


def create_new():
    identity = ask_identity()
    mu = ask_traits()
    personality, _, _ = build_character(identity, mu)
    path = save_character(personality)
    print(f"\nCreated '{identity['name']}'  ->  {path}")
    print_state(personality)
    return personality


# --- Selection --------------------------------------------------------------
def choose_existing():
    characters = list_characters()
    if not characters:
        print("  (no saved characters yet -- let's create one)")
        return None
    print("\n-- Existing characters --")
    for i, character in enumerate(characters, 1):
        identity = character.get("identity") or {}
        name = identity.get("name") or "unnamed"
        extra = ", ".join(f"{k}={identity[k]}" for k in ("age", "origin") if identity.get(k))
        count = character.get("experience_count", 0)
        print(f"  {i}) {name}" + (f"  ({extra})" if extra else "") + f"   [{count} experiences]")
    answer = input("  pick a number: ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(characters):
        return characters[int(answer) - 1]
    print("  invalid choice.")
    return None


def start():
    print("\n=== MindForm ===")
    print("  1) Use existing character")
    print("  2) Create new character")
    if input("> ").strip() == "1":
        chosen = choose_existing()
        if chosen is not None:
            print(f"\nContinuing as {chosen['identity'].get('name', 'unnamed')}.")
            print_state(chosen)
            return chosen
    return create_new()


# --- Talk loop --------------------------------------------------------------
def run():
    personality = start()
    print("Now talk -- each line is an experience.  Commands: /switch /new /show /exit\n")

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

        if text == "/switch":
            chosen = choose_existing()
            if chosen is not None:
                personality = chosen
                print(f"\nNow talking with {personality['identity'].get('name', 'unnamed')}.")
                print_state(personality)
            continue

        if text.startswith("/genesis"):
            bio = text[len("/genesis"):].strip()
            if not bio:
                print("usage: /genesis <a short biography>")
                continue
            personality, source, reasoning = genesis(bio)
            save_character(personality)
            print(f"\nBorn a character (via {source})."
                  + (f"  --  {reasoning}" if reasoning else ""))
            print_state(personality)
            continue

        # --- experience -> personality update ---
        name = (personality.get("identity") or {}).get("name")
        embedding = encode_text(text)
        seen = recurrence(embedding, name=name)
        recalled = recall(embedding, name=name)              # past only (before interpret)
        appraisal = interpret(appraise(text), personality, recalled=recalled)  # traits + memory
        view = lens(personality, recalled=recalled)

        push, source, reasoning = push_from_text(text, appraisal, lens=view)
        personality = update_personality(personality, push)

        # CHARACTER: the same experience forms the Schwartz values and the moral
        # outlook, and a recurring one (this occurrence included) settles into a habit.
        values_push, _, _ = values_push_from_text(text, appraisal, lens=view)
        moral_push, _, _ = moral_push_from_text(text, appraisal, lens=view)
        character = update_values(personality.get("character") or default_character(), values_push)
        character = update_moral(character, moral_push)
        character = note_habit(character, text, seen + 1)
        personality = {**personality, "character": character}

        create_memory(text, embedding, appraisal, push, personality, name=name)
        # BELIEF: turn this experience (now in memory) -- and any offline backlog --
        # into beliefs, deduped by embedding similarity. A no-op without the LLM.
        character = form_beliefs(personality["character"], load_memories(name), embedder=encode_text)
        personality = {**personality, "character": character}
        save_character(personality)

        print("\n--- RESULT ---")
        print(f"Seen before: {seen}")
        print(f"Push via: {source}" + (f"  --  {reasoning}" if reasoning else ""))
        print("\nPush:")
        for k, v in push.items():
            print(f"  {k:10s}: {v:+.3f}")
        print_state(personality)


if __name__ == "__main__":
    run()
