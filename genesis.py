"""Birth a character from a biography (Temperament / genesis -- Slice 1).

    python genesis.py "Aisha, 24, an anxious but disciplined medical student..."
        -> seeds identity + OCEAN baseline (mu) + per-trait stickiness (tau),
           starts the current traits AT the baseline (x = mu), and saves the
           character to data/personality.json.

    python genesis.py
        -> demo: two contrasting bios side by side, showing they yield distinct
           temperaments (nothing is saved).

The primary seed is DeepSeek; without a key it falls back to a heuristic, so this
runs with no network.
"""

import sys

from config import BASIS, BASIS_NAMES
from temperament import genesis
from personality import save_personality


def _print_character(personality, source):
    identity = personality["identity"]
    temperament = personality["temperament"]
    print(f"  source  : {source}")
    facts = ", ".join(f"{k}={v}" for k, v in identity.items() if v and k != "bio")
    if facts:
        print(f"  identity: {facts}")
    print("  trait              baseline(mu)  stickiness(tau)   start(x)")
    for d in BASIS:
        print(f"   {BASIS_NAMES[d]:16s}  {temperament['mu'][d]:+.2f}          "
              f"{temperament['tau'][d]:.2f}            {personality['traits'][d]:+.2f}")


def _demo():
    bios = [
        "Aisha, a shy, anxious, deeply creative poet who grew up sheltered and sensitive.",
        "Marcus, a bold, outgoing, disciplined athlete, calm under pressure and endlessly social.",
    ]
    for bio in bios:
        personality, source, _ = genesis(bio)
        print("\n" + "=" * 72)
        print(bio)
        _print_character(personality, source)


def main():
    if len(sys.argv) > 1:
        bio = " ".join(sys.argv[1:])
        personality, source, reasoning = genesis(bio)
        save_personality(personality)
        print(f"\nBorn a character from:\n  {bio}\n")
        _print_character(personality, source)
        if reasoning:
            print(f"\n  reasoning: {reasoning}")
        print("\nSaved -> data/personality.json")
    else:
        _demo()


if __name__ == "__main__":
    main()
