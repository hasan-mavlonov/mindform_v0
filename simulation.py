"""End-to-end MindForm interaction loop.

    text -> MiniLM embedding
         -> signed push        (llm_impact.py: DeepSeek delta x rate -- primary;
                                appraisal.py -> impact.py heuristic -- fallback)
         -> diminishing-returns update of the five traits (updater.py)
         -> memory             (memory.py)

The Pandora text->OCEAN model is used only as a read-only readout (readout.py).

Run with: python simulation.py
"""

from encoder import encode_text
from personality import load_personality, save_personality, read_traits
from appraisal import appraise
from llm_impact import push_from_text
from updater import update_personality
from memory import create_memory, recurrence
from response import generate_response


def run_interaction(text):
    personality = load_personality()

    embedding = encode_text(text)
    appraisal = appraise(text)
    seen = recurrence(embedding)

    push, source, reasoning = push_from_text(text, appraisal)
    personality = update_personality(personality, push)

    create_memory(text, embedding, appraisal, push, personality)
    save_personality(personality)

    print(f"\nINPUT: {text}   (seen {seen} similar before)")
    print(f"PUSH via {source}" + (f"  --  {reasoning}" if reasoning else ""))
    print("\nPUSH  ->  TRAIT:")
    for dim, value in personality["traits"].items():
        print(f"  {dim}  push {push[dim]:+.3f}  ->  trait {value:+.3f}")
    print(f"\nRESPONSE:\n  {generate_response(read_traits(personality))}")

    return personality


if __name__ == "__main__":
    run_interaction("I went to a party and had fun.")
