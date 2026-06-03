"""End-to-end MindForm interaction loop.

    text -> MiniLM embedding
         -> appraisal vector   (appraisal.py: heuristic now, learned head later)
         -> signed push        (impact.py: appraisal -> per-trait push)
         -> two-timescale update of the five traits (updater.py)
         -> memory             (memory.py)

The formed personality is observed directly (evaluation.py), not estimated from
text. There is no text->trait model in the loop. See docs/RESEARCH_REVIEW.md.

Run with: python simulation.py
"""

from encoder import encode_text
from personality import load_personality, save_personality, read_traits
from appraisal import appraise, blend_appraisal, bias_appraisal
from impact import impact
from updater import update_personality
from memory import create_memory, recurrence, retrieve_similar
from response import generate_response


def run_interaction(text):
    personality = load_personality()

    embedding = encode_text(text)
    neighbors = retrieve_similar(embedding)
    appraisal = blend_appraisal(appraise(text), [m["appraisal"] for m in neighbors])  # memory tint
    appraisal = bias_appraisal(appraisal, personality)                                # who-he-was tint
    seen = recurrence(embedding)

    push = impact(appraisal, personality)
    personality = update_personality(personality, push, recurrence=seen)

    create_memory(text, embedding, appraisal, push, personality)
    save_personality(personality)

    print(f"\nINPUT: {text}   (seen {seen} similar before)")
    print("\nPUSH  ->  MOOD (state)  ->  DISPOSITION (trait):")
    for dim, layers in personality["traits"].items():
        print(f"  {dim}  push {push[dim]:+.3f}  ->  mood {layers['state']:+.3f}  ->  trait {layers['trait']:+.3f}")
    print(f"\nRESPONSE:\n  {generate_response(read_traits(personality))}")

    return personality


if __name__ == "__main__":
    run_interaction("I went to a party and had fun.")
