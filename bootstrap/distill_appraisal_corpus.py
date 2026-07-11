"""Manufacture a (text -> appraisal) training corpus by distilling the online model.

No external dataset needed: the LLM both WRITES diverse first-person experience sentences
(a domains x tones grid keeps the corpus from collapsing onto one register) and LABELS each
with the eight appraisal dims (``nodes.llm_appraisal``), appending rows to
``data/appraisal_dataset.jsonl`` -- the exact file ``bootstrap/train_appraisal_head.py``
trains from. Live chat rows (logged automatically by the cockpit whenever a key is set)
supply realism on top; this script supplies volume and diversity.

Run locally with a key set (needs only ``openai``, no torch):

    python bootstrap/distill_appraisal_corpus.py            # ~1000 rows
    python bootstrap/distill_appraisal_corpus.py 200        # smaller run
    python bootstrap/train_appraisal_head.py                # then train the head

Append-only and resumable: rerunning adds rows; duplicates are harmless frequency weight.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import complete_json  # noqa: E402
from nodes.llm_appraisal import _llm_appraise  # noqa: E402
from core.appraisal_log import log_appraisal, DATASET_PATH  # noqa: E402

DOMAINS = [
    "work and ambition", "family", "friendship", "romance and heartbreak",
    "health and the body", "loss and grief", "conflict and confrontation",
    "solitude and quiet", "adventure and travel", "learning and mastery",
    "money and security", "helping and being helped", "failure and embarrassment",
    "recognition and praise", "everyday routine",
]
TONES = [
    "clearly wonderful", "quietly pleasant", "neutral / ambiguous",
    "quietly painful", "clearly awful", "frightening", "a hard-won triumph",
]

WRITER_PROMPT = """You are MindForm's Experience Writer.

Write {n} distinct first-person sentences, each describing ONE everyday life experience
about {domain}, with a tone that is {tone}. One sentence each, plain language, varied
situations, no names, no numbering.

Return ONLY valid JSON, no markdown, exactly:
{{"experiences": ["...", "..."]}}"""


def generate_batch(domain, tone, n=8):
    data = complete_json(WRITER_PROMPT.format(n=n, domain=domain, tone=tone),
                         "Write them now.", temperature=0.9)
    return [s.strip() for s in data.get("experiences", []) if isinstance(s, str) and s.strip()]


def main(target_rows=1000, per_call=8, out=DATASET_PATH):
    written = 0
    cell = 0
    while written < target_rows:
        domain = DOMAINS[cell % len(DOMAINS)]
        tone = TONES[(cell // len(DOMAINS)) % len(TONES)]
        cell += 1
        try:
            sentences = generate_batch(domain, tone, per_call)
        except Exception as exc:
            print(f"  writer failed on [{domain} / {tone}]: {exc}")
            continue
        for text in sentences:
            try:
                appraisal = _llm_appraise(text)      # the teacher labels its own sentence
            except Exception as exc:
                print(f"  labeller failed: {exc}")
                continue
            if log_appraisal(text, appraisal, out):
                written += 1
                if written % 50 == 0:
                    print(f"{written}/{target_rows} rows -> {out}")
            if written >= target_rows:
                break
    print(f"done: {written} rows -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
