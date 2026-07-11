"""The appraisal distillation log: every LLM-labelled reading becomes training data.

Appends ``{"text": ..., "appraisal": {...}}`` rows to ``data/appraisal_dataset.jsonl`` --
the exact file ``bootstrap/train_appraisal_head.py`` trains the offline head from. The
online model is the teacher: chat with a key set and the dataset grows with real usage;
retrain locally when it has grown and ``core.appraisal.appraise`` automatically prefers
the trained head. ``bootstrap/distill_appraisal_corpus.py`` batch-manufactures additional
synthetic rows for volume and diversity.

THE IRON RULE: only LLM-labelled rows enter the log (``maybe_log`` gates on the source).
The head must never train on its own -- or the lexicon's -- outputs, or distillation
degenerates into a feedback loop.

The RAW (pre-lens) appraisal is what gets logged: the head learns the character-independent
reading; the character's tilts are applied on top by cognition, per character, at run time.
Duplicate texts are left in place -- repetition is honest frequency weighting for training.
Logging must never break a turn: every failure is swallowed with a warning.
"""

import json
import logging
import os

from core.config import LLM_LABEL

log = logging.getLogger("mindform.appraisal_log")

DATASET_PATH = "data/appraisal_dataset.jsonl"


def log_appraisal(text, appraisal, path=DATASET_PATH):
    """Append one (text -> appraisal) training row. Returns True when written."""
    text = (text or "").strip()
    if not text or not appraisal:
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({"text": text, "appraisal": dict(appraisal)}) + "\n")
        return True
    except Exception as exc:
        log.warning("appraisal log failed (%s); continuing", exc)
        return False


def maybe_log(text, appraisal, source, path=DATASET_PATH):
    """Log ONLY when the reading came from the LLM teacher (the iron rule)."""
    if source != LLM_LABEL:
        return False
    return log_appraisal(text, appraisal, path)
