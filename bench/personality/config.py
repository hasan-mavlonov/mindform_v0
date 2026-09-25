"""Configuration for personality-instrument administration.

Deliberately independent of bench/heart/config.py -- reusing HEART's benchmark
config here would be a needless coupling between two things this package is
built to keep apart. Only the true product-wide LLM plumbing (core.config,
core.llm) is shared, the same way every formation node already shares it.
"""

import os

from core.config import LLM_MODEL

# Same model as the rest of MindForm by default; overridable independently of
# HEART's own HEART_MODEL so the two tests can be pointed at different models
# if that's ever useful, without either affecting the other.
MODEL = os.environ.get("PERSONALITY_MODEL") or LLM_MODEL

# "Use deterministic model settings where possible": temperature 0 for every
# item response, so a re-run against an unchanged frozen snapshot is expected
# to reproduce the same answers (LLM provider nondeterminism aside -- the same
# caveat bench/heart's own report.py already documents for its own temp-0 runs).
TEMPERATURE = float(os.environ.get("PERSONALITY_TEMPERATURE", "0"))
MAX_TOKENS = int(os.environ.get("PERSONALITY_MAX_TOKENS", "512"))

RESULTS_ROOT = os.environ.get("PERSONALITY_RESULTS", "data/benchmark/personality")
