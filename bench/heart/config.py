"""Configuration for the HEART-Bench integration.

Everything here is benchmark-side only; nothing in core/, nodes/ or web/ is
touched. Paths to the HEART-Bench checkout come from HEART_BENCH_PATH.
"""

import os

# --- where HEART-Bench lives (an external clone, pinned by commit) ----------
HEART_PATH = os.environ.get(
    "HEART_BENCH_PATH",
    "/tmp/claude-0/-home-user-mindform-v0/6f2b3bdc-3954-5605-b6f5-1225ff9776a3/scratchpad/HEART-BENCH",
)

# --- where our results go ---------------------------------------------------
RESULTS_ROOT = os.environ.get("HEART_RESULTS", "data/benchmark/heart")

# --- model under test (identical for every arm) -----------------------------
# Defaults follow the engine's own resolution order so the benchmark and the
# product speak to the same endpoint unless deliberately overridden.
MODEL = os.environ.get("HEART_MODEL") or os.environ.get("LLM_MODEL") or "gemini-3.5-flash"
TEMPERATURE = float(os.environ.get("HEART_TEMPERATURE", "0"))
MAX_TOKENS = int(os.environ.get("HEART_MAX_TOKENS", "2000"))

# The default Gemini model is a thinking model and `max_tokens` caps thinking +
# output together, so reasoning silently eats the visible answer (measured:
# max_tokens=1600 -> 62 output tokens, finish_reason="length"). "none" disables
# it. Applied identically to every arm; set HEART_REASONING_EFFORT=low|medium
# to re-enable. Recorded in every event so it is never a hidden variable.
REASONING_EFFORT = os.environ.get("HEART_REASONING_EFFORT", "none").strip() or None

# HEART's own runners hardcode temperature 0 in llm_client.call_llm while their
# RESULTS.md reports 0.7. We default to 0 (their shipped code) and record the
# value in every event so the discrepancy is never silent.
TEMPERATURE_NOTE = (
    "HEART llm_client.call_llm hardcodes temperature=0; their RESULTS.md reports 0.7. "
    "We use the shipped-code value and log it per event."
)

# --- retrieval --------------------------------------------------------------
TOP_K = int(os.environ.get("HEART_TOP_K", "30"))  # matches naive_rag --top-k default

# --- cost rates (USD per 1M tokens). Unset -> cost reported as None ---------
def _rate(name):
    v = os.environ.get(name)
    try:
        return float(v) if v else None
    except ValueError:
        return None

RATE_IN = _rate("HEART_RATE_IN")
RATE_OUT = _rate("HEART_RATE_OUT")


def estimate_cost(input_tokens, output_tokens):
    """USD estimate, or None when rates are not configured."""
    if RATE_IN is None or RATE_OUT is None:
        return None
    return (input_tokens / 1e6) * RATE_IN + (output_tokens / 1e6) * RATE_OUT


# --- fields of the HEART character record that may reach a prompt -----------
# Mirrors run_naive_rag.build_basic_info exactly: id + occupation only.
ALLOWED_CHARACTER_FIELDS = ("id", "occupation")

# Fields that must NEVER reach the model. HEART's own runners call these the
# "answer profile"; source_character and correct_answer are the answer key.
FORBIDDEN_CHARACTER_FIELDS = (
    "big_five", "description", "self_value_logic", "core_patterns", "name",
)
FORBIDDEN_OPTION_FIELDS = ("is_correct", "source_character")
FORBIDDEN_FILES = ("ground_truth.json", "activated_memories_step1.json",
                   "activated_memories_step2.json")
