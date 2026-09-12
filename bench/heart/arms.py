"""The three arms: prompt construction, the answer call, parsing, leak checking.

Arms A and D1 build a byte-identical prompt template (a port of HEART's
``run_naive_rag.build_prompt``); they differ only in which memories fill the
Past Experiences block. D2 adds one extra section carrying MindForm's formed
persistent state -- which is deliberately NOT protocol-identical to naive RAG
and is therefore always reported as its own arm.
"""

import hashlib
import json
import re
import time

from bench.heart.config import (
    MODEL, TEMPERATURE, MAX_TOKENS, REASONING_EFFORT, estimate_cost,
    FORBIDDEN_CHARACTER_FIELDS, FORBIDDEN_OPTION_FIELDS,
)

SYSTEM_PROMPT = (
    "You are simulating a specific person's inner life and behaviour. "
    "Reply with a single JSON object and nothing else:\n"
    '{\n'
    '  "system_1": "the immediate intuitive reaction",\n'
    '  "system_2": "the considered reasoning after calming down",\n'
    '  "inner_consciousness": "100-150 words, first person, fusing emotional tone, '
    'core reasons and value orientation",\n'
    '  "final_decision": "what the person actually does or says, first person",\n'
    '  "decision_choice": "the single letter (A, B, C or D) of the option that best '
    'matches this person"\n'
    "}"
)


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------
def _memory_block(memories):
    return "\n".join(
        f"  - [{m.get('anon_id','?')}][{m.get('timeline','?')}] {m.get('text','')}"
        for m in memories
    )


def _options_block(options):
    return "\n\n".join(f"{o['label']}. {o['content']}" for o in options)


def build_prompt(char_public, scenario, memories, options, state=None):
    """Port of HEART's build_prompt. ``state`` is D2-only and appends one section."""
    setting = scenario.get("setting") or {}
    trigger = scenario.get("trigger_event") or {}

    prompt = f"""## Background
- Character ID: {char_public.get('id', 'N/A')}
- Occupation: {char_public.get('occupation', 'N/A')}

## Key Social Relationships
  N/A

## Past Experiences
The following are important fragments from this person's life — use them to understand who this person is:
{_memory_block(memories)}

## Current Situation
Scene: {scenario.get('name', 'Unknown')}
Location: {setting.get('location', 'Unknown')} | Time: {setting.get('time', 'Unknown')} | Atmosphere: {setting.get('atmosphere', 'Unknown')}

Context: {scenario.get('context_text', 'Unknown')}

## Trigger Event
Sender: {trigger.get('sender', 'Unknown')}
Message: {trigger.get('message_content', 'Unknown')}
Action required: {trigger.get('action_required', 'Unknown')}

## Task
Using the experiences above, understand this person's thinking patterns, emotional tendencies, and behavioural habits, then simulate the real reaction they would have in the current situation.

Requirements:
1. System 1 (intuitive impulse): the person's first reaction; cite the activated memories.
2. System 2 (rational analysis): how the person would analyse and reason after calming down.
3. Final Decision: two parts — inner_consciousness is the inner monologue 'I plan to do/say ...' (the last layer of consciousness before outward behaviour, fusing emotional tone, core reasons, and value orientation); response_text is what the person actually says/sends."""

    if state:
        prompt += "\n\n" + _state_block(state)

    prompt += f"""

## Behavioural Decision Options
Below are possible behavioural decisions different people might take in this scenario. Pick the one that best matches you (in this character's role) and output the corresponding letter in the decision_choice field:

{_options_block(options)}"""
    return prompt


def _state_block(state):
    """D2 only: the persistent state MindForm formed from these same memories.

    Every number here was produced by the engine from the raw memories. None of
    it comes from HEART's withheld labels.
    """
    t = state["traits"]
    lines = [
        "## Formed Disposition",
        "This is the persistent disposition that has formed in this person over the "
        "experiences above (each −1..+1, 0 = unremarkable). Treat it as who they have "
        "become, not as instructions:",
        "  Openness {O:+.2f} | Conscientiousness {C:+.2f} | Extraversion {E:+.2f} | "
        "Agreeableness {A:+.2f} | Neuroticism {N:+.2f}".format(**t),
    ]
    if state.get("top_values"):
        lines.append("  What they have come to prize: "
                     + ", ".join(f"{n} ({v:+.2f})" for n, v in state["top_values"]))
    if state.get("top_moral"):
        lines.append("  Moral weighting: "
                     + ", ".join(f"{n} ({v:+.2f})" for n, v in state["top_moral"]))
    if state.get("top_drives"):
        lines.append("  Currently unmet needs: "
                     + ", ".join(f"{n} ({v:.2f})" for n, v in state["top_drives"]))
    si = state.get("self_image") or {}
    if si:
        lines.append("  How they see themselves: "
                     + ", ".join(f"{k} {si.get(k, 0.0):+.2f}" for k in "OCEAN")
                     + f" | self-regard {state.get('esteem', 0.0):+.2f}")
    if state.get("beliefs"):
        lines.append("  Beliefs they hold: " + "; ".join(state["beliefs"][:3]))
    return "\n".join(lines)


def prompt_sha(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# leak checking
# --------------------------------------------------------------------------
def leak_check(prompt, forbidden, char):
    """Hard check that no withheld label reached the prompt.

    Returns (ok, detail). Any hit invalidates the run.
    """
    hits = []
    low = prompt.lower()
    for s in forbidden:
        if s and len(s) > 3 and s.lower() in low:
            hits.append(s[:60])
    # numeric big_five values, e.g. "0.95", only flagged next to a trait word
    for trait, val in (char.get("big_five") or {}).items():
        pat = re.compile(rf"{trait}\D{{0,12}}{val}", re.I)
        if pat.search(prompt):
            hits.append(f"big_five:{trait}={val}")
    for field in FORBIDDEN_CHARACTER_FIELDS:
        v = char.get(field)
        if isinstance(v, str) and len(v) > 20 and v[:40].lower() in low:
            hits.append(f"field:{field}")
    for token in ("correct_answer", "is_correct", "source_character",
                  "_N_HIGH_", "_N_LOW_", "_C_HIGH_", "_C_LOW_", "_E_HIGH_",
                  "_E_LOW_", "_A_HIGH_", "_A_LOW_", "_O_HIGH_", "_O_LOW_",
                  "_NEUTRAL_"):
        if token.lower() in low:
            hits.append(f"token:{token}")
    return (len(hits) == 0), "; ".join(sorted(set(hits))[:5])


def assert_no_forbidden_fields(options):
    """Options handed to a prompt must never carry the answer key."""
    for o in options:
        for f in FORBIDDEN_OPTION_FIELDS:
            if f in o:
                raise AssertionError(f"option carries forbidden field {f!r}")


# --------------------------------------------------------------------------
# the answer call
# --------------------------------------------------------------------------
def _client():
    from openai import OpenAI
    from core.config import LLM_API_KEY, LLM_BASE_URL
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


_LETTER = re.compile(r"\b([ABCD])\b")
# Survives a truncated or malformed object: find the field directly.
_CHOICE_FIELD = re.compile(r'"decision_choice"\s*:\s*"?\s*([ABCD])', re.I)


def parse_choice(raw):
    """Pull decision_choice out of the reply. Returns (letter|None, how, error)."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.startswith("```"))
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        c = str(obj.get("decision_choice", "")).strip().upper()
        m = _LETTER.search(c)
        if m:
            return m.group(1), "json", None
        m2 = _CHOICE_FIELD.search(text)
        if m2:
            return m2.group(1).upper(), "field-regex", "decision_choice not A-D in parsed object"
        return None, "json", "decision_choice missing or not A-D"
    except Exception as exc:
        m = _CHOICE_FIELD.search(text)
        if m:
            return m.group(1).upper(), "field-regex", f"json parse failed: {type(exc).__name__}"
        return None, "none", f"unparseable: {type(exc).__name__}"


def answer(prompt, temperature=None, max_tokens=None, retries=1):
    """One answer call. Returns a dict with the choice, raw text, usage and timing."""
    temperature = TEMPERATURE if temperature is None else temperature
    max_tokens = MAX_TOKENS if max_tokens is None else max_tokens
    client = _client()

    errors = []
    t0 = time.time()
    raw, usage = "", {"input_tokens": None, "output_tokens": None}
    choice, how = None, "none"

    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            # The default model is a thinking model: without this, reasoning
            # consumes the whole max_tokens budget and the visible JSON is
            # truncated (measured: 1600 budget -> 62 output tokens). Applied
            # identically to every arm and recorded in each event.
            if REASONING_EFFORT:
                kwargs["reasoning_effort"] = REASONING_EFFORT
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content or ""
            u = getattr(resp, "usage", None)
            if u:
                usage = {"input_tokens": getattr(u, "prompt_tokens", None),
                         "output_tokens": getattr(u, "completion_tokens", None)}
            choice, how, err = parse_choice(raw)
            if err:
                errors.append(f"attempt{attempt}: {err}")
            if choice:
                break
        except Exception as exc:
            errors.append(f"attempt{attempt}: {type(exc).__name__}: {exc}")
            time.sleep(1.5 * (attempt + 1))

    latency_ms = (time.time() - t0) * 1000.0
    return {
        "choice": choice,
        "parse_method": how,
        "raw_output": raw,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": estimate_cost(usage["input_tokens"] or 0, usage["output_tokens"] or 0),
        "latency_ms": round(latency_ms, 1),
        "model": MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": REASONING_EFFORT,
        "errors": errors,
    }
