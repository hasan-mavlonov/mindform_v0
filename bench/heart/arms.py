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


# The D2 section header, used both to render the block and to find it again when
# leak-checking. The leak checker must know exactly which span of the prompt is
# engine-derived state, so this string is the contract between the two.
STATE_BLOCK_MARKER = "## Formed Disposition"


def _state_block(state):
    """D2 only: the persistent state MindForm formed from these same memories.

    Every number here was produced by the engine from the raw memories. None of
    it comes from HEART's withheld labels.
    """
    t = state["traits"]
    lines = [
        STATE_BLOCK_MARKER,
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
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def split_state_block(prompt):
    """Separate the D2 Formed Disposition block from the rest of the prompt.

    Returns (block, rest). ``block`` is "" when the prompt carries no state
    section. The split matters because D2's whole purpose is to state MindForm's
    own formed OCEAN numbers, so that one span is the only place in any prompt
    where a trait word legitimately sits next to a number.
    """
    i = prompt.find(STATE_BLOCK_MARKER)
    if i < 0:
        return "", prompt
    j = prompt.find("\n\n## ", i)
    if j < 0:
        j = len(prompt)
    return prompt[i:j], prompt[:i] + prompt[j:]


def _numbers_after(text, word, window=16):
    """Every numeric token appearing just after each occurrence of ``word``."""
    out = []
    for m in re.finditer(re.escape(word), text, re.I):
        for tok in _NUMBER_RE.findall(text[m.end(): m.end() + window]):
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def check_state_provenance(block, state, live_traits=None):
    """Confirm the D2 block is exactly what the engine state renders to.

    This is what lets the numeric trait check below skip the block without
    weakening anything. Rather than pattern-matching the numbers, it proves
    where they came from: the block must be byte-identical to ``_state_block``
    run over the engine's own state, and that state's traits must equal the
    traits currently on disk for the character. A HEART value could only pass
    both if the engine had independently formed that exact number, which is the
    definition of MindForm-derived rather than leaked.
    """
    hits = []
    if block and not state:
        hits.append("state:block present but no engine state was supplied")
        return hits
    if not block:
        return hits
    if block != _state_block(state):
        hits.append("state:block does not match the engine state it claims to render")
    if live_traits is not None:
        for d, v in (state.get("traits") or {}).items():
            if round(float(live_traits.get(d, 0.0)), 4) != round(float(v), 4):
                hits.append(f"state:{d} does not match the character on disk")
    return hits


def check_state_not_seeded(state, char):
    """Catch a character that was seeded with HEART's labels rather than formed.

    Provenance proves the D2 numbers came off the engine's own disk, which is
    the right question for a leak in a PROMPT. It cannot see a leak introduced
    earlier, at formation time, by building the character from the withheld
    big_five instead of from the raw memories. One signature of that is
    unmistakable: all five formed traits landing exactly on HEART's five hidden
    values. Independent formation reproducing all five to two decimals is not
    something that happens by chance.
    """
    gt = char.get("big_five") or {}
    traits = (state or {}).get("traits") or {}
    letter = {"openness": "O", "conscientiousness": "C", "extraversion": "E",
              "agreeableness": "A", "neuroticism": "N"}
    pairs = [(float(v), float(traits[letter[k]]))
             for k, v in gt.items() if letter.get(k) in traits]
    if len(pairs) == len(letter) and all(round(a, 2) == round(b, 2) for a, b in pairs):
        return ["state:every formed trait equals HEART's withheld big_five exactly "
                "-- the character looks seeded, not formed"]
    return []


def leak_check(prompt, forbidden, char, state=None, live_traits=None):
    """Hard check that no withheld label reached the prompt.

    Returns (ok, detail). Any hit invalidates the run.

    Every withheld STRING -- descriptions, answer keys, source_character, trait
    tags, the parenthetical in ``name`` -- is scanned for across the WHOLE
    prompt, D2's state block included. Only the numeric big_five heuristic is
    scoped to outside that block, because a number next to a trait word is
    exactly what D2 is supposed to contain: MindForm's own formed disposition.
    The block is not simply trusted for being D2's -- it is checked by
    provenance instead, which is the stricter test of the two.
    """
    hits = []
    low = prompt.lower()
    for s in forbidden:
        if s and len(s) > 3 and s.lower() in low:
            hits.append(s[:60])
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

    block, rest = split_state_block(prompt)
    hits.extend(check_state_provenance(block, state, live_traits))
    if block:
        hits.extend(check_state_not_seeded(state, char))

    # Withheld big_five values, e.g. "conscientiousness: 0.5", anywhere OUTSIDE
    # the engine's own state block. Compared as numbers rather than as a regex
    # over the raw value: the old pattern interpolated the value unescaped, so
    # HEART's "0.5" became `0.5` with "." as a wildcard and matched the first
    # three characters of any 0.5x, and it ignored sign, so a MindForm-derived
    # -0.53 was flagged as HEART's +0.5. Both are real numbers the engine can
    # legitimately form.
    for trait, val in (char.get("big_five") or {}).items():
        try:
            want = float(val)
        except (TypeError, ValueError):
            continue
        if any(n == want for n in _numbers_after(rest, trait)):
            hits.append(f"big_five:{trait}={val}")
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
    # Whether the model ever actually answered. A reply we could not parse is a
    # measurement (the model had its chance and produced nothing usable); a
    # transport or auth failure is not a measurement at all, and must never be
    # scored as a wrong answer.
    responded = False

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
            responded = True
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
        "responded": responded,
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
