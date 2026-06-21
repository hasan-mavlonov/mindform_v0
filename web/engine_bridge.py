"""Bridge: turn the unchanged MindForm engine into UI-friendly JSON snapshots.

This module is pure orchestration. It calls the engine's own functions in the
same order ``interactive.py`` does -- ``appraise`` -> ``push_from_text`` ->
``update_personality`` -> ``save_character`` (plus the optional encoder/memory
recurrence path) -- and reshapes the result into a single ``snapshot`` dict the
front-end animates against. No personality math lives here; it all stays in the
engine modules. Anything the engine doesn't compute (a conversational reply,
plain-language trait glyphs) is clearly presentation, added on top.

Optional heavy dependencies (sentence-transformers for the encoder, numpy for
memory recurrence) are imported lazily and guarded: when they are absent the
cockpit still runs the full personality-formation pipeline -- it simply skips
the "seen before" recurrence count and the persisted memory log.
"""

import logging

from config import (
    BASIS, BASIS_NAMES, TRAIT_QUESTIONS, TRAIT_LEVELS, IDENTITY_FIELDS,
    APPRAISAL_DIMS, DEFAULT_TAU, VALUES, VALUES_NAMES, MORAL, MORAL_NAMES,
)
from appraisal import appraise
from llm_impact import push_from_text
from values import values_push_from_text
from moral import moral_push_from_text
from updater import update_personality
from temperament import genesis, build_character
from character import (
    default_character, update_values, update_moral, note_habit, form_beliefs,
    higher_order, dominant_value,
)
from personality import (
    save_character, load_character, list_characters,
    read_traits, read_temperament,
)
from web.reply import generate_reply

log = logging.getLogger("mindform.web")

# Plain-language poles per trait, sourced from config.TRAIT_QUESTIONS so the UI
# never invents wording the engine doesn't already use. Shape: {key: (name, low, high)}.
_POLES = {key: (name, low, high) for key, name, low, high in TRAIT_QUESTIONS}

# How big an actual trait move must be to be worth narrating in the banner.
_FORMATION_EPS = 0.005


# --- Config the front-end renders its forms from (single source of truth) ----
def ui_config():
    """Everything the client needs to stay in sync with config.py."""
    return {
        "basis": BASIS,
        "basis_names": BASIS_NAMES,
        "trait_questions": [
            {"key": k, "name": n, "low": low, "high": high}
            for k, n, low, high in TRAIT_QUESTIONS
        ],
        "trait_levels": {str(k): v for k, v in TRAIT_LEVELS.items()},
        "identity_fields": [{"key": k, "label": label} for k, label in IDENTITY_FIELDS],
        "appraisal_dims": APPRAISAL_DIMS,
        "default_tau": DEFAULT_TAU,
    }


# --- Snapshot building -------------------------------------------------------
def _glyph(key, value):
    """The pole a trait currently leans toward, in plain language."""
    _, low, high = _POLES.get(key, ("", "low", "high"))
    return high if value >= 0 else low


def _trait_rows(personality):
    """Five OCEAN traits with current value, baseline (mu), stickiness, poles."""
    traits = personality["traits"]
    temperament = personality["temperament"]
    rows = []
    for key in BASIS:
        name, low, high = _POLES.get(key, (BASIS_NAMES[key].title(), "low", "high"))
        value = traits[key]
        rows.append({
            "key": key,
            "label": BASIS_NAMES[key],
            "name": name,
            "value": value,
            "base": temperament["mu"][key],
            "tau": temperament["tau"][key],
            "low": low,
            "high": high,
            "glyph": _glyph(key, value),
        })
    return rows


def _push_rows(push):
    """The signed per-trait pressure this experience applied (zeros if none)."""
    push = push or {}
    return [{
        "key": key,
        "label": BASIS_NAMES[key],
        "name": _POLES.get(key, (BASIS_NAMES[key].title(),))[0],
        "value": float(push.get(key, 0.0)),
    } for key in BASIS]


def _dominant(trait_rows):
    """The trait furthest from neutral -- the character's defining quality."""
    row = max(trait_rows, key=lambda r: abs(r["value"]))
    return {
        "key": row["key"],
        "name": row["name"],
        "glyph": row["glyph"],
        "value": row["value"],
        "dir": 1 if row["value"] >= 0 else -1,
    }


def _value_rows(character):
    """The ten Schwartz values at their current standing (signed [-1, 1])."""
    values = (character or {}).get("values") or {}
    return [{
        "key": v,
        "label": VALUES_NAMES[v],
        "value": float(values.get(v, 0.0)),
    } for v in VALUES]


def _values_push_rows(push):
    """The signed per-value pressure this experience applied (zeros if none)."""
    push = push or {}
    return [{
        "key": v,
        "label": VALUES_NAMES[v],
        "value": float(push.get(v, 0.0)),
    } for v in VALUES]


def _moral_rows(character):
    """The six Moral Foundations at their current standing (signed [-1, 1])."""
    moral = (character or {}).get("moral") or {}
    return [{
        "key": m,
        "label": MORAL_NAMES[m],
        "value": float(moral.get(m, 0.0)),
    } for m in MORAL]


def _moral_push_rows(push):
    """The signed per-foundation pressure this experience applied (zeros if none)."""
    push = push or {}
    return [{
        "key": m,
        "label": MORAL_NAMES[m],
        "value": float(push.get(m, 0.0)),
    } for m in MORAL]


def _belief_rows(character):
    """The character's beliefs, strongest conviction first (for the UI)."""
    beliefs = (character or {}).get("beliefs") or []
    ordered = sorted(beliefs, key=lambda b: -abs(b.get("confidence", 0.0)))
    return [{
        "statement": b.get("statement", ""),
        "confidence": float(b.get("confidence", 0.0)),
        "count": int(b.get("count", 1)),
    } for b in ordered]


def _character_block(personality, *, push=None, source=None, reasoning="", moral_push=None):
    """CHARACTER snapshot: current values + moral outlook, the values' higher-order
    roll-up, the dominant value, the habits formed so far, and the pushes applied."""
    character = personality.get("character") or {}
    return {
        "values": _value_rows(character),
        "higher_order": higher_order(character.get("values") or {}),
        "dominant": dominant_value(character),
        "moral": _moral_rows(character),
        "moral_push": _moral_push_rows(moral_push),
        "beliefs": _belief_rows(character),
        "habits": character.get("habits") or [],
        "push": _values_push_rows(push),
        "source": source,
        "reasoning": reasoning or "",
    }


def _formation(before, after):
    """Narrate the single biggest *actual* trait move (post diminishing-returns)."""
    best_key, best_delta = None, 0.0
    for key in BASIS:
        delta = after["traits"][key] - before.get(key, 0.0)
        if abs(delta) > abs(best_delta):
            best_key, best_delta = key, delta
    if best_key is None or abs(best_delta) < _FORMATION_EPS:
        return None
    name, low, high = _POLES.get(best_key, (BASIS_NAMES[best_key].title(), "low", "high"))
    leaning = high if best_delta >= 0 else low
    return {
        "key": best_key,
        "name": name,
        "delta": best_delta,
        "dir": 1 if best_delta >= 0 else -1,
        "note": f"grew more {leaning}",
    }


def snapshot(personality, *, push=None, appraisal=None, source=None,
             reasoning="", seen=None, formation=None, reply=None,
             values_push=None, values_source=None, values_reasoning="",
             moral_push=None):
    """The full state object the cockpit reads. Engine values pass straight through."""
    trait_rows = _trait_rows(personality)
    identity = dict(personality.get("identity") or {})
    return {
        "name": identity.get("name") or "unnamed",
        "identity": identity,
        "turn": personality.get("experience_count", 0),
        "traits": trait_rows,
        "push": _push_rows(push),
        "appraisal": appraisal,
        "source": source,
        "reasoning": reasoning or "",
        "seen": seen,
        "formation": formation,
        "dominant": _dominant(trait_rows),
        "character": _character_block(
            personality, push=values_push, source=values_source,
            reasoning=values_reasoning, moral_push=moral_push,
        ),
        "reply": reply,
    }


# --- Roster (start screen) ---------------------------------------------------
def roster():
    """Lightweight list of saved characters for the picker."""
    out = []
    for character in list_characters():
        identity = dict(character.get("identity") or {})
        out.append({
            "name": identity.get("name") or "unnamed",
            "identity": identity,
            "turn": character.get("experience_count", 0),
            "traits": _trait_rows(character),
        })
    return out


def load_snapshot(name):
    """First-paint snapshot for an existing character."""
    return snapshot(load_character(name))


# --- Creation paths (mirror interactive.py's three authoring routes) ---------
def create_genesis(bio):
    """Born from a one-line biography (DeepSeek seed, heuristic fallback)."""
    personality, source, reasoning = genesis(bio)
    save_character(personality)
    snap = snapshot(personality, source=source, reasoning=reasoning)
    snap["created_via"] = source
    return snap


def create_manual(identity, levels):
    """Built from explicit identity fields + a 1-5 answer per OCEAN trait.

    ``levels`` maps trait key -> questionnaire answer (1-5); we resolve it to a
    baseline ``mu`` via the engine's own ``TRAIT_LEVELS`` so the mapping never
    diverges from ``interactive.py``.
    """
    mu = {}
    for key in BASIS:
        try:
            level = int(levels.get(key, 3))
        except (TypeError, ValueError):
            level = 3
        level = level if level in TRAIT_LEVELS else 3
        mu[key] = TRAIT_LEVELS[level]
    clean_identity = {k: v for k, v in (identity or {}).items() if v not in (None, "")}
    personality, _, _ = build_character(clean_identity, mu)
    save_character(personality)
    snap = snapshot(personality)
    snap["created_via"] = "manual"
    return snap


# --- The talk loop: one experience -> personality update ---------------------
def _recurrence_and_memory(text, appraisal, push, personality, name):
    """Optional encoder + memory recurrence. Returns ``seen`` or ``None``.

    Mirrors ``interactive.py`` but guarded: if sentence-transformers / numpy are
    not installed, formation still works -- we just can't count recurrences or
    persist the memory log this turn.
    """
    try:
        from encoder import encode_text          # heavy: sentence-transformers
        from memory import create_memory, recurrence  # heavy: numpy
    except Exception as exc:                       # deps absent -> skip cleanly
        log.info("memory/encoder unavailable (%s); skipping recurrence", exc)
        return None
    try:
        embedding = encode_text(text)
        seen = recurrence(embedding, name=name)
        create_memory(text, embedding, appraisal, push, personality, name=name)
        return seen
    except Exception as exc:
        log.warning("recurrence/memory step failed (%s); continuing", exc)
        return None


def _form_beliefs(personality, char_name):
    """Reflection pass: turn unreviewed memories into beliefs (guarded; no-op offline).

    Reads the per-character memory log and the encoder for embedding dedup; if those deps
    are absent there is no backlog to read, so the character is returned unchanged.
    ``form_beliefs`` itself is a no-op when the LLM is unavailable, leaving the watermark.
    """
    character = personality.get("character") or default_character()
    try:
        from memory import load_memories
        from encoder import encode_text
        memories, embedder = load_memories(char_name), encode_text
    except Exception:
        memories, embedder = [], None
    return form_beliefs(character, memories, embedder)


def run_turn(name, message):
    """Feed one experience to a character and return the new state + a reply.

    This is exactly ``interactive.py``'s per-line pipeline:
        appraise -> push_from_text -> update_personality -> (memory) -> save.
    The conversational ``reply`` is presentation layered on top; it never feeds
    back into the personality math.
    """
    text = (message or "").strip()
    personality = load_character(name)
    char_name = (personality.get("identity") or {}).get("name")

    appraisal = appraise(text)
    push, source, reasoning = push_from_text(text, appraisal)
    values_push, values_source, values_reasoning = values_push_from_text(text, appraisal)
    moral_push, _, _ = moral_push_from_text(text, appraisal)

    before_traits = dict(personality["traits"])
    personality = update_personality(personality, push)

    seen = _recurrence_and_memory(text, appraisal, push, personality, char_name)

    # CHARACTER: the same experience forms the values and the moral outlook, and a
    # recurring one (seen before, this occurrence included) settles into a habit.
    character = update_values(personality.get("character") or default_character(), values_push)
    character = update_moral(character, moral_push)
    character = note_habit(character, text, (seen or 0) + 1)
    personality = {**personality, "character": character}

    # BELIEF: the same experience (now in memory) -- and any offline backlog -- forms
    # beliefs; a no-op without the LLM, which the next online turn catches up.
    personality = {**personality, "character": _form_beliefs(personality, char_name)}

    save_character(personality)

    formation = _formation(before_traits, personality)
    reply = generate_reply(personality, text)

    return snapshot(
        personality, push=push, appraisal=appraisal, source=source,
        reasoning=reasoning, seen=seen, formation=formation, reply=reply,
        values_push=values_push, values_source=values_source,
        values_reasoning=values_reasoning, moral_push=moral_push,
    )
