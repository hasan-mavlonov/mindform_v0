"""CHARACTER: the values, moral outlook, beliefs, and habits a person accumulates.

Where ``temperament`` is the innate OCEAN baseline a character is *born* with, CHARACTER
is what they *become* by living:

  * values  -- the ten Schwartz basic values (``config.VALUES``),
  * moral   -- the six Moral Foundations / moral outlook (``config.MORAL``),
  * beliefs -- open-ended propositions they come to hold (``beliefs.py``),
  * habits  -- the recurring experiences they fall into.

Values and moral foundations form by the SAME diminishing-returns dynamics as the traits
(``updater.apply_diminishing``). Beliefs are an open store: the LLM extracts propositions
from each experience and their conviction accumulates with the same diminishing returns;
experiences logged while the LLM is offline are turned into beliefs later by ``form_beliefs``
(a reflection pass over the memory backlog, tracked by ``character["beliefs_reviewed"]``).
A recurring experience (``memory.recurrence``) past ``config.HABIT_MIN_RECURRENCE`` becomes
a named habit.

State lives in ``personality["character"]`` and persists with the rest of the character.
Values/foundations start at 0 and beliefs at none -- earned, not innate -- which is exactly
what separates CHARACTER (experience) from TEMPERAMENT (biology).
"""

import logging
import re

from core.config import (
    VALUES, VALUES_NAMES, VALUES_HIGHER_ORDER, MORAL, MORAL_NAMES,
    HABIT_MIN_RECURRENCE, LLM_FORMATION_RATE, BELIEF_SIM_THRESHOLD, BELIEF_BACKLOG_CAP,
)
from core.updater import apply_diminishing, clamp
from nodes.beliefs import extract_beliefs

log = logging.getLogger("mindform.character")


def default_character():
    """A blank character layer: neutral values + foundations, no beliefs or habits."""
    return {
        "values": {v: 0.0 for v in VALUES},
        "moral": {m: 0.0 for m in MORAL},
        "beliefs": [],
        "beliefs_reviewed": 0,
        "habits": [],
    }


def update_values(character, push):
    """Return a new character with each value moved by its push (input unchanged)."""
    values = apply_diminishing(character.get("values") or {}, push, VALUES)
    return {**character, "values": values}


def update_moral(character, push):
    """Return a new character with each moral foundation moved by its push (input unchanged)."""
    moral = apply_diminishing(character.get("moral") or {}, push, MORAL)
    return {**character, "moral": moral}


def _normalize(text):
    return " ".join((text or "").lower().split())


def note_habit(character, text, recurrence_count, *, min_recurrence=HABIT_MIN_RECURRENCE):
    """Register or refresh a habit once an experience has recurred enough.

    ``recurrence_count`` is how many times this kind of experience has now been seen
    (this occurrence included). Below ``min_recurrence`` nothing changes; at or above
    it the experience is recorded as a habit, deduped by normalized text and keeping
    the highest count seen. Returns a new character (input unchanged).
    """
    if recurrence_count < min_recurrence:
        return character
    key = _normalize(text)
    habits = [dict(h) for h in character.get("habits") or []]
    for habit in habits:
        if habit.get("key") == key:
            habit["count"] = max(habit.get("count", 0), recurrence_count)
            habit["text"] = text
            break
    else:
        habits.append({"key": key, "text": text, "count": recurrence_count})
    return {**character, "habits": habits}


# --- Beliefs: an open propositional store --------------------------------------
def _belief_key(statement):
    """Punctuation-insensitive, lowercased key for offline (text) belief dedup."""
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (statement or "").lower()).split())


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _find_belief(beliefs, statement, embedder):
    """The existing belief this statement matches, or None. Embedding-similarity when an
    ``embedder`` (encoder) is available; punctuation-insensitive text match otherwise."""
    if not beliefs:
        return None
    if embedder is None:
        key = _belief_key(statement)
        return next((b for b in beliefs if _belief_key(b.get("statement", "")) == key), None)
    target = embedder(statement)
    best, best_sim = None, 0.0
    for belief in beliefs:
        sim = _cosine(target, embedder(belief.get("statement", "")))
        if sim > best_sim:
            best, best_sim = belief, sim
    return best if best_sim >= BELIEF_SIM_THRESHOLD else None


def update_beliefs(character, extracted, embedder=None):
    """Merge extracted ``{statement, confidence}`` beliefs into the store.

    A new statement is added; a matching one is reinforced. ``confidence`` is the per-
    occurrence delta in [-1, 1]; conviction accumulates with the SAME diminishing returns
    as the traits/values (scaled by ``LLM_FORMATION_RATE``). Returns a new character.
    """
    beliefs = [dict(b) for b in character.get("beliefs") or []]
    for item in extracted or []:
        statement = (item.get("statement") or "").strip()
        if not statement:
            continue
        push = clamp(LLM_FORMATION_RATE * float(item.get("confidence", 0.0)))
        match = _find_belief(beliefs, statement, embedder)
        if match is not None:
            match["confidence"] = clamp(match["confidence"] + push * (1 - abs(match["confidence"])))
            match["count"] = match.get("count", 0) + 1
        else:
            beliefs.append({"statement": statement, "confidence": clamp(push), "count": 1})
    return {**character, "beliefs": beliefs}


def form_beliefs(character, memories, embedder=None):
    """Turn unreviewed memories into beliefs (the reflection pass).

    Walks ``memories[beliefs_reviewed:]`` (capped at ``BELIEF_BACKLOG_CAP`` per call),
    extracting beliefs from each text and merging them, then advances the
    ``beliefs_reviewed`` watermark over only what it processed -- so experiences logged
    while the LLM was unavailable are picked up by a later pass instead of being lost.
    A no-op (and the watermark is left untouched) when the LLM is unavailable.
    """
    reviewed = character.get("beliefs_reviewed", 0)
    backlog = (memories or [])[reviewed: reviewed + BELIEF_BACKLOG_CAP]
    processed = reviewed
    for memory in backlog:
        try:
            extracted, _ = extract_beliefs(memory.get("text", ""))
        except Exception as exc:        # no LLM / failure -> leave the rest for later
            log.info("belief extraction unavailable (%s); leaving backlog", exc)
            break
        character = update_beliefs(character, extracted, embedder)
        processed += 1
    if processed != reviewed:
        character = {**character, "beliefs_reviewed": processed}
    return character


# --- Read-outs -----------------------------------------------------------------
def higher_order(values):
    """Roll the ten values up onto Schwartz's four higher-order poles (weighted mean)."""
    out = {}
    for pole, members in VALUES_HIGHER_ORDER.items():
        weight_total = sum(members.values())
        out[pole] = sum(
            weight * values.get(v, 0.0) for v, weight in members.items()
        ) / weight_total
    return out


def read_values(character):
    """Human-readable values read-out: {long_name: value}, strongest first."""
    values = character.get("values") or {}
    ordered = sorted(VALUES, key=lambda v: -abs(values.get(v, 0.0)))
    return {VALUES_NAMES[v]: values.get(v, 0.0) for v in ordered}


def read_moral(character):
    """Human-readable moral outlook: {long_name: value}, strongest first."""
    moral = character.get("moral") or {}
    ordered = sorted(MORAL, key=lambda m: -abs(moral.get(m, 0.0)))
    return {MORAL_NAMES[m]: moral.get(m, 0.0) for m in ordered}


def read_beliefs(character):
    """The character's beliefs, strongest conviction first."""
    beliefs = character.get("beliefs") or []
    return sorted(beliefs, key=lambda b: -abs(b.get("confidence", 0.0)))


def dominant_value(character):
    """The value furthest from neutral -- what the character most prizes (or rejects)."""
    values = character.get("values") or {}
    if not values:
        return None
    key = max(VALUES, key=lambda v: abs(values.get(v, 0.0)))
    return {"key": key, "name": VALUES_NAMES[key], "value": values.get(key, 0.0)}
