"""CHARACTER: the values, moral outlook, and habits a person accumulates from experience.

Where ``temperament`` is the innate OCEAN baseline a character is *born* with, CHARACTER
is what they *become* by living -- modeled as:

  * values  -- the ten Schwartz basic values (``config.VALUES``),
  * moral   -- the six Moral Foundations / moral outlook (``config.MORAL``),
  * habits  -- the recurring experiences they fall into.

Values and moral foundations both form from experience with the SAME diminishing-returns
dynamics as the traits (``updater.apply_diminishing``); a recurring experience (counted by
``memory.recurrence``) that crosses ``config.HABIT_MIN_RECURRENCE`` becomes a named habit.

State lives in ``personality["character"] = {"values": {...}, "moral": {...},
"habits": [...]}`` and persists with the rest of the character. Values and foundations
start at 0 -- earned, not innate -- which is exactly what separates CHARACTER (experience)
from TEMPERAMENT (biology).
"""

from config import (
    VALUES, VALUES_NAMES, VALUES_HIGHER_ORDER, MORAL, MORAL_NAMES,
    HABIT_MIN_RECURRENCE,
)
from updater import apply_diminishing


def default_character():
    """A blank character layer: every value and foundation neutral (0), no habits yet."""
    return {
        "values": {v: 0.0 for v in VALUES},
        "moral": {m: 0.0 for m in MORAL},
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


def dominant_value(character):
    """The value furthest from neutral -- what the character most prizes (or rejects)."""
    values = character.get("values") or {}
    if not values:
        return None
    key = max(VALUES, key=lambda v: abs(values.get(v, 0.0)))
    return {"key": key, "name": VALUES_NAMES[key], "value": values.get(key, 0.0)}
