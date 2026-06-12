"""CHARACTER: the values and habits a person accumulates from experience.

Where ``temperament`` is the innate OCEAN baseline a character is *born* with,
CHARACTER is what they *become* by living -- modeled as the ten Schwartz basic values
(``config.VALUES``) plus the habits they fall into. Both form from experience:

  * values -- moved every turn by ``values.values_push_from_text`` with the same
    diminishing-returns dynamics as the traits (``updater.apply_diminishing``);
  * habits -- a recurring experience (counted by ``memory.recurrence``) that crosses
    ``config.HABIT_MIN_RECURRENCE`` becomes a named habit.

State lives in ``personality["character"] = {"values": {...}, "habits": [...]}`` and
persists with the rest of the character. Values start at 0 -- earned, not innate --
which is exactly what separates CHARACTER (experience) from TEMPERAMENT (biology).
"""

from config import VALUES, VALUES_NAMES, VALUES_HIGHER_ORDER, HABIT_MIN_RECURRENCE
from updater import apply_diminishing


def default_character():
    """A blank character layer: every value neutral (0), no habits yet."""
    return {"values": {v: 0.0 for v in VALUES}, "habits": []}


def update_values(character, push):
    """Return a new character with each value moved by its push (input unchanged)."""
    values = apply_diminishing(character.get("values") or {}, push, VALUES)
    return {**character, "values": values}


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


def dominant_value(character):
    """The value furthest from neutral -- what the character most prizes (or rejects)."""
    values = character.get("values") or {}
    if not values:
        return None
    key = max(VALUES, key=lambda v: abs(values.get(v, 0.0)))
    return {"key": key, "name": VALUES_NAMES[key], "value": values.get(key, 0.0)}
