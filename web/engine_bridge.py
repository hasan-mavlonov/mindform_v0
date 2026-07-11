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

from core.config import (
    BASIS, BASIS_NAMES, TRAIT_QUESTIONS, TRAIT_LEVELS, IDENTITY_FIELDS,
    APPRAISAL_DIMS, DEFAULT_TAU, VALUES, VALUES_NAMES, MORAL, MORAL_NAMES,
    RECALL_CANDIDATES,
)
from core.appraisal import appraise
from nodes.llm_appraisal import appraise_from_text
from core.appraisal_log import maybe_log as log_appraisal_row
from nodes.cognition import interpret, lens, read_lens
from nodes.llm_impact import push_from_text
from nodes.values import values_push_from_text
from nodes.moral import moral_push_from_text
from core.updater import update_personality
from nodes.temperament import genesis, build_character
from nodes.character import (
    default_character, update_values, update_moral, note_habit, form_beliefs,
    higher_order, dominant_value,
)
from nodes.drives import (
    refresh as refresh_drives, apply_event as apply_drive_event,
    tensions as drive_tensions, read_drives, dominant_drive, recall_bias,
)
from nodes.self_concept import (
    refresh as refresh_self, apply_event as apply_self_event, read_self, self_signal,
)
from nodes.expression import (
    read_expression, style as expression_style,
    refresh as refresh_expression, apply_event as apply_expression_event,
)
from nodes.behavior import (
    refresh as refresh_behavior, apply_event as apply_behavior_event,
    note_act, intake as behavior_intake, read_behavior,
)
from core.impact import clamp
from core.personality import (
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


def _behavior_block(personality, before, appraisal, appraisal_raw):
    """The behavior snapshot, with the intake reported as the REALIZED factor -- what this
    experience's intensity was actually multiplied by (the clamp at the [0,1] ceiling can
    shrink the requested gate, and the display must say what landed, not what was asked)."""
    block = read_behavior(personality, before)
    raw_intensity = (appraisal_raw or {}).get("intensity", 0.0)
    if appraisal and raw_intensity > 0:
        block["intake"] = appraisal.get("intensity", 0.0) / raw_intensity
    return block


def _recalled_rows(recalled):
    """The past memories this turn surfaced, for the UI: text, the honest cosine score, and --
    when an active need pulled the memory up -- which need (motivated retrieval, named)."""
    return [{"text": m.get("text", ""), "score": float(m.get("score", 0.0)),
             "need": m.get("need"), "need_pull": float(m.get("need_pull", 0.0))}
            for m in (recalled or [])]


def _drive_rows(personality, before_tension=None):
    """The three SDT needs: name, resting weight (the ghost tick), current tension, and this
    turn's satisfy/frustrate delta -- ``sat`` > 0 = the event fed the need, < 0 = frustrated it."""
    before = before_tension or {}
    rows = read_drives(personality)
    for row in rows:
        prev = before.get(row["key"], row["tension"])
        row["sat"] = prev - row["tension"]     # tension fell -> satisfied; rose -> frustrated
    return rows


def _self_row(personality, before=None, signal=None):
    """The self-concept for the UI: per-OCEAN self-image vs the actual trait (the gap is
    self-deception), esteem + its dispositional baseline, this turn's esteem delta, and the
    self-consistency ``align`` (> 0 the experience affirmed the self-view, < 0 contradicted it)."""
    row = read_self(personality)
    before = before or {}
    row["esteem_delta"] = row["esteem"] - float(before.get("esteem", row["esteem"]))
    row["align"] = float((signal or {}).get("align", 0.0))
    return row


def snapshot(personality, *, push=None, appraisal=None, appraisal_raw=None, source=None,
             reasoning="", seen=None, formation=None, reply=None,
             values_push=None, values_source=None, values_reasoning="",
             moral_push=None, recalled=None, drive_before=None, self_before=None, self_sig=None,
             reply_source=None, behavior_before=None, expression_before=None,
             appraisal_source=None):
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
        "appraisal_raw": appraisal_raw,
        "appraisal_source": appraisal_source,
        "source": source,
        "reasoning": reasoning or "",
        "seen": seen,
        "recalled": _recalled_rows(recalled),
        "formation": formation,
        "dominant": _dominant(trait_rows),
        "lens": read_lens(personality, recalled=recalled),
        "character": _character_block(
            personality, push=values_push, source=values_source,
            reasoning=values_reasoning, moral_push=moral_push,
        ),
        "drives": _drive_rows(personality, drive_before),
        "drive": dominant_drive(personality),
        "self": _self_row(personality, self_before, self_sig),
        "expression": {**read_expression(personality, appraisal, expression_before),
                       "source": reply_source},
        "behavior": _behavior_block(personality, behavior_before, appraisal, appraisal_raw),
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
def _recall(text, name, personality=None):
    """Encode the experience and look back, BEFORE it is stored. Returns
    ``(embedding, seen, recalled)``.

    Guarded: if sentence-transformers / numpy are absent, formation still works -- we just
    can't count recurrences or recall similar episodes (returns ``(None, None, [])``).
    Recall runs before the current experience is stored, so a message never just recalls
    itself -- and crucially before ``interpret``, so memory can colour the reading.

    MOTIVATED RETRIEVAL: recall fetches a wider candidate pool, then the character's
    active needs re-rank it (``drives.recall_bias``) -- what they lack shapes what comes
    to mind. Pass ``personality`` with its drives already refreshed.
    """
    try:
        from core.encoder import encode_text          # heavy: sentence-transformers
        from core.memory import recurrence, recall     # heavy: numpy
    except Exception as exc:                       # deps absent -> skip cleanly
        log.info("memory/encoder unavailable (%s); skipping recurrence/recall", exc)
        return None, None, []
    try:
        embedding = encode_text(text)
        candidates = recall(embedding, name=name, k=RECALL_CANDIDATES)
        recalled = recall_bias(candidates, (personality or {}).get("drives"))
        return embedding, recurrence(embedding, name=name), recalled
    except Exception as exc:
        log.warning("recall step failed (%s); continuing", exc)
        return None, None, []


def _store_memory(text, embedding, appraisal, push, personality, name):
    """Persist this experience (log record + embedding sidecar) once it has formed.

    No-op when the encoder/memory deps are absent (``embedding is None``). Stores the
    *interpreted* appraisal -- what the character actually perceived -- so future recalls
    rebuild the schema from how things felt to them, not from an abstract reading.
    """
    if embedding is None:
        return
    try:
        from core.memory import create_memory          # heavy: numpy
        create_memory(text, embedding, appraisal, push, personality, name=name)
    except Exception as exc:
        log.warning("memory store failed (%s); continuing", exc)


def _form_beliefs(personality, char_name):
    """Reflection pass: turn unreviewed memories into beliefs (guarded; no-op offline).

    Reads the per-character memory log and the encoder for embedding dedup; if those deps
    are absent there is no backlog to read, so the character is returned unchanged.
    ``form_beliefs`` itself is a no-op when the LLM is unavailable, leaving the watermark.
    """
    character = personality.get("character") or default_character()
    try:
        from core.memory import load_memories
        from core.encoder import encode_text
        memories, embedder = load_memories(char_name), encode_text
    except Exception:
        memories, embedder = [], None
    return form_beliefs(character, memories, embedder)


def run_turn(name, message):
    """Feed one experience to a character and return the new state + a reply.

    This is exactly ``interactive.py``'s per-line pipeline:
        appraise -> push_from_text -> update_personality -> (memory) -> save.
    The conversational ``reply``'s TEXT never re-enters the personality math; what
    does feed forward (BEHAVIOR) is the deterministic stance recorded before the
    reply, resolved against the NEXT user message as its outcome.
    """
    text = (message or "").strip()
    personality = load_character(name)
    char_name = (personality.get("identity") or {}).get("name")

    # MOTIVATION: recompute each need's resting weight from the current values and let unmet
    # needs rebuild pressure (regeneration) FIRST -- the refreshed needs steer both what is
    # remembered (motivated retrieval, below) and how the experience is read (interpret).
    personality = refresh_drives(personality)
    drive_before = drive_tensions(personality.get("drives"))   # rest level, to measure the event

    # Look back BEFORE interpreting, so memory can colour how this experience is read --
    # with the active needs re-ranking what comes to mind (drives.recall_bias).
    embedding, seen, recalled = _recall(text, char_name, personality)

    # SELF-CONCEPT: relax self-esteem toward its dispositional baseline BEFORE interpreting, so the
    # self they carry (self-image + regard) colours the read.
    personality = refresh_self(personality)
    self_before = dict(personality.get("self") or {})          # esteem/image before the event

    # BEHAVIOR: relax the sensitivities toward their trait set-points; the CARRIED stance
    # (last turn's act) is what gates this experience's intake inside interpret.
    personality = refresh_behavior(personality)
    behavior_before = dict(personality.get("behavior") or {})
    engagement = behavior_intake(personality)                  # the intake factor, for the mirror

    # EXPRESSION: the formed manner relaxes toward what the inner state now calls for
    # (manner lags inner change); the reply later speaks with this formed style.
    personality = refresh_expression(personality)
    expression_before = dict(personality.get("expression") or {})

    # PERCEPTION: the base reading -- LLM-primary (offline head/lexicon fallback). Every
    # LLM-labelled reading is also logged as (text -> appraisal) training data, so the
    # offline appraiser learns from the online one with use (distillation).
    raw_appraisal, appraisal_source = appraise_from_text(text)
    log_appraisal_row(text, raw_appraisal, appraisal_source)
    appraisal = interpret(raw_appraisal, personality, recalled=recalled)   # bent by the lens
    self_sig = self_signal(personality.get("self"), appraisal)  # did it affirm / contradict the self-view
    view = lens(personality, recalled=recalled)
    push, source, reasoning = push_from_text(text, appraisal, lens=view)
    values_push, values_source, values_reasoning = values_push_from_text(text, appraisal, lens=view)
    moral_push, moral_source, _ = moral_push_from_text(text, appraisal, lens=view)

    # BEHAVIOR (the intake gate, LLM mirror): the heuristic pushes already inherit the gate
    # through salience (they read the gated intensity); the LLM pushes read raw text, so the
    # same factor is applied explicitly -- both paths land quantitatively identically gated.
    # Mirror the REALIZED ratio (interpreted/raw intensity -- exact, intensity is behavior's
    # private channel), not the requested factor: at the intensity ceiling the clamp shrinks
    # what the heuristic path actually received, and the LLM path must shrink with it.
    raw_intensity = raw_appraisal.get("intensity", 0.0)
    realized = (appraisal.get("intensity", 0.0) / raw_intensity) if raw_intensity > 0 \
        else engagement
    if realized != 1.0:
        if source != "heuristic":
            push = {k: clamp(v * realized) for k, v in push.items()}
        if values_source != "heuristic":
            values_push = {k: clamp(v * realized) for k, v in values_push.items()}
        if moral_source != "heuristic":
            moral_push = {k: clamp(v * realized) for k, v in moral_push.items()}

    before_traits = dict(personality["traits"])
    personality = update_personality(personality, push)

    # Now that the experience has formed, commit it to memory (with the interpreted reading).
    _store_memory(text, embedding, appraisal, push, personality, char_name)

    # CHARACTER: the same experience forms the values and the moral outlook, and a
    # recurring one (seen before, this occurrence included) settles into a habit.
    character = update_values(personality.get("character") or default_character(), values_push)
    character = update_moral(character, moral_push)
    character = note_habit(character, text, (seen or 0) + 1)
    personality = {**personality, "character": character}

    # BELIEF: the same experience (now in memory) -- and any offline backlog -- forms
    # beliefs; a no-op without the LLM, which the next online turn catches up.
    personality = {**personality, "character": _form_beliefs(personality, char_name)}

    # MOTIVATION: the interpreted experience now satisfies or frustrates the active needs
    # (tension falls when a need is met, rises when it is denied) -- the loop-closing edge.
    personality = {**personality,
                   "drives": apply_drive_event(personality.get("drives"), appraisal)}

    # SELF-CONCEPT: self-regard responds to the interpreted experience (sociometer) and the
    # self-image drifts toward the just-formed traits (self-perception, resisting disconfirmation).
    personality = apply_self_event(personality, appraisal, personality["traits"])

    # BEHAVIOR: the world's answer trains the sensitivities (rewarded own action teaches
    # approach; threat trains inhibition; a carried lean-in is credited with how this
    # message received it), and the new action readiness is read and carried forward.
    personality = apply_behavior_event(personality, appraisal)

    # EXPRESSION (operant shaping): the manner actually SPOKEN WITH last turn (frozen in the
    # act record) is entrenched or extinguished by how this message received it -- read
    # BEFORE note_act overwrites the record with this turn's act.
    last_act = (personality.get("behavior") or {}).get("last") or {}
    personality = apply_expression_event(
        personality, last_act.get("style"),
        (personality.get("behavior") or {}).get("reception"))

    # Freeze the act this reply performs -- stance + mode + the (newly shaped) style it is
    # spoken with -- as the record next turn's outcome is credited against.
    personality = note_act(personality, style=expression_style(personality))

    save_character(personality)

    formation = _formation(before_traits, personality)
    # SOCIAL EXPRESSION: the reply speaks from the self-view in the character's current
    # voice, and sees this turn's INTERPRETED appraisal -- the lens drives the mouth.
    reply, reply_source = generate_reply(personality, text, memories=recalled,
                                         appraisal=appraisal)

    return snapshot(
        personality, push=push, appraisal=appraisal, appraisal_raw=raw_appraisal,
        source=source, reasoning=reasoning, seen=seen, formation=formation, reply=reply,
        values_push=values_push, values_source=values_source,
        values_reasoning=values_reasoning, moral_push=moral_push, recalled=recalled,
        drive_before=drive_before, self_before=self_before, self_sig=self_sig,
        reply_source=reply_source, behavior_before=behavior_before,
        expression_before=expression_before, appraisal_source=appraisal_source,
    )
