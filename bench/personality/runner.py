"""Administer a personality instrument to an already-frozen MindForm character.

Never forms, never re-forms, never mutates. Restores the requested frozen
snapshot, asks the persisted character to rate each item in the instrument
one at a time (a pure LLM read: no ``run_turn``, no ``save_character``
anywhere in this file), scores the responses with the instrument's own
official key, and returns a fully audited record -- every question, every
raw model response, every parsed rating, whether that item was reverse-keyed,
its scored value, and its facet/domain assignment.

Shared with bench/heart only at the data layer (heartdata, mfadapter,
snapshots, config.require_heart_bench) -- see bench/personality/__init__.py
for why that specific boundary is drawn where it is.
"""

import hashlib
import os
import time

from bench.heart import heartdata, mfadapter, snapshots
from bench.heart.config import require_heart_bench
from bench.heart.logbook import Logbook, new_run_id
from bench.personality import config, persona
from bench.personality.instrument import Instrument

SYSTEM_PROMPT = """You are answering a personality questionnaire IN CHARACTER, as the \
person described below -- based on who they have become through the experiences that \
formed them. This is self-report: answer as that person would honestly describe \
themselves, not as an assistant.

Rate the single statement using this scale:
{scale}

Respond with ONLY a JSON object: {{"rating": <integer, one of the scale numbers above>}}"""


class NoSnapshot(Exception):
    pass


class AdministrationError(Exception):
    """A model call never produced a usable rating after every retry."""


def _snapshot_digest(run_dir, label):
    d = os.path.join(run_dir, "snapshots", label)
    h = hashlib.sha256()
    for fname in sorted(os.listdir(d)):
        h.update(fname.encode())
        with open(os.path.join(d, fname), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _render_persona(char_ctx, state):
    """A short, plain-language persona summary -- occupation plus the engine's
    own formed state. No HEART label, ever (see persona.leak_check)."""
    t = state["traits"]
    lines = [f"Occupation: {char_ctx.get('occupation', 'N/A')}", "",
             "Formed disposition (each -1..+1, 0 = unremarkable):",
             "  O {O:+.2f}  C {C:+.2f}  E {E:+.2f}  A {A:+.2f}  N {N:+.2f}".format(**t)]
    if state.get("top_values"):
        lines.append("Values: " + ", ".join(f"{n} ({v:+.2f})" for n, v in state["top_values"]))
    if state.get("top_drives"):
        lines.append("Currently unmet needs: " +
                     ", ".join(f"{n} ({v:.2f})" for n, v in state["top_drives"]))
    if state.get("beliefs"):
        lines.append("Beliefs: " + "; ".join(state["beliefs"][:3]))
    return "\n".join(lines)


def _scale_text(instrument):
    return "\n".join(f"{n} = {label}" for n, label in sorted(instrument.scale_labels.items()))


def _ask_item(instrument, persona_text, item, log, retries=2):
    from core.llm import complete_json

    system = SYSTEM_PROMPT.format(scale=_scale_text(instrument))
    user = (f"{persona_text}\n\n{instrument.stem} {item.text}\n\n"
           f"How much do you agree, as this person?")
    last_raw, last_err = None, None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            out = complete_json(system, user, temperature=config.TEMPERATURE,
                                max_tokens=config.MAX_TOKENS)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            log.event("bfi2_item_call_failed", item=item.number, attempt=attempt,
                      error=last_err)
            continue
        latency_ms = (time.time() - t0) * 1000.0
        rating = out.get("rating") if isinstance(out, dict) else None
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            rating = None
        if rating is not None and instrument.scale_min <= rating <= instrument.scale_max:
            return rating, out, latency_ms
        last_raw, last_err = out, f"unusable rating: {rating!r}"
        log.event("bfi2_item_unparseable", item=item.number, attempt=attempt,
                  raw=out, error=last_err)
    raise AdministrationError(
        f"item {item.number} ({item.text!r}): no usable rating after "
        f"{retries + 1} attempts -- last error: {last_err}")


def administer(character_id, instrument, tier=None, on_item=None):
    """Run the whole instrument against a character's frozen snapshot.

    ``tier``: "full" or "dev" to require that specific tier, None to prefer
    full and fall back to dev (mirrors bench.heart.snapshots' own convention).
    ``on_item(i, total, item, rating)`` is an optional progress callback.

    Returns a dict with the full audit trail and the instrument's scoring
    output. Raises ``NoSnapshot`` if nothing is prepared, and re-raises
    ``RuntimeError`` if the frozen snapshot changes during the run (it must
    not -- this function only ever calls ``mfadapter.restore``, never
    ``create_neutral``, ``ingest_all``, ``ingest_memory`` or ``freeze``).
    """
    require_heart_bench()
    chars = heartdata.load_characters()
    if character_id not in chars:
        raise NoSnapshot(f"{character_id} is not a HEART character")
    char = chars[character_id]

    if tier is not None:
        snap = snapshots.best_for_tier(character_id, tier)
        if not snap:
            raise NoSnapshot(f"{character_id} has no {tier} snapshot prepared.")
    else:
        snap = snapshots.best_for_tier(character_id, "full") or snapshots.best_for(character_id)
        if not snap:
            raise NoSnapshot(f"{character_id} has no frozen snapshot at all.")

    name, run_dir, label = snap["bench_name"], snap["run_dir"], snap["label"]
    digest_before = _snapshot_digest(run_dir, label)

    restored_id = mfadapter.restore(name, run_dir, label)   # read-only: copies FROM the
                                                            # frozen dir, never TO it
    char_ctx = persona.character_context(char)
    state = persona.state_context(name)
    persona_text = _render_persona(char_ctx, state)

    run_id = new_run_id(f"bfi2-{character_id.lower()}")
    log = Logbook(run_id, config.RESULTS_ROOT)
    log.event("bfi2_run_start", character=character_id, instrument=instrument.id,
             tier=("full" if snap["full_protocol"] else "dev"),
             snapshot_id=restored_id, model=config.MODEL, temperature=config.TEMPERATURE)

    # Leak check on the persona text BEFORE a single item is sent -- the same
    # text is reused verbatim in every item's prompt, so one check covers all.
    ok, detail = persona.leak_check(persona_text, char)
    if not ok:
        log.event("bfi2_leak_detected", detail=detail)
        raise RuntimeError(f"LEAK in personality-test persona for {character_id}: {detail}")

    rows = []
    responses = {}
    total = len(instrument.items)
    for i, item in enumerate(instrument.items):
        prompt_preview = f"{instrument.stem} {item.text}"
        # Leak-check EVERY item's full prompt, not just the shared persona text --
        # cheap, and it means a future instrument that interpolates per-item
        # character data is covered automatically.
        full_prompt = f"{persona_text}\n\n{prompt_preview}"
        ok, detail = persona.leak_check(full_prompt, char)
        if not ok:
            log.event("bfi2_leak_detected", item=item.number, detail=detail)
            raise RuntimeError(f"LEAK in item {item.number} prompt for {character_id}: {detail}")

        rating, raw, latency_ms = _ask_item(instrument, persona_text, item, log)
        responses[item.number] = rating
        row = {
            "item_number": item.number,
            "question": prompt_preview,
            "domain": item.domain,
            "facet": item.facet,
            "reverse_keyed": item.reverse_keyed,
            "raw_response": raw,
            "parsed_rating": rating,
            "latency_ms": round(latency_ms, 1),
        }
        rows.append(row)
        log.event("bfi2_item_answered", **row)
        if on_item:
            on_item(i + 1, total, item, rating)

    scored = instrument.score(responses)
    for row in rows:
        row["scored_value"] = scored["keyed_responses"][row["item_number"]]

    digest_after = _snapshot_digest(run_dir, label)
    if digest_after != digest_before:
        log.event("bfi2_snapshot_mutated", before=digest_before, after=digest_after)
        raise RuntimeError(
            f"frozen snapshot at {run_dir}/snapshots/{label} changed during a "
            f"personality-test run ({digest_before[:12]}… -> {digest_after[:12]}…) "
            f"-- refusing the result")

    log.event("bfi2_run_end", **{k: v for k, v in scored.items() if k != "keyed_responses"})

    return {
        "run_id": run_id,
        "character": character_id,
        "occupation": char_ctx.get("occupation"),
        "instrument": {"id": instrument.id, "name": instrument.name,
                       "full_name": instrument.full_name,
                       "license_notice": instrument.license_notice,
                       "citation": instrument.citation,
                       "facets_are_exploratory": instrument.facets_are_exploratory,
                       "reliability_note": instrument.reliability_note},
        "tier": "full" if snap["full_protocol"] else "dev",
        "snapshot": {"memories": snap["memories"], "snapshot_id": restored_id,
                    "run_id": snap["run_id"]},
        "items": rows,
        "scored": scored,
        "mindform_internal_traits": state["traits"],   # for the comparison view;
                                                        # NOT the questionnaire result
    }
