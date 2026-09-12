"""Loading HEART-Bench artefacts, with leakage control at the boundary.

Two rules enforced here rather than at the prompt, so a leak cannot slip
through a code path that forgot to sanitise:

  * ``character_public()`` returns ONLY the fields HEART's own runners expose
    (id + occupation). The withheld big_five / description / self_value_logic /
    core_patterns / name never leave this module toward a prompt.
  * ``anonymize_mem_id()`` is a straight port of HEART's function. Every one of
    a character's memory ids carries the same trait tag (CHAR_01 -> N_HIGH x1000),
    so an unstripped id names the answer profile outright.

The ground truth loader is deliberately separate and is never imported by the
answering path -- see ``runner`` for the reveal-after-commit ordering.
"""

import hashlib
import json
import os
import re

from bench.heart.config import (
    HEART_PATH, ALLOWED_CHARACTER_FIELDS, FORBIDDEN_CHARACTER_FIELDS,
)

_TRAIT_ID_RE = re.compile(r"^(MEM_CHAR_\d+)_(?:[NCEAO]_(?:HIGH|LOW)|NEUTRAL)_(\d+)$")

# Chronological ordering. HEART stores life stage as free text in `timeline`
# ("Childhood (age 6)", "Adolescence (age 15)"). File order is a single solid
# block of one trait tag, which is not a life; we sort by stage then by the age
# mentioned, falling back to the original index for ties (a stable sort).
_STAGE_ORDER = [
    ("childhood", 0), ("kindergarten", 0), ("school", 1), ("primary", 1),
    ("elementary", 1), ("adolescen", 2), ("teen", 2), ("youth", 3),
    ("early adult", 3), ("college", 3), ("university", 3), ("adult", 4),
    ("work", 4), ("career", 4), ("middle", 5), ("later", 6), ("old", 6),
]


def _path(*parts):
    return os.path.join(HEART_PATH, *parts)


def _load(name):
    with open(_path("benchmark", name), encoding="utf-8") as fh:
        return json.load(fh)


def anonymize_mem_id(mem_id):
    """Port of HEART's anonymize_mem_id: MEM_CHAR_01_N_HIGH_0049 -> MEM_CHAR_01_0049."""
    if not isinstance(mem_id, str):
        return mem_id
    m = _TRAIT_ID_RE.match(mem_id)
    return f"{m.group(1)}_{m.group(2)}" if m else mem_id


def _stage_rank(timeline):
    t = (timeline or "").lower()
    for key, rank in _STAGE_ORDER:
        if key in t:
            return rank
    return 99


def _age_in(timeline):
    m = re.search(r"(\d{1,2})", timeline or "")
    return int(m.group(1)) if m else 999


def load_characters():
    return {c["id"]: c for c in _load("characters.json")["characters"]}


def load_scenarios():
    raw = _load("scenarios.json")["scenarios"]
    out = {}
    for stage_list in raw.values():
        for s in stage_list:
            out[s["id"]] = s
    return out


def load_questions():
    return _load("mcq.json")["questions"]


def character_public(char):
    """The only character fields allowed anywhere near a prompt."""
    return {k: char.get(k, "N/A") for k in ALLOWED_CHARACTER_FIELDS}


def withheld_big_five(char):
    """HEART's hidden label. ONLY for post-hoc analysis, never for a prompt.

    Callers must not pass the result into any prompt-building function; the
    leak checker scans final prompts for these values regardless.
    """
    return dict(char.get("big_five") or {})


def forbidden_strings(char, questions):
    """Every string that would constitute a leak if it appeared in a prompt."""
    bad = []
    for field in FORBIDDEN_CHARACTER_FIELDS:
        v = char.get(field)
        if isinstance(v, str) and v.strip():
            bad.append(v.strip())
    # the parenthetical in `name` encodes trait + direction, e.g.
    # "Role A (high-neuroticism free creator)"
    name = char.get("name") or ""
    m = re.search(r"\((.*?)\)", name)
    if m:
        bad.append(m.group(1).strip())
    # Other characters' ids: source_character is the answer key. The character's
    # OWN id is exempt -- HEART's build_basic_info exposes it deliberately, so
    # flagging it would be a false positive on every legitimate prompt.
    own_id = char.get("id")
    for q in questions:
        for o in q.get("options", []):
            sc = o.get("source_character")
            if sc and sc != own_id:
                bad.append(sc)
    # raw trait-bearing memory ids
    for mem in char.get("episodic_memory_set", []):
        mid = mem.get("id", "")
        if _TRAIT_ID_RE.match(mid):
            bad.append(mid)
    return sorted({b for b in bad if b and len(b) > 3})


def ingestible_memories(char):
    """Raw memories, chronologically ordered, stripped to what MindForm may see.

    Returns dicts with an anonymised id, the timeline label, the raw
    ``content_full`` text, and a sha256 of that text. Nothing else -- in
    particular no trait tag and no summary-derived label.
    """
    mems = char.get("episodic_memory_set", [])
    indexed = list(enumerate(mems))
    indexed.sort(key=lambda p: (_stage_rank(p[1].get("timeline")),
                                _age_in(p[1].get("timeline")),
                                p[0]))
    out = []
    for chrono_pos, (orig_idx, m) in enumerate(indexed):
        text = m.get("content_full") or m.get("content_summary") or ""
        out.append({
            "anon_id": anonymize_mem_id(m.get("id", "?")),
            "timeline": m.get("timeline", "?"),
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chrono_position": chrono_pos,
            "original_index": orig_idx,
        })
    return out


def questions_for(character_id, limit=None):
    qs = [q for q in load_questions() if q["character_id"] == character_id]
    qs.sort(key=lambda q: q["question_id"])
    return qs[:limit] if limit else qs


def public_options(question):
    """Options with the answer key stripped: label + content only.

    Matches HEART's build_options_text, which also only reads label/content.
    """
    return [{"label": o["label"], "content": o["content"]} for o in question["options"]]


def ground_truth(question):
    """The answer key. Callers MUST only touch this after an answer is committed."""
    return question["correct_answer"]
