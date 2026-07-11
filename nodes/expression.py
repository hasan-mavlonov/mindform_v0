"""SOCIAL EXPRESSION: the outward voice -- how the inner state comes OUT.

The output-side twin of the cognitive lens. Where ``cognition.interpret``/``lens`` bend how
the world gets *in*, this node shapes how the self gets *out*: a communication STYLE on the
interpersonal-circumplex axes -- assertion, warmth, energy, strain -- derived fresh each turn
(no stored state) from:

  * the SELF-VIEW, not the raw traits: we perform who we THINK we are (Goffman), so a
    character whose self-image lags its real traits still *talks* like the old self;
  * self-regard (``self.esteem``): confidence asserts plainly, doubt hedges;
  * the collective pressure of unmet needs (``drives``): starvation strains the voice.

Around the dims, three wordable signals: the ACTIVE NEED (the Motivation -> voice edge --
what they most lack shows in what they reach for), the DOMINANT VALUE, and THIS TURN'S MOOD
read from the *interpreted* appraisal -- so the same event is answered brightly by the secure
character and braced by the anxious one (the lens drives the mouth).

Two mouths read one derivation:
  * ``voice()``       -- a short imperative brief injected into the LLM reply prompt;
  * ``plain_reply()`` -- the deterministic offline reply: the same dims drive its SHAPE
                         (hedged openers, a reach toward the other, a length budget, a worry
                         tail), so two characters sound different with no LLM at all.

Deliberately DERIVED, not formed: every ingredient is already formed state one layer down,
so stored style would just be a cache that can fall out of sync. Style as its own formed
layer (entrenched or extinguished by how utterances actually land) needs the interlocutor's
NEXT message as the reception signal -- that is the intended Slice 2, alongside Behavior.
Pure arithmetic; no LLM, no numpy -- it degrades exactly like every other node.
"""

from core.config import (
    BASIS, STYLE_SOURCES, STYLE_THRESH, VOICE_MOOD_THRESH, VALUES_NAMES,
)
from nodes.drives import tensions as drive_tensions
from nodes.character import dominant_value
# Shared with the input-side lens so the lens and the voice never disagree about what is loud.
from nodes.cognition import _active_need, _DRIVE_NEED_PHRASE

STYLE_DIMS = ["assertion", "warmth", "energy", "strain"]
STYLE_LABELS = {
    "assertion": "assertion",   # says it plainly <-> hedges and qualifies
    "warmth": "warmth",         # takes care of the other <-> blunt
    "energy": "energy",         # quick, animated <-> quiet, spare
    "strain": "strain",         # tension shows through <-> even, composed
}


def _clamp(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, value))


def _self_image(personality):
    """The self they speak from: the self-image, falling back to the actual traits when no
    self exists yet (matching the ``_ensure_self`` backfill semantics)."""
    self_state = (personality or {}).get("self") or {}
    image = self_state.get("image")
    source = image if isinstance(image, dict) else (personality.get("traits") or {})
    return {k: float(source.get(k, 0.0)) for k in BASIS}


def _pressure(personality):
    """Collective need pressure in [0, 1]: how starved the character is across the needs."""
    t = drive_tensions((personality or {}).get("drives"))
    mean = sum(t.values()) / max(len(t), 1)
    return _clamp(2.0 * (mean - 0.5), 0.0, 1.0)


def style(personality):
    """The four style dims, signed [-1, 1], derived fresh from the layers (no stored state)."""
    img = _self_image(personality)
    esteem = float(((personality or {}).get("self") or {}).get("esteem", 0.0))
    press = _pressure(personality)
    out = {}
    for dim, src in STYLE_SOURCES.items():
        v = sum(w * img[k] for k, w in (src.get("image") or {}).items())
        v += src.get("esteem", 0.0) * esteem
        v += src.get("pressure", 0.0) * press
        out[dim] = _clamp(v)
    return out


def _mood_family(appraisal):
    """This turn's emotional lead, from the INTERPRETED appraisal (threat outranks brightness;
    threat_challenge is signed, -1 = threat)."""
    if not appraisal:
        return None
    if appraisal.get("threat_challenge", 0.0) < -VOICE_MOOD_THRESH:
        return "neg"
    v = appraisal.get("valence", 0.0)
    if v < -0.15:
        return "neg"
    if v > 0.15:
        return "pos"
    return "neutral"


# --- the LLM mouth: a short imperative brief for the reply prompt ----------------
def voice(personality, appraisal=None):
    """How they carry themselves when they speak, as imperatives for the reply prompt.
    Empty for a blank character (preserves the old prompt exactly)."""
    s = style(personality)
    t = STYLE_THRESH
    lines = []

    fam = _mood_family(appraisal)
    if appraisal and appraisal.get("threat_challenge", 0.0) < -VOICE_MOOD_THRESH:
        lines.append("This lands on you as a threat -- you're braced.")
    elif appraisal and appraisal.get("valence", 0.0) <= -VOICE_MOOD_THRESH:
        lines.append("It stings, but you soften it for their sake."
                     if s["warmth"] >= t else "This lands heavy on you right now.")
    elif appraisal and appraisal.get("valence", 0.0) >= VOICE_MOOD_THRESH:
        lines.append("This lands bright on you -- let it lift you.")

    need = _active_need(personality)
    if need:
        lines.append(f"Right now you most need {_DRIVE_NEED_PHRASE[need]} -- "
                     "let that pull show in what you reach for.")

    if s["warmth"] >= t:
        lines.append("Speak warmly; take care of the other person.")
    elif s["warmth"] <= -t:
        lines.append("Speak bluntly; don't cushion it.")
    if s["energy"] >= t:
        lines.append("Talk with energy, quick and animated.")
    elif s["energy"] <= -t:
        lines.append("Keep it quiet and spare; short sentences.")
    if s["assertion"] >= t:
        lines.append("Say it plainly, with quiet confidence.")
    elif s["assertion"] <= -t:
        lines.append("Hedge and qualify -- you doubt yourself right now.")
    if s["strain"] >= t:
        lines.append("You're wound tight; let the tension show at the edges.")

    dom = dominant_value(personality.get("character") or {})
    if dom and abs(dom["value"]) >= t:
        verb = "prize" if dom["value"] > 0 else "have come to reject"
        lines.append(f"You {verb} {dom['name']} -- it colors what you notice and bring up.")

    # the enacted stance (Behavior): the inclination the reply should carry
    inclination = _INCLINATION_BRIEF.get(_stance_mode(personality))
    if inclination:
        lines.append(inclination)

    return lines[:6]


# --- the enacted stance (Behavior node): the act the reply performs ----------------
def _stance_mode(personality):
    behavior = (personality or {}).get("behavior") or {}
    return (behavior.get("set") or {}).get("mode", "steady")


_INCLINATION_BRIEF = {
    "approach": "You're leaning in -- say what you want to do next.",
    "withdraw": "You're pulling back -- keep it guarded; don't promise reach you don't feel.",
    "conflicted": "Part of you wants to dive in and part wants to disappear -- let both show.",
}
_INCLINATION_TAIL = {
    "approach": "I want to go back at it.",
    "withdraw": "For now I'd rather keep my distance.",
    "conflicted": "Part of me wants to dive in; part of me wants to disappear.",
}


# --- the offline mouth: a deterministic compositor whose SHAPE is the style ------
# Reaction and aside tables relocated from web/reply.py -- the voice now owns them.
_REACTIONS = {
    "pos": [
        "That actually felt good -- better than I expected.",
        "Something in me opened up a little there.",
        "I didn't want that to end.",
    ],
    "neg": [
        "That one sat heavy with me.",
        "I'm still carrying it, if I'm honest.",
        "It shook me more than I'd like to admit.",
    ],
    "neutral": [
        "I keep turning that over.",
        "It left a quiet mark.",
        "I'm still working out how it sits with me.",
    ],
}
# Aside coloured by the strongest SELF-IMAGE dim -- the self they speak from.
_HIGH = {
    "N": "and a part of me is bracing for the next thing already.",
    "E": "and now I just want to be around people, doing more.",
    "O": "and I can't help wondering what else is underneath it.",
    "A": "and mostly I hope everyone else came through it okay.",
    "C": "so I'd rather make sense of it and handle it properly.",
}
_LOW = {
    "N": "but it doesn't rattle me much -- I feel steady.",
    "E": "though I'd rather sit with it on my own a while.",
    "O": "and I'll just take it as it is, no need to overthink it.",
    "A": "and I'll say what I think about it plainly.",
    "C": "and I'm easy about it -- no need to make a system of it.",
}
# What the active need reaches for, out loud.
_NEED_ASIDE = {
    "autonomy": "and I need room to do this my own way.",
    "competence": "and I need to get this right -- to know I can.",
    "relatedness": "and honestly, I just don't want to feel alone in it.",
}
_HEDGES = ["I don't know --", "Maybe it's just me, but"]

# Lexical sentiment: the fallback lead when no appraisal is supplied (legacy path).
_POS = {"party", "fun", "laughed", "danced", "loved", "win", "won", "great",
        "joy", "proud", "friends", "celebrated", "happy", "success", "amazing",
        "enjoyed", "confident", "excited", "warm", "kind", "together", "praised"}
_NEG = {"failed", "alone", "anxious", "scared", "afraid", "lost", "hurt", "cried",
        "panic", "sad", "lonely", "rejected", "sick", "angry", "ashamed",
        "terrified", "avoided", "isolated", "exam", "argued", "embarrassed"}


def _sentiment(text):
    tokens = set(text.lower().replace(".", " ").replace(",", " ").split())
    pos, neg = len(tokens & _POS), len(tokens & _NEG)
    if pos > neg:
        return "pos"
    if neg > pos:
        return "neg"
    return "neutral"


def _clip_to_first_sentence(text):
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1] == " "):
            return text[: i + 1]
    return text


def _decap(text):
    """Lowercase the first letter for joining after an opener -- except the pronoun 'I'."""
    if not text or text.startswith(("I ", "I'")):
        return text
    return text[0].lower() + text[1:]


def plain_reply(personality, user_text, appraisal=None):
    """The deterministic offline reply. The lead comes from how THIS character read the
    event (the interpreted appraisal -- the lens drives the mouth); the aside voices the
    active need or the strongest self-image dim; the style dims then shape the whole line:
    hedged or plain openers, a reach toward the other, a worry tail, a length budget."""
    family = _mood_family(appraisal) or _sentiment(user_text)
    reactions = _REACTIONS[family]
    lead = reactions[hash(user_text) % len(reactions)]   # varied per message

    # ASIDE: what they reach for (the active need) beats who they think they are.
    need = _active_need(personality)
    if need:
        aside = _NEED_ASIDE[need]
    else:
        img = _self_image(personality)
        key = max(BASIS, key=lambda k: abs(img[k]))
        aside = (_HIGH if img[key] >= 0 else _LOW)[key] if abs(img[key]) >= 0.12 else ""
    text = lead.rstrip(" .") + " -- " + aside if aside else lead

    s = style(personality)
    t = STYLE_THRESH
    if s["strain"] >= t and family == "neg":
        text += " I keep bracing for what comes next."
    if s["warmth"] >= t:
        text += " I'm glad you told me." if family == "pos" else " How are you holding up?"
    if s["assertion"] <= -t:
        text = _HEDGES[hash(user_text) % len(_HEDGES)] + " " + _decap(text)
    elif s["assertion"] >= t:
        text = "Honestly, " + _decap(text)
    if s["energy"] <= -t:                                # quiet and spare: the budget clips last
        text = _clip_to_first_sentence(text)
    # the enacted stance is the act itself -- always voiced, even by a quiet character
    # (appended after the clip on purpose)
    tail = _INCLINATION_TAIL.get(_stance_mode(personality))
    if tail:
        text += " " + tail
    return text


# --- Read-outs -------------------------------------------------------------------
_TAGS = [
    ("warmth", "warm", "blunt"),
    ("assertion", "plainspoken", "hedged"),
    ("energy", "animated", "quiet"),
    ("strain", "wound tight", None),        # composed is the absence, not a tag
]
_NEED_TAG = {
    "autonomy": "guarding their independence",
    "competence": "out to prove something",
    "relatedness": "reaching for connection",
}


def read_voice(personality, appraisal=None):
    """The voice as a short display phrase for the cockpit (empty when nothing stands out)."""
    s = style(personality)
    bits = []
    for dim, high, low in _TAGS:
        if s[dim] >= STYLE_THRESH and high:
            bits.append(high)
        elif s[dim] <= -STYLE_THRESH and low:
            bits.append(low)
    need = _active_need(personality)
    if need:
        bits.append(_NEED_TAG[need])
    return " · ".join(bits)


def read_expression(personality, appraisal=None):
    """Expression snapshot for the UI: the four style dims + the voice line."""
    s = style(personality)
    return {
        "style": [{"key": d, "label": STYLE_LABELS[d], "value": s[d]} for d in STYLE_DIMS],
        "line": read_voice(personality, appraisal),
    }
