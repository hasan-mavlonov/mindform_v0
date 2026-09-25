"""The exact 30 BFI-2-S items, facet/domain assignment, and reverse-keying.

Transcribed verbatim from Appendix A of Soto & John (2017), Journal of
Research in Personality, 68, 69-81 -- the officially reprinted item list
(Appendix A.1) and scoring key (Appendix A.2). Every item's text, its facet,
its domain, and its reverse-keying below can be checked against that appendix
directly; none of it is reconstructed or paraphrased from memory, per the
project's own "do not invent mappings" rule.

    "BFI-2 items copyright 2015 by Oliver P. John and Christopher J. Soto.
     Reprinted with permission." -- exact notice printed on every table in
    the source paper.

Domain correspondence to MindForm's O/C/E/A/N is explicit and disclosed, never
assumed: BFI-2 uses updated domain names (Negative Emotionality instead of
Neuroticism, Open-Mindedness instead of Openness) for the SAME five
constructs, scored in the SAME direction (a person who worries, feels
depressed, and is emotionally volatile scores HIGH on Negative Emotionality --
this is Neuroticism under a newer name, not its inverse).
"""

from bench.personality.instrument import Instrument, Item

STEM = "I am someone who..."

SCALE_LABELS = {
    1: "Disagree strongly",
    2: "Disagree a little",
    3: "Neutral; no opinion",
    4: "Agree a little",
    5: "Agree strongly",
}

# (number, text, domain, facet, reverse_keyed) -- one row per official item,
# in the paper's own numbering (1-30). Cross-checked against Appendix A.2's
# scoring key ("R" suffix = reverse-keyed) for every single item.
_ROWS = [
    (1,  "Tends to be quiet.",                          "Extraversion",         "Sociability",         True),
    (2,  "Is compassionate, has a soft heart.",         "Agreeableness",        "Compassion",          False),
    (3,  "Tends to be disorganized.",                   "Conscientiousness",    "Organization",        True),
    (4,  "Worries a lot.",                               "Negative Emotionality","Anxiety",             False),
    (5,  "Is fascinated by art, music, or literature.",  "Open-Mindedness",     "Aesthetic Sensitivity",False),
    (6,  "Is dominant, acts as a leader.",               "Extraversion",         "Assertiveness",       False),
    (7,  "Is sometimes rude to others.",                 "Agreeableness",        "Respectfulness",      True),
    (8,  "Has difficulty getting started on tasks.",     "Conscientiousness",    "Productiveness",      True),
    (9,  "Tends to feel depressed, blue.",               "Negative Emotionality","Depression",          False),
    (10, "Has little interest in abstract ideas.",       "Open-Mindedness",     "Intellectual Curiosity",True),
    (11, "Is full of energy.",                           "Extraversion",         "Energy Level",        False),
    (12, "Assumes the best about people.",               "Agreeableness",        "Trust",               False),
    (13, "Is reliable, can always be counted on.",       "Conscientiousness",    "Responsibility",      False),
    (14, "Is emotionally stable, not easily upset.",     "Negative Emotionality","Emotional Volatility",True),
    (15, "Is original, comes up with new ideas.",        "Open-Mindedness",     "Creative Imagination", False),
    (16, "Is outgoing, sociable.",                       "Extraversion",         "Sociability",         False),
    (17, "Can be cold and uncaring.",                    "Agreeableness",        "Compassion",          True),
    (18, "Keeps things neat and tidy.",                  "Conscientiousness",    "Organization",        False),
    (19, "Is relaxed, handles stress well.",             "Negative Emotionality","Anxiety",             True),
    (20, "Has few artistic interests.",                  "Open-Mindedness",     "Aesthetic Sensitivity",True),
    (21, "Prefers to have others take charge.",          "Extraversion",         "Assertiveness",       True),
    (22, "Is respectful, treats others with respect.",   "Agreeableness",        "Respectfulness",      False),
    (23, "Is persistent, works until the task is finished.","Conscientiousness","Productiveness",       False),
    (24, "Feels secure, comfortable with self.",         "Negative Emotionality","Depression",          True),
    (25, "Is complex, a deep thinker.",                  "Open-Mindedness",     "Intellectual Curiosity",False),
    (26, "Is less active than other people.",            "Extraversion",         "Energy Level",        True),
    (27, "Tends to find fault with others.",             "Agreeableness",        "Trust",               True),
    (28, "Can be somewhat careless.",                    "Conscientiousness",    "Responsibility",      True),
    (29, "Is temperamental, gets emotional easily.",     "Negative Emotionality","Emotional Volatility",False),
    (30, "Has little creativity.",                       "Open-Mindedness",     "Creative Imagination", True),
]

DOMAIN_ORDER = ("Extraversion", "Agreeableness", "Conscientiousness",
                "Negative Emotionality", "Open-Mindedness")

# Official facet order per domain, as listed in the paper's own scoring key.
FACET_ORDER = {
    "Extraversion":          ("Sociability", "Assertiveness", "Energy Level"),
    "Agreeableness":         ("Compassion", "Respectfulness", "Trust"),
    "Conscientiousness":     ("Organization", "Productiveness", "Responsibility"),
    "Negative Emotionality": ("Anxiety", "Depression", "Emotional Volatility"),
    "Open-Mindedness":       ("Aesthetic Sensitivity", "Intellectual Curiosity", "Creative Imagination"),
}

DOMAIN_TO_OCEAN = {
    "Extraversion": "E",
    "Agreeableness": "A",
    "Conscientiousness": "C",
    "Negative Emotionality": "N",
    "Open-Mindedness": "O",
}

LICENSE_NOTICE = (
    "BFI-2 items copyright 2015 by Oliver P. John and Christopher J. Soto. "
    "Reprinted with permission. Free for non-commercial research use; "
    "commercial use requires the authors' explicit permission "
    "(ucbpersonalitylab@gmail.com) and is not currently granted. "
    "INTERNAL EVALUATION USE ONLY -- do not ship to paying customers."
)

CITATION = (
    "Soto, C. J., & John, O. P. (2017). Short and extra-short forms of the "
    "Big Five Inventory-2: The BFI-2-S and BFI-2-XS. Journal of Research in "
    "Personality, 68, 69-81. Appendix A (30-item BFI-2-S)."
)

RELIABILITY_NOTE = (
    "The BFI-2-S is the official 30-item short form of the 60-item BFI-2 (2 "
    "items/facet instead of 4). Its authors report it retains ~90-91% of the "
    "full form's domain-level reliability and validity, and recommend "
    "facet-level scores only be trusted in samples of ~400+ observations -- "
    "a condition a single administration to one agent does not meet. Facet "
    "scores here are exploratory description, not a validated measurement, "
    "independent of the separate fact that no facet-level ground truth "
    "exists for any HEART character either."
)


def build_instrument():
    items = tuple(Item(number=n, text=text, domain=domain, facet=facet,
                       reverse_keyed=rev)
                  for n, text, domain, facet, rev in _ROWS)
    return Instrument(
        id="bfi2s",
        name="BFI-2-S",
        full_name="Big Five Inventory-2, Short Form",
        stem=STEM,
        scale_min=1,
        scale_max=5,
        scale_labels=dict(SCALE_LABELS),
        items=items,
        domain_order=DOMAIN_ORDER,
        facet_order={k: tuple(v) for k, v in FACET_ORDER.items()},
        domain_to_ocean=dict(DOMAIN_TO_OCEAN),
        license_notice=LICENSE_NOTICE,
        citation=CITATION,
        facets_are_exploratory=True,
        reliability_note=RELIABILITY_NOTE,
    )
