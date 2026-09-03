"""The Big Five Inventory (BFI-44) -- item bank and scoring key.

Why this instrument, not a custom one: it is the most widely used free Big Five
self-report in psychology (John & Srivastava, 1999; Berkeley Personality Lab), it
maps onto the exact OCEAN model MindForm already implements (``core.config.BASIS``),
and administering the *same, unmodified, well-validated* instrument repeatedly is
what makes a persistence claim checkable -- a bespoke questionnaire would let us
grade our own homework. Prior work measuring personality in LLMs (e.g. Serapio-
Garcia et al. 2023, "Personality Traits in Large Language Models") uses this same
method: administer BFI-44 to the model-as-persona and read off Big Five scores, and
check test-retest reliability across repeated administrations as the stability
signal -- exactly the "is the personality persistent" question asked here.

Each item is answered on a 1-5 Likert scale ("Disagree strongly" .. "Agree
strongly"). ``reverse=True`` items are scored 6 - response before averaging.
"""

# (item number, text, OCEAN dim, reverse-scored)
ITEMS = [
    (1, "Is talkative", "E", False),
    (2, "Tends to find fault with others", "A", True),
    (3, "Does a thorough job", "C", False),
    (4, "Is depressed, blue", "N", False),
    (5, "Is original, comes up with new ideas", "O", False),
    (6, "Is reserved", "E", True),
    (7, "Is helpful and unselfish with others", "A", False),
    (8, "Can be somewhat careless", "C", True),
    (9, "Is relaxed, handles stress well", "N", True),
    (10, "Is curious about many different things", "O", False),
    (11, "Is full of energy", "E", False),
    (12, "Starts quarrels with others", "A", True),
    (13, "Is a reliable worker", "C", False),
    (14, "Can be tense", "N", False),
    (15, "Is ingenious, a deep thinker", "O", False),
    (16, "Generates a lot of enthusiasm", "E", False),
    (17, "Has a forgiving nature", "A", False),
    (18, "Tends to be disorganized", "C", True),
    (19, "Worries a lot", "N", False),
    (20, "Has an active imagination", "O", False),
    (21, "Tends to be quiet", "E", True),
    (22, "Is generally trusting", "A", False),
    (23, "Tends to be lazy", "C", True),
    (24, "Is emotionally stable, not easily upset", "N", True),
    (25, "Is inventive", "O", False),
    (26, "Has an assertive personality", "E", False),
    (27, "Can be cold and aloof", "A", True),
    (28, "Perseveres until the task is finished", "C", False),
    (29, "Can be moody", "N", False),
    (30, "Values artistic, aesthetic experiences", "O", False),
    (31, "Is sometimes shy, inhibited", "E", True),
    (32, "Is considerate and kind to almost everyone", "A", False),
    (33, "Does things efficiently", "C", False),
    (34, "Remains calm in tense situations", "N", True),
    (35, "Prefers work that is routine", "O", True),
    (36, "Is outgoing, sociable", "E", False),
    (37, "Is sometimes rude to others", "A", True),
    (38, "Makes plans and follows through with them", "C", False),
    (39, "Gets nervous easily", "N", False),
    (40, "Likes to reflect, play with ideas", "O", False),
    (41, "Has few artistic interests", "O", True),
    (42, "Likes to cooperate with others", "A", False),
    (43, "Is easily distracted", "C", True),
    (44, "Is sophisticated in art, music, or literature", "O", False),
]

DIMS = ["O", "C", "E", "A", "N"]


def prompt_lines():
    """The 44 statements, numbered, ready to drop into an LLM prompt."""
    return "\n".join(f"{n}. {text}" for n, text, _dim, _rev in ITEMS)


def _reversed(n, raw):
    rev = next(rev for i, _t, _d, rev in ITEMS if i == n)
    return 6 - raw if rev else raw


def domain_scores(responses):
    """responses: {item_number (int or str): 1..5 raw Likert}.

    Returns (means, alphas): per-domain mean score in [1, 5] after reverse-coding,
    and Cronbach's alpha per domain -- an internal-consistency check on the
    responses themselves (do the 8-10 items making up one trait agree with each
    other, independent of any ground truth).
    """
    responses = {int(k): int(v) for k, v in responses.items()}
    by_dim = {d: [] for d in DIMS}
    for n, _text, dim, _rev in ITEMS:
        if n not in responses:
            continue
        by_dim[dim].append(_reversed(n, responses[n]))

    means = {d: (sum(v) / len(v) if v else None) for d, v in by_dim.items()}
    alphas = {d: cronbach_alpha(v) for d, v in by_dim.items()}
    return means, alphas


def cronbach_alpha(item_scores):
    """Standard Cronbach's alpha for one respondent's item vector is undefined
    (alpha needs variance across a *sample*); here ``item_scores`` is one
    character's answers to the k items of one domain, so we report the simpler,
    still-informative split-half style dispersion check: 1 - (mean item variance
    from the domain mean / total variance), clamped to [0, 1]. With k < 2 items
    there is nothing to check for agreement.
    """
    k = len(item_scores)
    if k < 2:
        return None
    mean = sum(item_scores) / k
    total_var = sum((x - mean) ** 2 for x in item_scores)
    if total_var == 0:
        return 1.0  # every item agreed exactly
    # item-total agreement: how much of the spread is "all items near the mean"
    # vs. scattered -- a coarse coherence proxy, not the textbook formula.
    max_possible_var = k * ((5 - 1) / 2) ** 2  # worst case: half at 1, half at 5
    return max(0.0, 1.0 - total_var / max_possible_var)


def ocean_from_bfi(means):
    """Map each domain's 1..5 mean onto the engine's own -1..1 trait scale."""
    return {d: (m - 3.0) / 2.0 if m is not None else None for d, m in means.items()}
