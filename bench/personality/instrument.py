"""The contract every personality instrument implements.

``runner.py``, ``report.py`` and ``live.py`` are written against this
interface only -- none of them mention "BFI-2" anywhere in their logic. Adding
a second instrument (say, IPIP-NEO-120, if that's ever decided) means writing
one more module that builds an ``Instrument`` and pointing the UI at it; the
administration loop, the audit log, the scoring pipeline and the results
screen do not change.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    """One questionnaire item, in the instrument's own official numbering."""
    number: int
    text: str            # appended to the instrument's stem, verbatim official wording
    facet: str            # official facet name
    domain: str           # official domain name
    reverse_keyed: bool   # true-keyed items still count toward a domain/facet's meaning;
                          # false-keyed ("reverse") items must be recoded before averaging


@dataclass(frozen=True)
class Instrument:
    """A complete, scoreable questionnaire."""
    id: str                     # short slug, e.g. "bfi2s" -- used in filenames/URLs
    name: str                   # exact public name, e.g. "BFI-2-S"
    full_name: str              # e.g. "Big Five Inventory-2, Short Form"
    stem: str                   # the common item stem, e.g. "I am someone who..."
    scale_min: int
    scale_max: int
    scale_labels: dict          # {1: "Disagree strongly", ..., 5: "Agree strongly"}
    items: tuple                # tuple[Item, ...], in official numbering order
    domain_order: tuple         # official domain display order
    facet_order: dict           # domain -> tuple of its facet names, official order
    domain_to_ocean: dict       # explicit, disclosed correspondence to MindForm's O/C/E/A/N --
                                # never assumed silently. e.g. {"Negative Emotionality": "N"}
    license_notice: str         # copyright + usage terms, shown wherever results are shown
    citation: str               # exact source of the item text and scoring key
    facets_are_exploratory: bool = True   # true unless independent facet ground truth exists
    reliability_note: str = ""

    def __post_init__(self):
        nums = [it.number for it in self.items]
        if len(nums) != len(set(nums)):
            raise ValueError(f"{self.id}: duplicate item numbers")
        domains = {it.domain for it in self.items}
        if domains != set(self.domain_order):
            raise ValueError(f"{self.id}: item domains {domains} != domain_order "
                             f"{set(self.domain_order)}")
        for d in self.domain_order:
            facets_in_items = {it.facet for it in self.items if it.domain == d}
            if facets_in_items != set(self.facet_order.get(d, ())):
                raise ValueError(f"{self.id}: facet mismatch for domain {d!r}: "
                                 f"items have {facets_in_items}, facet_order has "
                                 f"{set(self.facet_order.get(d, ()))}")
        if set(self.domain_to_ocean) != set(self.domain_order):
            raise ValueError(f"{self.id}: domain_to_ocean must cover exactly the "
                             f"instrument's domains")
        if len(set(self.domain_to_ocean.values())) != len(self.domain_order):
            raise ValueError(f"{self.id}: domain_to_ocean must map onto distinct "
                             f"OCEAN letters")

    def item(self, number):
        return next(it for it in self.items if it.number == number)

    def score(self, responses):
        """``responses``: {item_number: int in [scale_min, scale_max]}.

        Returns a full audit record: every item's keyed value, every facet's
        score, every domain's score (raw instrument scale AND rescaled to
        MindForm's -1..+1, via the same kind of disclosed linear transform
        ``fidelity.py`` already uses for HEART's 0..1 scale -- never an
        invented mapping).
        """
        mid = (self.scale_min + self.scale_max) / 2.0
        half_range = (self.scale_max - self.scale_min) / 2.0
        missing = [it.number for it in self.items if it.number not in responses]
        if missing:
            raise ValueError(f"missing responses for items {missing}")
        for n, v in responses.items():
            if not (self.scale_min <= v <= self.scale_max):
                raise ValueError(f"item {n}: response {v} outside "
                                 f"[{self.scale_min}, {self.scale_max}]")

        keyed = {}
        for it in self.items:
            raw = responses[it.number]
            keyed[it.number] = (self.scale_min + self.scale_max - raw) if it.reverse_keyed else raw

        facet_scores = {}
        for d in self.domain_order:
            for f in self.facet_order[d]:
                vals = [keyed[it.number] for it in self.items if it.domain == d and it.facet == f]
                facet_scores[f] = sum(vals) / len(vals)

        domain_scores = {}
        domain_signed = {}
        for d in self.domain_order:
            facet_vals = [facet_scores[f] for f in self.facet_order[d]]
            score = sum(facet_vals) / len(facet_vals)
            domain_scores[d] = score
            domain_signed[d] = round((score - mid) / half_range, 4)

        return {
            "instrument": self.id,
            "keyed_responses": keyed,
            "facet_scores": {k: round(v, 4) for k, v in facet_scores.items()},
            "domain_scores": {k: round(v, 4) for k, v in domain_scores.items()},
            "domain_scores_signed": domain_signed,          # -1..+1, comparable to MindForm/HEART
            "ocean_signed": {self.domain_to_ocean[d]: domain_signed[d] for d in self.domain_order},
        }
