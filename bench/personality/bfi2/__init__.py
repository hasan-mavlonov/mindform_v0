"""BFI-2-S -- the officially published SHORT FORM of the Big Five Inventory-2.

Read this before assuming this is "the BFI-2": it is not the full 60-item
instrument. It is the 30-item BFI-2-S, the same authors' own validated short
form (2 items per facet instead of 4), used here because its exact item text
and scoring key are verifiably, legitimately public -- reprinted with the
copyright holders' explicit permission in Appendix A of:

    Soto, C. J., & John, O. P. (2017). Short and extra-short forms of the Big
    Five Inventory-2: The BFI-2-S and BFI-2-XS. Journal of Research in
    Personality, 68, 69-81. https://doi.org/10.1016/j.jrp.2017.02.004
    (openly hosted by Colby College: colby.edu/wp-content/uploads/2013/08/
    Soto_John_2017b.pdf, fetched and transcribed directly from that PDF)

The full 60-item BFI-2 exists too, but its item text and scoring instructions
are gated behind a registration survey at the authors' own site
(ocf.berkeley.edu/~johnlab/bfi.html) that requires a human to disclose their
identity and research purpose -- something only the person running this
project can do, not this agent on their behalf. Rather than fabricate the
other 30 items from memory or silently substitute a different instrument
(explicitly ruled out), this module uses the one version whose text is both
official and independently verifiable right now. See LICENSE_NOTICE below;
swapping in the full 60-item form later is a one-file change (this module),
because everything downstream (runner/report/live) is written against the
generic ``Instrument`` interface, not against "BFI-2" specifically.

Licensing (verified directly from ocf.berkeley.edu/~johnlab/bfi.html):
  "Christopher J. Soto and I hold the copyright to the BFI-2 and it is not in
  the public domain per se." Free for non-commercial research use. Commercial
  use is "not currently permitted" without a separate request to the authors
  (ucbpersonalitylab@gmail.com). THIS MODULE IS FOR INTERNAL EVALUATION AND
  RESEARCH USE ONLY. It must not ship in a commercial product or be exposed to
  paying customers without first obtaining the authors' explicit permission.
"""

from bench.personality.bfi2.items import build_instrument

INSTRUMENT = build_instrument()
