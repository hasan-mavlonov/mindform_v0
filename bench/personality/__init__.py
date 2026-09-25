"""Personality testing framework -- separate from, and never coupled to, bench/heart.

    Personality Test  -> "What personality did MindForm form?"
    Behavioral Test   -> "Does that personality actually behave like the target?"
                          (bench/heart -- unchanged, untouched by this package)

This package administers a standardized, published personality inventory to an
already-frozen MindForm character and scores it with the instrument's own
official key. It answers a different question than bench/heart, through a
different mechanism (self-report, not dramatic-scenario choice), and the two
are never merged: no shared runner, no shared leak-checker, no shared UI. The
only things imported from bench/heart are read-only data primitives that
bench/heart itself already treats as shared infrastructure, not methodology --
``heartdata`` (character roster, occupation, the forbidden-field list),
``mfadapter`` (restore/read a frozen snapshot), ``snapshots`` (find one), and
``config.require_heart_bench`` (fail fast with instructions). Nothing here
imports bench.heart.arms, bench.heart.runner, or bench.heart.live -- the MCQ
scenarios, retrieval arms, and per-question scoring that make HEART what it is.

Instruments are pluggable (see ``instrument.py``): today there is exactly one,
``bfi2`` (see that package's own module docstring for what it is and, just as
importantly, what it is NOT -- it is a validated 30-item SHORT FORM of the
BFI-2, not the full 60-item instrument). Adding a second instrument later
should mean adding a second subpackage next to ``bfi2/``, not touching
``runner.py``, ``report.py``, or ``live.py``.
"""
