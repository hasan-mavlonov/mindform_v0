"""MindForm Console -- a web UI integration layer over the personality engine.

This package is the *integration* surface only. It imports the engine modules
(config, temperament, appraisal, llm_impact, updater, personality, memory,
encoder) and orchestrates them exactly as ``interactive.py`` does, then serves
the result to a browser cockpit. It never modifies the engine's logic.

Run it with ``python console.py`` (repo root) or ``python -m web.server``.
"""
