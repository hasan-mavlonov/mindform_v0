"""Shared OpenAI-compatible JSON call for the formation engines (Gemma-hardened).

Every formation node -- trait push (``llm_impact``), values, moral, belief, and genesis
(``temperament``) -- asks the configured chat model (Gemini 3.5 Flash by default) for a
small JSON object. Gemma emits a ``<thought>`` block before its answer that cannot be
turned off, so a too-small token budget truncates the reply mid-thought and leaves no
JSON to parse (the classic ``Expecting value: line 1 column 1 (char 0)``).

This helper centralizes the fix:
  * a generous ``max_tokens`` so the thought can finish AND the JSON still follows;
  * ``config.parse_json_object`` to strip the reasoning / code fences before parsing;
  * one retry that nudges the model to emit JSON only, for the rare truncation.

It raises on failure (no key, transport error after the client's own retries, or no
parseable JSON) so callers fall back to their deterministic heuristic. Transport retries
on 5xx are left to the OpenAI client; the retry here is specifically for the parse-empty
case Gemma's ``<thought>`` causes.
"""

import logging

from core.config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, parse_json_object

log = logging.getLogger("mindform.llm")

# Gemma's <thought> can run long; give it room to finish and still reach the JSON.
# max_tokens is a ceiling, not a target -- short replies stop early and cost nothing extra.
JSON_MAX_TOKENS = 2048
_JSON_ONLY_NUDGE = ("\n\nReturn ONLY the JSON object now -- no <thought>, no thinking, "
                    "no explanation before or after it.")


def complete_json(system_prompt, user_content, *, temperature=0.2,
                  max_tokens=JSON_MAX_TOKENS, retries=1):
    """Call the configured model and return a parsed JSON object (a dict).

    Raises ``RuntimeError`` when no API key is set, propagates a transport error if the
    request fails, and re-raises the last parse error if every attempt yields no JSON --
    so the caller can fall back to its heuristic.
    """
    if not LLM_API_KEY:
        raise RuntimeError("no LLM API key is set (GEMINI_API_KEY)")

    from openai import OpenAI  # lazy: the heuristic fallbacks work without this package

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    last_exc = None
    for attempt in range(retries + 1):
        content = user_content if attempt == 0 else user_content + _JSON_ONLY_NUDGE
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
        )
        try:
            return parse_json_object(completion.choices[0].message.content)
        except ValueError as exc:   # truncated mid-thought / no JSON -> nudge and retry
            last_exc = exc
            log.info("JSON parse failed (attempt %d/%d): %s",
                     attempt + 1, retries + 1, exc)
    raise last_exc
