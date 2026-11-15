"""Deterministic checks that code can make exactly, without a model.

Agent-evaluation frameworks prefer code checks wherever they cannot give a
false positive. The benchmark has no structured tool calls, reference
trajectories or environment state, so the only exact checks left are on
the text itself: the required output format and the evidence quotes.
"""

import json
import re

from evaluator.models import CaseInput

_JSON_MENTION = re.compile(r"\bjson\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def check_json_format(case: CaseInput) -> bool | None:
    """Return whether the final answer is valid JSON, if JSON is asked for.

    Returns None when neither the task nor the constraints mention JSON,
    so the check does not apply. A mention of JSON does not always mean
    "JSON only", so the result is passed to the judge as a fact rather
    than deciding the verdict.
    """
    if not (_JSON_MENTION.search(case.task) or _JSON_MENTION.search(case.constraints)):
        return None
    try:
        json.loads(case.agent_final_answer.strip())
    except ValueError:
        return False
    return True


def tools_text(case: CaseInput) -> str:
    """The tool list exactly as the judge sees it ("none" when empty)."""
    return ", ".join(case.available_tools) if case.available_tools else "none"


def source_text(case: CaseInput, source: str) -> str:
    """The case text an evidence item may quote from, as the judge saw it."""
    if source == "available_tools":
        return tools_text(case)
    return getattr(case, source)


def find_unverified_quotes(case: CaseInput, evidence: list[dict]) -> list[dict]:
    """Return the evidence items whose quote does not appear in its source.

    Matching ignores case and collapses whitespace. A match proves the
    quote exists in the case; it does not prove the quote supports the
    judgment.
    """
    misses = []
    for item in evidence:
        if item["source"] == "none":
            continue
        if _normalize(item["quote"]) not in _normalize(source_text(case, item["source"])):
            misses.append(item)
    return misses


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()
