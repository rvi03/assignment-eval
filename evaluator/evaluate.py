"""Evaluate one case: code checks, one judge call, then the verdict table.

    case ─► format check ─► judge ─► reply checks ─► verdict table ─► quote check
                (code)      (LLM)      (code)           (code)          (code)

The quality score comes from a second fixed table over the same
answers, so it can never contradict them.

Every failure of the judge becomes a failed prediction with a clear
error, never a verdict. Evaluator uncertainty is therefore never mixed
up with ESCALATE, which means the case itself needs a human.
"""

from evaluator.checks import check_json_format, find_unverified_quotes
from evaluator.judge import ASSESSMENT_SCHEMA, InvalidAssessment, build_messages, parse_assessment
from evaluator.llm import CallError, JudgeClient
from evaluator.models import CaseInput, Prediction
from evaluator.verdict import decide_quality, decide_verdict

_REASK = (
    "Your reply did not follow the required schema: {error}. "
    "Reply again with only the JSON object, following the schema exactly."
)

def evaluate_case(case: CaseInput, client: JudgeClient) -> Prediction:
    format_check = check_json_format(case)
    messages = build_messages(case, format_check)

    try:
        assessment = _ask(client, messages)
    except _Failure as failure:
        return Prediction(
            case_id=case.case_id,
            status="failed",
            verdict=None,
            quality_score=None,
            policy_violation=None,
            reason=f"Evaluator error ({failure.kind}): {failure.message}",
            error_type=failure.kind,
            format_check=format_check,
        )

    if assessment["assessable"] == "no":
        return Prediction(
            case_id=case.case_id,
            status="failed",
            verdict=None,
            quality_score=None,
            policy_violation=None,
            reason=f"Evaluator error (not_assessable): {assessment['reason']}",
            error_type="not_assessable",
            format_check=format_check,
            assessment=assessment,
        )

    verdict, policy_violation = decide_verdict(assessment)
    return Prediction(
        case_id=case.case_id,
        status="ok",
        verdict=verdict,
        quality_score=decide_quality(assessment),
        policy_violation=policy_violation,
        reason=assessment["reason"].strip(),
        format_check=format_check,
        unverified_quotes=len(find_unverified_quotes(case, assessment["evidence"])),
        assessment=assessment,
    )


class _Failure(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def _ask(client: JudgeClient, messages: list[dict]) -> dict:
    """Call the judge; re-ask once if the reply breaks the schema."""
    content = _call(client, messages)
    try:
        return parse_assessment(content)
    except InvalidAssessment as first:
        retry = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": _REASK.format(error=first)},
        ]
        content = _call(client, retry)
        try:
            return parse_assessment(content)
        except InvalidAssessment as second:
            raise _Failure("invalid_reply", f"{second} (after one re-ask)") from second


def _call(client: JudgeClient, messages: list[dict]) -> str:
    try:
        reply = client.complete(messages, ASSESSMENT_SCHEMA)
    except CallError as exc:
        raise _Failure("api_error", str(exc)) from exc
    if reply.refusal or reply.finish_reason == "content_filter":
        raise _Failure("refused", reply.refusal or "the response was blocked by a content filter")
    if reply.finish_reason == "length":
        raise _Failure("truncated", "the response was cut off before it was complete")
    if not reply.content or not reply.content.strip():
        raise _Failure("empty", "the response had no content")
    return reply.content
