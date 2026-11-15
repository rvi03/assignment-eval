"""Read and check a predictions file before it is scored.

The file is a JSON array with one object per case:

    {
      "case_id": "AE-001",
      "status": "ok",
      "verdict": "PASS",
      "quality_score": 4,
      "policy_violation": false,
      "reason": "..."
    }

- `status` is "ok" or "failed" and defaults to "ok" when it is absent.
- A failed record has null labels; its `reason` says what went wrong, and
  `error_type` may name the kind of failure.
- Other fields are ignored, so a file may carry extra diagnostics.

Every problem in the file is collected and reported at once, so one run
shows everything that needs fixing.
"""

import json
from pathlib import Path

from evaluator.models import QUALITY_SCORES, VERDICTS, Prediction

STATUSES = ("ok", "failed")
_LABEL_FIELDS = ("verdict", "quality_score", "policy_violation")


class PredictionsFileError(Exception):
    """The predictions file cannot be scored; the message lists each problem."""

    def __init__(self, path: Path, problems: list[str]):
        self.problems = problems
        listed = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Predictions file {path} cannot be scored:\n{listed}")


def read_predictions(path: Path | str, expected_ids: list[str]) -> list[Prediction]:
    """Return one Prediction per expected case_id, in the file's order.

    Raises PredictionsFileError when the file is unreadable, a record is
    invalid, or the case_ids do not match `expected_ids` exactly.
    """
    path = Path(path)
    records = _read_json(path)

    problems: list[str] = []
    predictions: list[Prediction] = []
    seen: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        prediction = _parse_record(index, record, problems)
        if prediction is None:
            continue
        if prediction.case_id in seen:
            problems.append(
                f"Record {index}: case_id {prediction.case_id!r} repeats "
                f"record {seen[prediction.case_id]}."
            )
            continue
        seen[prediction.case_id] = index
        predictions.append(prediction)

    expected = set(expected_ids)
    unknown = [p.case_id for p in predictions if p.case_id not in expected]
    if unknown:
        problems.append(
            "case_ids not in the selected split of the data file: "
            + ", ".join(unknown)
            + "."
        )
    missing = [case_id for case_id in expected_ids if case_id not in seen]
    if missing:
        problems.append("No prediction for case_ids: " + ", ".join(missing) + ".")

    if problems:
        raise PredictionsFileError(path, problems)
    return predictions


def _read_json(path: Path) -> list:
    if not path.exists():
        raise PredictionsFileError(path, ["The file does not exist."])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise PredictionsFileError(path, [f"The file cannot be read: {exc}"]) from exc
    except json.JSONDecodeError as exc:
        raise PredictionsFileError(
            path,
            [f"Malformed JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."],
        ) from exc
    if not isinstance(data, list):
        raise PredictionsFileError(
            path, ["The file must hold a JSON array of prediction objects."]
        )
    return data


def _parse_record(index: int, record, problems: list[str]) -> Prediction | None:
    where = f"Record {index}"
    if not isinstance(record, dict):
        problems.append(f"{where}: must be a JSON object.")
        return None

    case_id = record.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        problems.append(f"{where}: case_id is missing or not a non-empty string.")
        return None
    case_id = case_id.strip()
    where = f"{where} ({case_id})"

    found: list[str] = []
    status = record.get("status", "ok")
    if status not in STATUSES:
        found.append(f"status {status!r} is not one of: {', '.join(STATUSES)}.")

    reason = record.get("reason")
    if not isinstance(reason, str):
        found.append("reason is missing or not a string.")

    verdict = record.get("verdict")
    quality = record.get("quality_score")
    policy = record.get("policy_violation")
    if status == "ok":
        if verdict not in VERDICTS:
            found.append(f"verdict {verdict!r} is not one of: {', '.join(VERDICTS)}.")
        # bool is a subclass of int in Python, so rule it out explicitly.
        if isinstance(quality, bool) or quality not in QUALITY_SCORES:
            found.append(f"quality_score {quality!r} is not an integer from 0 to 4.")
        if not isinstance(policy, bool):
            found.append(f"policy_violation {policy!r} is not true or false.")
    elif status == "failed":
        filled = [name for name in _LABEL_FIELDS if record.get(name) is not None]
        if filled:
            found.append(
                "a failed record must have null labels, but has: "
                + ", ".join(filled)
                + "."
            )

    if found:
        problems.extend(f"{where}: {message}" for message in found)
        return None

    error_type = record.get("error_type")
    return Prediction(
        case_id=case_id,
        status=status,
        verdict=verdict if status == "ok" else None,
        quality_score=quality if status == "ok" else None,
        policy_violation=policy if status == "ok" else None,
        reason=reason.strip(),
        error_type=error_type if isinstance(error_type, str) else None,
    )
