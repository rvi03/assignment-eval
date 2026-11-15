"""Load and validate the benchmark CSV.

- A missing or unreadable file, or a missing header column, stops the load
  with DataFileError, because no row can be read correctly then.
- Every other problem makes that row a RowError. Valid rows still load,
  and no invalid row is dropped silently.
- For case_id, task, constraints, available_tools, agent_trajectory and
  agent_final_answer, a filled top-level column wins; state_json is used
  only when the column is empty.
- Every row with a duplicated case_id is an error, because the loader
  cannot tell which copy is correct.
- The whole file is read into memory.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluator.models import (
    DIFFICULTIES,
    QUALITY_SCORES,
    REQUIRED_COLUMNS,
    SPLITS,
    VERDICTS,
    Case,
    CaseInput,
    DataFileError,
    FieldError,
    GoldLabels,
    LoadResult,
    RowError,
)

DEFAULT_DATA_PATH = Path(".data/ai_agent_evaluator_benchmark.csv")
EXTRA_FIELDS_KEY = "_extra_fields"

# Fields that may come from the column or, when it is empty, from state_json.
_TEXT_FIELDS = ("task", "constraints", "agent_trajectory", "agent_final_answer")
_NO_TOOLS = "none"


def load_cases(path: Path | str = DEFAULT_DATA_PATH) -> LoadResult:
    """Read the CSV at `path` and return valid cases and row errors."""
    rows = _read_rows(Path(path))
    rows_by_id = defaultdict(list)
    for row_number, raw in rows:
        case_id = (raw.get("case_id") or "").strip() or _case_id_from_state(raw)
        if case_id:
            rows_by_id[case_id].append(row_number)

    result = LoadResult()
    for row_number, raw in rows:
        outcome = _parse_row(row_number, raw, rows_by_id)
        if isinstance(outcome, RowError):
            result.errors.append(outcome)
        else:
            result.cases.append(outcome)
    return result


def write_errors(errors: list[RowError], path: Path | str) -> None:
    """Write one JSON line per invalid row: reasons plus the raw record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for error in errors:
            line = {
                "row": error.row,
                "case_id": error.case_id,
                "errors": [
                    {"field": e.field, "message": e.message} for e in error.errors
                ],
                "record": error.record,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def _read_rows(path: Path) -> list[tuple[int, dict]]:
    if not path.exists():
        raise DataFileError(
            f"Data file not found: {path}. Place the benchmark CSV at this path."
        )
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, restkey=EXTRA_FIELDS_KEY)
            header = reader.fieldnames
            if not header:
                raise DataFileError(f"Data file is empty or has no header: {path}.")
            missing = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing:
                raise DataFileError(
                    f"Data file {path} is missing required columns: "
                    f"{', '.join(missing)}."
                )
            # Row 1 is the header, so the first data row is row 2.
            return [(index + 2, raw) for index, raw in enumerate(reader)]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DataFileError(f"Cannot read data file {path}: {exc}") from exc


def _case_id_from_state(raw: dict) -> str:
    try:
        state = json.loads(raw.get("state_json") or "")
    except json.JSONDecodeError:
        return ""
    value = state.get("case_id") if isinstance(state, dict) else None
    return value.strip() if isinstance(value, str) else ""


def _parse_row(row_number: int, raw: dict, rows_by_id: dict) -> Case | RowError:
    errors: list[FieldError] = []

    if raw.get(EXTRA_FIELDS_KEY) is not None or None in raw.values():
        errors.append(
            FieldError("row", "Row has a different number of fields than the header.")
        )

    state, state_error = _parse_state(raw.get("state_json"))
    if state_error:
        errors.append(FieldError("state_json", state_error))

    case_id = _resolve_text(raw, state, "case_id", errors, state_error)
    if case_id and len(rows_by_id.get(case_id, [])) > 1:
        rows = ", ".join(str(n) for n in rows_by_id[case_id])
        errors.append(
            FieldError("case_id", f"Duplicate case_id {case_id!r} in rows {rows}.")
        )

    text = {
        name: _resolve_text(raw, state, name, errors, state_error)
        for name in _TEXT_FIELDS
    }
    tools = _resolve_tools(raw, state, errors, state_error)

    split = _clean(raw.get("split"))
    if split not in SPLITS:
        errors.append(_not_allowed("split", split, SPLITS))
    difficulty = _clean(raw.get("difficulty"))
    if difficulty not in DIFFICULTIES:
        errors.append(_not_allowed("difficulty", difficulty, DIFFICULTIES))

    verdict = _clean(raw.get("gold_verdict"))
    if verdict not in VERDICTS:
        errors.append(_not_allowed("gold_verdict", verdict, VERDICTS))
    quality = _parse_quality(_clean(raw.get("gold_quality")), errors)
    policy = _parse_policy(_clean(raw.get("gold_policy_violation")), errors)

    if errors:
        return RowError(
            row=row_number,
            case_id=case_id or None,
            errors=tuple(errors),
            record=dict(raw),
        )

    case_input = CaseInput(
        case_id=case_id,
        split=split,
        capability=_clean(raw.get("capability")),
        domain=_clean(raw.get("domain")),
        difficulty=difficulty,
        language=_clean(raw.get("language")),
        task=text["task"],
        constraints=text["constraints"],
        available_tools=tools,
        agent_trajectory=text["agent_trajectory"],
        agent_final_answer=text["agent_final_answer"],
    )
    labels = GoldLabels(
        case_id=case_id,
        verdict=verdict,
        quality=quality,
        policy_violation=policy,
        rationale=_clean(raw.get("gold_rationale")),
        trap=_clean(raw.get("trap")),
    )
    return Case(input=case_input, labels=labels)


def _parse_state(value: str | None) -> tuple[dict, str | None]:
    if not _clean(value):
        return {}, "state_json is empty."
    try:
        state = json.loads(value)
    except json.JSONDecodeError as exc:
        return {}, (
            f"Malformed JSON in state_json: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        )
    if not isinstance(state, dict):
        return {}, "state_json must be a JSON object."
    return state, None


def _resolve_text(
    raw: dict, state: dict, name: str, errors: list, state_error: str | None
) -> str:
    column = _clean(raw.get(name))
    if column:
        return column
    value = state.get(name)
    if value is not None and not isinstance(value, str):
        errors.append(FieldError(name, f"{name} in state_json must be a string."))
        return ""
    value = (value or "").strip()
    # A broken state_json is already reported; do not repeat it per field.
    if not value and not state_error:
        errors.append(
            FieldError(name, f"{name} is empty in both the column and state_json.")
        )
    return value


def _resolve_tools(
    raw: dict, state: dict, errors: list, state_error: str | None
) -> tuple[str, ...]:
    column = _clean(raw.get("available_tools"))
    if column:
        if column.lower() == _NO_TOOLS:
            return ()
        return tuple(t.strip() for t in column.split(",") if t.strip())
    if state_error:
        return ()
    value = state.get("available_tools")
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        errors.append(
            FieldError(
                "available_tools",
                "available_tools is empty in the column and is not a list of "
                "strings in state_json.",
            )
        )
        return ()
    return tuple(t.strip() for t in value if t.strip())


def _parse_quality(value: str, errors: list) -> int:
    if value.isdigit() and int(value) in QUALITY_SCORES:
        return int(value)
    errors.append(
        FieldError(
            "gold_quality", f"gold_quality {value!r} is not an integer from 0 to 4."
        )
    )
    return -1


def _parse_policy(value: str, errors: list) -> bool:
    if value in ("true", "false"):
        return value == "true"
    errors.append(
        FieldError(
            "gold_policy_violation",
            f"gold_policy_violation {value!r} is not 'true' or 'false'.",
        )
    )
    return False


def _not_allowed(name: str, value: str, allowed: tuple) -> FieldError:
    return FieldError(
        name, f"{name} {value!r} is not one of: {', '.join(allowed)}."
    )


def _clean(value: str | None) -> str:
    return value.strip() if isinstance(value, str) else ""
