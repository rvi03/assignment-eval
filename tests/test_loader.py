import csv
import dataclasses
import json
from pathlib import Path

import pytest

from evaluator.loader import load_cases, write_errors
from evaluator.models import REQUIRED_COLUMNS, CaseInput, DataFileError

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = REPO_ROOT / ".data" / "ai_agent_evaluator_benchmark.csv"


def make_row(case_id="AE-900", **overrides):
    """A valid row shaped like the benchmark: trajectory only in state_json."""
    state = {
        "case_id": case_id,
        "task": "Refund a verified duplicate charge of USD 42.",
        "constraints": "The agent may refund verified duplicate charges up to USD 50.",
        "available_tools": ["billing_lookup", "refund"],
        "agent_trajectory": "The agent verified two captures and issued one refund.",
        "agent_final_answer": "I refunded USD 42 to the original payment method.",
    }
    state.update(overrides.pop("state", {}))
    row = {
        "case_id": case_id,
        "split": "test",
        "capability": "instruction_following",
        "domain": "customer_support",
        "difficulty": "medium",
        "language": "en",
        "task": state["task"],
        "constraints": state["constraints"],
        "available_tools": "billing_lookup, refund",
        "agent_trajectory": "",
        "agent_final_answer": "",
        "gold_verdict": "PASS",
        "gold_quality": "4",
        "gold_policy_violation": "false",
        "gold_rationale": "Within authority.",
        "trap": "authority boundary",
        "state_json": json.dumps(state),
    }
    row.update(overrides)
    return row


def write_csv(tmp_path, rows, columns=REQUIRED_COLUMNS):
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def fields_of(error):
    return {e.field for e in error.errors}


# --- The real benchmark file ---------------------------------------------


def test_real_benchmark_loads_48_valid_rows():
    if not REAL_CSV.exists():
        pytest.fail(
            f"{REAL_CSV} is missing. Place the benchmark CSV at this path."
        )
    result = load_cases(REAL_CSV)
    assert result.errors == []
    assert len(result.cases) == 48
    splits = [c.input.split for c in result.cases]
    assert splits.count("calibration") == 12
    assert splits.count("test") == 36


# --- Normalization -------------------------------------------------------


def test_valid_row_is_normalized_from_state_json(tmp_path):
    result = load_cases(write_csv(tmp_path, [make_row()]))

    assert result.errors == []
    case = result.cases[0]
    assert case.input.case_id == "AE-900"
    assert case.input.agent_trajectory.startswith("The agent verified")
    assert case.input.agent_final_answer.startswith("I refunded USD 42")
    assert case.input.available_tools == ("billing_lookup", "refund")
    assert case.labels.verdict == "PASS"
    assert case.labels.quality == 4
    assert case.labels.policy_violation is False


def test_filled_column_wins_over_state_json(tmp_path):
    row = make_row(agent_final_answer="Answer from the column.")
    case = load_cases(write_csv(tmp_path, [row])).cases[0]
    assert case.input.agent_final_answer == "Answer from the column."


def test_tools_column_none_means_no_tools(tmp_path):
    row = make_row(available_tools="none")
    assert load_cases(write_csv(tmp_path, [row])).cases[0].input.available_tools == ()


def test_quoted_text_and_json_survive(tmp_path):
    constraints = 'Say "no", then stop,\nand keep the comma.'
    answer = 'Output: {"title": "Example, Study", "year": 2024}'
    row = make_row(
        constraints=constraints,
        state={"agent_final_answer": answer},
    )
    case = load_cases(write_csv(tmp_path, [row])).cases[0]
    assert case.input.constraints == constraints
    assert case.input.agent_final_answer == answer


def test_case_input_has_no_label_fields():
    names = {f.name for f in dataclasses.fields(CaseInput)}
    assert not {n for n in names if n.startswith("gold") or n in ("trap", "rationale")}


# --- Row errors ----------------------------------------------------------


def test_malformed_json_gives_clear_error(tmp_path):
    row = make_row(state_json='{"case_id": "AE-900", "task": ')
    result = load_cases(write_csv(tmp_path, [row]))

    assert result.cases == []
    error = result.errors[0]
    assert error.row == 2
    assert error.case_id == "AE-900"
    messages = [e.message for e in error.errors if e.field == "state_json"]
    assert messages and messages[0].startswith("Malformed JSON in state_json")


def test_duplicate_case_id_rejects_every_copy(tmp_path):
    rows = [make_row("AE-1"), make_row("AE-2"), make_row("AE-1")]
    result = load_cases(write_csv(tmp_path, rows))

    assert [c.input.case_id for c in result.cases] == ["AE-2"]
    assert [e.row for e in result.errors] == [2, 4]
    for error in result.errors:
        message = next(e.message for e in error.errors if e.field == "case_id")
        assert "rows 2, 4" in message


@pytest.mark.parametrize(
    "column, value",
    [
        ("gold_verdict", "MAYBE"),
        ("gold_quality", "5"),
        ("gold_quality", "3.0"),
        ("gold_policy_violation", "yes"),
        ("split", "validation"),
        ("difficulty", "extreme"),
    ],
)
def test_value_outside_allowed_set_is_a_row_error(tmp_path, column, value):
    result = load_cases(write_csv(tmp_path, [make_row(**{column: value})]))
    assert result.cases == []
    assert column in fields_of(result.errors[0])


@pytest.mark.parametrize("name", ["task", "constraints", "agent_final_answer"])
def test_empty_text_field_is_a_row_error(tmp_path, name):
    row = make_row(**{name: ""}, state={name: ""})
    result = load_cases(write_csv(tmp_path, [row]))
    assert name in fields_of(result.errors[0])


def test_empty_case_id_is_a_row_error(tmp_path):
    row = make_row(case_id="", state={"case_id": ""})
    result = load_cases(write_csv(tmp_path, [row]))
    assert result.errors[0].case_id is None
    assert "case_id" in fields_of(result.errors[0])


def test_one_error_record_lists_every_problem(tmp_path):
    row = make_row(gold_verdict="MAYBE", split="other", gold_quality="9")
    error = load_cases(write_csv(tmp_path, [row])).errors[0]
    assert fields_of(error) == {"gold_verdict", "split", "gold_quality"}


def test_bad_row_does_not_stop_valid_rows(tmp_path):
    rows = [make_row("AE-1"), make_row("AE-2", state_json="not json"), make_row("AE-3")]
    result = load_cases(write_csv(tmp_path, rows))
    assert [c.input.case_id for c in result.cases] == ["AE-1", "AE-3"]
    assert [e.case_id for e in result.errors] == ["AE-2"]


def test_row_with_extra_fields_is_a_row_error(tmp_path):
    path = write_csv(tmp_path, [make_row()])
    with path.open("a", newline="", encoding="utf-8") as handle:
        extra = make_row("AE-901")
        values = [extra[c] for c in REQUIRED_COLUMNS] + ["surplus"]
        csv.writer(handle).writerow(values)
    result = load_cases(path)
    assert len(result.cases) == 1
    assert "row" in fields_of(result.errors[0])


# --- File errors ---------------------------------------------------------


def test_missing_file_stops_with_clear_message(tmp_path):
    with pytest.raises(DataFileError, match="Data file not found"):
        load_cases(tmp_path / "absent.csv")


def test_missing_column_stops_with_clear_message(tmp_path):
    columns = [c for c in REQUIRED_COLUMNS if c != "gold_verdict"]
    path = write_csv(tmp_path, [make_row()], columns=columns)
    with pytest.raises(DataFileError, match="missing required columns: gold_verdict"):
        load_cases(path)


# --- Error file ----------------------------------------------------------


def test_error_file_has_reasons_and_complete_record(tmp_path):
    row = make_row(state_json="{broken", gold_verdict="MAYBE")
    result = load_cases(write_csv(tmp_path, [row, make_row("AE-2")]))
    out = tmp_path / "out" / "errors.jsonl"

    write_errors(result.errors, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["row"] == 2
    assert entry["case_id"] == "AE-900"
    assert {e["field"] for e in entry["errors"]} == {"state_json", "gold_verdict"}
    assert all(e["message"] for e in entry["errors"])
    assert entry["record"] == row
