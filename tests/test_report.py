import csv
import json

import pytest

from evaluator.cli import main
from evaluator.models import REQUIRED_COLUMNS
from evaluator.predictions import PredictionsFileError, read_predictions


def make_row(case_id, split="test", verdict="PASS", quality="4", policy="false"):
    state = {
        "case_id": case_id,
        "task": "Summarise the three open support tickets for the team lead.",
        "constraints": "Do not include customer email addresses.",
        "available_tools": ["ticket_list"],
        "agent_trajectory": "The agent listed open tickets and wrote a summary.",
        "agent_final_answer": "Three tickets are open: login, billing and export.",
    }
    return {
        "case_id": case_id,
        "split": split,
        "capability": "summarisation",
        "domain": "support",
        "difficulty": "easy",
        "language": "en",
        "task": "",
        "constraints": "",
        "available_tools": "",
        "agent_trajectory": "",
        "agent_final_answer": "",
        "gold_verdict": verdict,
        "gold_quality": quality,
        "gold_policy_violation": policy,
        "gold_rationale": f"Gold rationale for {case_id} <with markup>.",
        "trap": "none",
        "state_json": json.dumps(state),
    }


@pytest.fixture
def data_csv(tmp_path):
    rows = [
        make_row("T-1"),
        make_row("T-2", verdict="FAIL", quality="1", policy="true"),
        make_row("T-3", verdict="ESCALATE", quality="2"),
        make_row("K-1", split="calibration"),
    ]
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def record(case_id, verdict="PASS", quality=4, policy=False, **extra):
    return {
        "case_id": case_id,
        "verdict": verdict,
        "quality_score": quality,
        "policy_violation": policy,
        "reason": f"Reason for {case_id}.",
        **extra,
    }


def failed_record(case_id):
    return {
        "case_id": case_id,
        "status": "failed",
        "verdict": None,
        "quality_score": None,
        "policy_violation": None,
        "reason": "Evaluator error (timeout): no reply.",
        "error_type": "timeout",
    }


def write_json(tmp_path, data, name="predictions.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --- Reading predictions -------------------------------------------------


def test_status_defaults_to_ok_and_extra_fields_are_ignored(tmp_path):
    path = write_json(tmp_path, [record("T-1", assessment={"x": 1})])
    [prediction] = read_predictions(path, ["T-1"])
    assert prediction.status == "ok"
    assert prediction.verdict == "PASS"


def test_failed_record_is_read_without_labels(tmp_path):
    [prediction] = read_predictions(write_json(tmp_path, [failed_record("T-1")]), ["T-1"])
    assert prediction.status == "failed"
    assert prediction.verdict is None
    assert prediction.error_type == "timeout"


@pytest.mark.parametrize(
    "records, expected",
    [
        ([record("T-1"), record("T-1")], "repeats record 1"),
        ([record("T-1"), record("X-9")], "not in the selected split"),
        ([], "No prediction for case_ids: T-1"),
        ([record("T-1", verdict="MAYBE")], "verdict 'MAYBE'"),
        ([record("T-1", quality=True)], "quality_score True"),
        ([record("T-1", quality=5)], "quality_score 5"),
        ([record("T-1", policy="false")], "policy_violation 'false'"),
        ([{**failed_record("T-1"), "verdict": "PASS"}], "must have null labels"),
        ([record("T-1", status="done")], "status 'done'"),
        ([{"verdict": "PASS"}], "case_id is missing"),
    ],
)
def test_invalid_predictions_give_clear_error(tmp_path, records, expected):
    with pytest.raises(PredictionsFileError, match=expected):
        read_predictions(write_json(tmp_path, records), ["T-1"])


def test_every_problem_is_listed_at_once(tmp_path):
    path = write_json(tmp_path, [record("T-1", verdict="MAYBE", quality=9)])
    with pytest.raises(PredictionsFileError) as info:
        read_predictions(path, ["T-1", "T-2"])
    assert len(info.value.problems) == 3


def test_malformed_json_and_wrong_shape(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[{", encoding="utf-8")
    with pytest.raises(PredictionsFileError, match="Malformed JSON"):
        read_predictions(bad, ["T-1"])
    with pytest.raises(PredictionsFileError, match="JSON array"):
        read_predictions(write_json(tmp_path, {"T-1": {}}), ["T-1"])
    with pytest.raises(PredictionsFileError, match="does not exist"):
        read_predictions(tmp_path / "missing.json", ["T-1"])


# --- The report command --------------------------------------------------


def run_report(data_csv, predictions, output, *extra):
    return main(
        [
            "report",
            "--input", str(data_csv),
            "--predictions", str(predictions),
            "--output", str(output),
            *extra,
        ]
    )


def test_report_writes_html_with_metrics(tmp_path, data_csv, capsys):
    predictions = write_json(
        tmp_path,
        [
            record("T-1"),  # right
            record("T-2", verdict="PASS", quality=3, policy=False),  # wrong
            failed_record("T-3"),
        ],
    )
    explanations = write_json(
        tmp_path, {"T-2": "Missed the <email> rule."}, "explanations.json"
    )
    output = tmp_path / "out" / "report.html"

    code = run_report(data_csv, predictions, output, "--explanations", str(explanations))

    assert code == 0
    assert "3 evaluated, 2 successful, 1 failed" in capsys.readouterr().out
    html = output.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<title>Agent Evaluator Report</title>" in html
    assert "50.0%" in html  # verdict accuracy, 1 of 2 scored
    assert "33.3%" in html  # strict accuracy, 1 of 3
    assert "Failed records" in html and "timeout" in html
    assert "Predicted count" in html  # verdict distribution next to the gold counts
    # Text from the data and the notes is escaped, never injected as markup.
    assert "Missed the &lt;email&gt; rule." in html
    assert "&lt;with markup&gt;" in html
    assert "<email>" not in html


def test_report_scores_only_the_chosen_split(tmp_path, data_csv):
    predictions = write_json(tmp_path, [record("K-1")])
    output = tmp_path / "report.html"
    assert run_report(data_csv, predictions, output, "--split", "calibration") == 0
    assert "calibration split" in output.read_text(encoding="utf-8")


def test_report_fails_clearly_on_bad_predictions(tmp_path, data_csv, capsys):
    predictions = write_json(tmp_path, [record("T-1")])
    output = tmp_path / "report.html"
    assert run_report(data_csv, predictions, output) == 1
    error = capsys.readouterr().err
    assert "cannot be scored" in error
    assert "No prediction for case_ids: T-2, T-3" in error
    assert not output.exists()


def test_explanation_for_a_correct_case_is_rejected(tmp_path, data_csv, capsys):
    predictions = write_json(
        tmp_path,
        [record("T-1"), record("T-2", verdict="FAIL", quality=1, policy=True),
         record("T-3", verdict="ESCALATE", quality=2)],
    )
    explanations = write_json(tmp_path, {"T-1": "Fine."}, "explanations.json")
    code = run_report(
        data_csv, predictions, tmp_path / "r.html", "--explanations", str(explanations)
    )
    assert code == 1
    assert "not error examples: T-1" in capsys.readouterr().err


def test_missing_data_file_fails_clearly(tmp_path, capsys):
    predictions = write_json(tmp_path, [])
    assert run_report(tmp_path / "none.csv", predictions, tmp_path / "r.html") == 1
    assert "Data file not found" in capsys.readouterr().err
