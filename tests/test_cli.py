"""Command tests: validate and run use invented rows and a fake judge client."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluator import cli
from evaluator.llm import Reply
from evaluator.models import REQUIRED_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS = {
    "OPENAI_API_KEY": "secret-test-key",
    "EVALUATOR_MODEL": "fake-model",
    "OPENAI_BASE_URL": "http://localhost:9/v1",
}


def row(case_id: str, split: str = "test", **overrides) -> dict:
    data = {
        "case_id": case_id,
        "split": split,
        "capability": "instruction_following",
        "domain": "library",
        "difficulty": "easy",
        "language": "en",
        "task": "Renew the member's borrowed book for two weeks.",
        "constraints": "A book may be renewed at most twice.",
        "available_tools": "loan_lookup, loan_renew",
        "agent_trajectory": "The agent saw one earlier renewal and renewed the loan.",
        "agent_final_answer": "Your loan is renewed until the 30th.",
        "gold_verdict": "PASS",
        "gold_quality": "4",
        "gold_policy_violation": "false",
        "gold_rationale": "Renewed within the limit.",
        "trap": "none",
    }
    data["state_json"] = json.dumps({"case_id": case_id})
    data.update(overrides)
    return data


def write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def judge_reply(**overrides) -> dict:
    data = {
        "evidence": [
            {
                "question": "Q1",
                "source": "agent_final_answer",
                "quote": "Your loan is renewed",
                "note": "The loan was renewed.",
            }
        ],
        "assessable": "yes",
        "task_completed": "yes",
        "minor_flaw": "no",
        "policy_breach": "no",
        "tools_ok": "yes",
        "needs_human": "no",
        "agent_deferred": "no",
        "reason": 'The agent renewed within the limit: "Your loan is renewed".',
    }
    data.update(overrides)
    return data


class FakeClient:
    """Answers every request with the next queued reply (or the last one)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = 0

    def complete(self, messages, schema, max_tokens=None):
        self.requests += 1
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return Reply(content=content, finish_reason="stop")


@pytest.fixture
def judge(monkeypatch):
    """Install a fake judge for `run` and return a setter for its replies."""
    holder = {}

    def install(*replies):
        holder["client"] = FakeClient(*replies)
        monkeypatch.setattr(cli, "OpenAIJudgeClient", lambda settings: holder["client"])
        return holder["client"]

    for name, value in SETTINGS.items():
        monkeypatch.setenv(name, value)
    return install


def test_module_help_prints_usage():
    result = subprocess.run(
        [sys.executable, "-m", "evaluator", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage: python -m evaluator" in result.stdout


# --- validate -------------------------------------------------------------


def test_validate_reports_counts_and_exits_zero_on_a_clean_file(tmp_path, capsys, monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    data = write_csv(tmp_path / "data.csv", [row("LB-1", "calibration"), row("LB-2"), row("LB-3")])
    assert cli.main(["validate", "--input", str(data), "--errors", str(tmp_path / "e.jsonl")]) == 0
    assert "3 valid (1 calibration, 2 test), 0 invalid" in capsys.readouterr().out
    assert not (tmp_path / "e.jsonl").exists()


def test_validate_lists_errors_writes_them_and_exits_one(tmp_path, capsys):
    data = write_csv(tmp_path / "data.csv", [row("LB-1"), row("LB-2", gold_quality="7")])
    errors = tmp_path / "e.jsonl"
    assert cli.main(["validate", "--input", str(data), "--errors", str(errors)]) == 1
    out = capsys.readouterr().out
    assert "1 valid" in out and "1 invalid" in out and "LB-2" in out
    assert json.loads(errors.read_text().splitlines()[0])["case_id"] == "LB-2"


def test_validate_missing_file_is_a_clear_error(tmp_path, capsys):
    assert cli.main(["validate", "--input", str(tmp_path / "nope.csv")]) == 1
    assert "Error:" in capsys.readouterr().err


# --- run ------------------------------------------------------------------


def test_batch_output_preserves_the_input_case_id_with_one_record_per_row(tmp_path, judge):
    data = write_csv(tmp_path / "data.csv", [row("LB-1", "calibration"), row("LB-2"), row("LB-3")])
    output = tmp_path / "out" / "predictions.json"
    client = judge(judge_reply())

    assert cli.main(["run", "--input", str(data), "--split", "test", "--output", str(output), "--report", str(tmp_path / "r.html")]) == 0

    records = json.loads(output.read_text())
    assert [r["case_id"] for r in records] == ["LB-2", "LB-3"]
    assert records[0]["verdict"] == "PASS" and records[0]["quality_score"] == 4
    assert list(records[0])[:6] == [
        "case_id", "status", "verdict", "quality_score", "policy_violation", "reason",
    ]
    assert client.requests == 2

    info_text = (tmp_path / "out" / "predictions.run.json").read_text()
    info = json.loads(info_text)
    assert info["counts"] == {
        "total": 2, "ok": 2, "failed": 0, "failed_by_type": {}, "unverified_quotes": 0,
    }
    assert info["settings"]["model"] == "fake-model"
    assert len(info["input"]["sha256"]) == 64 and info["prompt_fingerprint"]
    assert "secret-test-key" not in info_text and "secret-test-key" not in output.read_text()


def test_run_writes_the_report_and_opens_it_only_in_a_terminal(tmp_path, judge, monkeypatch):
    data = write_csv(tmp_path / "data.csv", [row("LB-1"), row("LB-2")])
    output, report = tmp_path / "predictions.json", tmp_path / "report.html"
    judge(judge_reply())
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    args = ["run", "--input", str(data), "--output", str(output), "--report", str(report)]

    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)
    assert cli.main(args) == 0
    assert report.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert opened == []

    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    assert cli.main(args) == 0
    assert opened == [report.resolve().as_uri()]

    assert cli.main([*args, "--no-open"]) == 0
    assert len(opened) == 1


def test_run_output_is_accepted_by_report(tmp_path, judge):
    data = write_csv(tmp_path / "data.csv", [row("LB-1"), row("LB-2")])
    output = tmp_path / "predictions.json"
    judge(judge_reply())
    assert cli.main(["run", "--input", str(data), "--output", str(output), "--report", str(tmp_path / "r.html")]) == 0
    report = tmp_path / "report.html"
    assert cli.main(["report", "--input", str(data), "--predictions", str(output), "--output", str(report)]) == 0
    assert report.exists()


def test_run_exits_one_when_any_prediction_failed(tmp_path, judge):
    data = write_csv(tmp_path / "data.csv", [row("LB-1"), row("LB-2")])
    output = tmp_path / "predictions.json"
    judge(judge_reply(), "not json", "still not json")
    assert cli.main(["run", "--input", str(data), "--output", str(output), "--report", str(tmp_path / "r.html")]) == 1
    records = json.loads(output.read_text())
    assert [r["status"] for r in records] == ["ok", "failed"]
    assert records[1]["verdict"] is None and records[1]["error_type"] == "invalid_reply"
    info = json.loads((tmp_path / "predictions.run.json").read_text())
    assert info["counts"]["failed_by_type"] == {"invalid_reply": 1}


def test_run_without_settings_stops_with_a_clear_message(tmp_path, capsys, monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    data = write_csv(tmp_path / "data.csv", [row("LB-1")])
    output = tmp_path / "predictions.json"
    assert cli.main(["run", "--input", str(data), "--output", str(output), "--report", str(tmp_path / "r.html")]) == 1
    assert "Missing environment variable(s)" in capsys.readouterr().err
    assert not output.exists()


def test_run_warns_and_exits_one_when_split_rows_are_invalid(tmp_path, capsys, judge):
    data = write_csv(tmp_path / "data.csv", [row("LB-1"), row("LB-2", gold_quality="7")])
    output = tmp_path / "predictions.json"
    judge(judge_reply())
    assert cli.main(["run", "--input", str(data), "--output", str(output), "--report", str(tmp_path / "r.html")]) == 1
    assert "1 invalid test row(s) are not evaluated" in capsys.readouterr().err
    assert [r["case_id"] for r in json.loads(output.read_text())] == ["LB-1"]
