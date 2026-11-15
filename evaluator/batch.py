"""Evaluate every case of one split and write the predictions and a run record.

The predictions file holds one record per case, in the shape the report
reads. The run record sits next to it and says how the predictions were
made: the settings (never the key), the prompt fingerprint, the input
file's hash, the times and the counts. Together they make a run
reproducible and auditable.
"""

import hashlib
import json
import platform
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from evaluator.evaluate import evaluate_case
from evaluator.judge import prompt_fingerprint
from evaluator.llm import JudgeClient, Settings
from evaluator.models import Case, Prediction


@dataclass(frozen=True)
class RunSummary:
    total: int
    ok: int
    failed: int


def evaluate_split(
    cases: list[Case],
    split: str,
    client: JudgeClient,
    on_result: Callable[[int, int, Prediction], None] | None = None,
) -> list[Prediction]:
    """Evaluate every case in `split`, in file order.

    Only each case's input reaches the evaluator; its gold labels stay here.
    `on_result` is called after each case with (index, total, prediction).
    """
    selected = [case for case in cases if case.input.split == split]
    predictions = []
    for index, case in enumerate(selected, start=1):
        prediction = evaluate_case(case.input, client)
        predictions.append(prediction)
        if on_result is not None:
            on_result(index, len(selected), prediction)
    return predictions


def prediction_record(prediction: Prediction) -> dict:
    """The output record: the brief's fields first, then diagnostics."""
    record = {
        "case_id": prediction.case_id,
        "status": prediction.status,
        "verdict": prediction.verdict,
        "quality_score": prediction.quality_score,
        "policy_violation": prediction.policy_violation,
        "reason": prediction.reason,
    }
    if prediction.error_type is not None:
        record["error_type"] = prediction.error_type
    record["format_check"] = prediction.format_check
    record["unverified_quotes"] = prediction.unverified_quotes
    record["assessment"] = prediction.assessment
    return record


def write_predictions(predictions: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [prediction_record(p) for p in predictions]
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_info_path(predictions_path: Path) -> Path:
    """Where the run record goes: next to the predictions, e.g. predictions.run.json."""
    return predictions_path.with_name(predictions_path.stem + ".run.json")


def explanations_path(predictions_path: Path) -> Path:
    """Where generated error explanations go, e.g. predictions.explanations.json."""
    return predictions_path.with_name(predictions_path.stem + ".explanations.json")


def write_explanations(path: Path, generator: str, explanations: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"generator": generator, "explanations": explanations}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(predictions: list[Prediction]) -> RunSummary:
    failed = sum(1 for p in predictions if p.status == "failed")
    return RunSummary(total=len(predictions), ok=len(predictions) - failed, failed=failed)


def write_run_info(
    path: Path,
    *,
    settings: Settings,
    split: str,
    input_path: Path,
    started_at: datetime,
    finished_at: datetime,
    predictions: list[Prediction],
) -> None:
    summary = summarize(predictions)
    info = {
        "split": split,
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "settings": settings.describe(),
        "prompt_fingerprint": prompt_fingerprint(),
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
        "seconds": round((finished_at - started_at).total_seconds(), 1),
        "counts": {
            "total": summary.total,
            "ok": summary.ok,
            "failed": summary.failed,
            "failed_by_type": dict(
                Counter(p.error_type for p in predictions if p.status == "failed")
            ),
            "unverified_quotes": sum(p.unverified_quotes for p in predictions),
        },
        "python": platform.python_version(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
