"""Command-line entry point for the evaluator."""

import argparse
import sys
import webbrowser
from collections import Counter
from dataclasses import replace
from pathlib import Path

from evaluator import batch
from evaluator.explain import generate_explanations
from evaluator.llm import ConfigError, OpenAIJudgeClient, Settings
from evaluator.loader import DEFAULT_DATA_PATH, load_cases, write_errors
from evaluator.models import SPLITS, DataFileError
from evaluator.predictions import PredictionsFileError
from evaluator.report import ReportError, ReportInputs, write_report

DEFAULT_PREDICTIONS_PATH = Path("outputs/predictions.json")
DEFAULT_REPORT_PATH = Path("outputs/report.html")
DEFAULT_ERRORS_PATH = Path("outputs/validation_errors.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluator",
        description="Evaluate AI agent responses and report metrics.",
    )
    commands = parser.add_subparsers(dest="command", metavar="command")

    validate = commands.add_parser(
        "validate",
        help="Load and check every row of a CSV; no API key needed.",
        description="Load and check every row of a CSV. Prints the counts and "
        "every row error, writes the errors as JSON Lines, and exits with 1 if "
        "any row is invalid. No API key is needed.",
    )
    validate.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Benchmark CSV (default: {DEFAULT_DATA_PATH}).",
    )
    validate.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_ERRORS_PATH,
        help=f"Where to write row errors as JSON Lines (default: {DEFAULT_ERRORS_PATH}).",
    )

    run = commands.add_parser(
        "run",
        help="Evaluate one split with the judge model and write predictions.",
        description="Evaluate every valid case of one split with the judge "
        "model and write the predictions and a run record next to them. "
        "Then it scores them against the gold labels, writes the HTML report "
        "and opens it in a browser when run in a terminal. Exits with 1 if "
        "any prediction failed. Needs the judge settings (see README, Judge "
        "settings).",
    )
    run.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Benchmark CSV (default: {DEFAULT_DATA_PATH}).",
    )
    run.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="Which split to evaluate (default: test).",
    )
    run.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help=f"Where to write the predictions (default: {DEFAULT_PREDICTIONS_PATH}); "
        "the run record is written next to it as <name>.run.json.",
    )
    run.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Where to write the HTML report at the end of the run (default: {DEFAULT_REPORT_PATH}).",
    )
    run.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip the model-written explanations of wrong predictions; the "
        "report still names the rule that decided each one.",
    )
    run.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the report in a browser. It is opened only when the "
        "command runs in a terminal.",
    )

    report = commands.add_parser(
        "report",
        help="Compare predictions with gold labels and write an HTML report.",
        description="Compare predictions with gold labels and write an HTML report.",
    )
    report.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Benchmark CSV with gold labels (default: {DEFAULT_DATA_PATH}).",
    )
    report.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help=f"Predictions JSON file (default: {DEFAULT_PREDICTIONS_PATH}).",
    )
    report.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Where to write the HTML report (default: {DEFAULT_REPORT_PATH}).",
    )
    report.add_argument(
        "--split",
        choices=SPLITS,
        default="test",
        help="Which split to score (default: test).",
    )
    report.add_argument(
        "--explanations",
        type=Path,
        default=None,
        help="Optional JSON object of case_id to a short note on why that "
        "prediction is wrong; shown with the error examples.",
    )
    report.add_argument(
        "--generated-explanations",
        type=Path,
        default=None,
        help="Optional file of model-written error explanations, as written "
        "by the run command (<name>.explanations.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "validate":
        return _validate(args)
    if args.command == "run":
        return _run(args)
    if args.command == "report":
        return _report(args)
    return 0


def _validate(args: argparse.Namespace) -> int:
    try:
        result = load_cases(args.input)
    except DataFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    splits = Counter(case.input.split for case in result.cases)
    total = len(result.cases) + len(result.errors)
    split_text = ", ".join(f"{splits[name]} {name}" for name in SPLITS)
    print(
        f"Read {total} rows from {args.input}: {len(result.cases)} valid "
        f"({split_text}), {len(result.errors)} invalid."
    )
    if not result.errors:
        return 0
    for error in result.errors:
        problems = "; ".join(f"{e.field}: {e.message}" for e in error.errors)
        print(f"  Row {error.row} ({error.case_id or 'no case_id'}): {problems}")
    write_errors(result.errors, args.errors)
    print(f"Wrote {len(result.errors)} row error(s) to {args.errors}.")
    return 1


def _run(args: argparse.Namespace) -> int:
    try:
        settings = Settings.from_env()
        result = load_cases(args.input)
    except (ConfigError, DataFileError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    skipped = [e for e in result.errors if e.record.get("split") == args.split]
    if skipped:
        print(
            f"Warning: {len(skipped)} invalid {args.split} row(s) are not evaluated; "
            "run validate to see why.",
            file=sys.stderr,
        )

    def progress(index: int, total: int, prediction) -> None:
        label = prediction.verdict if prediction.status == "ok" else f"failed ({prediction.error_type})"
        print(f"[{index}/{total}] {prediction.case_id}: {label}", flush=True)

    client = OpenAIJudgeClient(settings)
    started_at = batch.now()
    predictions = batch.evaluate_split(result.cases, args.split, client, on_result=progress)
    finished_at = batch.now()

    batch.write_predictions(predictions, args.output)
    info_path = batch.run_info_path(args.output)
    batch.write_run_info(
        info_path,
        settings=settings,
        split=args.split,
        input_path=args.input,
        started_at=started_at,
        finished_at=finished_at,
        predictions=predictions,
    )
    summary = batch.summarize(predictions)
    print(
        f"Wrote {summary.total} predictions to {args.output} ({summary.ok} ok, "
        f"{summary.failed} failed) and the run record to {info_path}."
    )

    split_cases = [c for c in result.cases if c.input.split == args.split]
    notes_path = batch.explanations_path(args.output)
    if notes_path.exists():
        notes_path.unlink()  # never pair a new run with old explanations
    inputs = ReportInputs(data_path=args.input, predictions_path=args.output, split=args.split)
    try:
        # The report is written before the slower explanation step, so a
        # slow or failing model can never block the main deliverable.
        metrics = write_report(inputs, args.report)
        if not args.no_explain and metrics.mismatches:
            print(
                f"Explaining {len(metrics.mismatches)} wrong prediction(s) with the judge model.",
                flush=True,
            )
            by_id = {p.case_id: p for p in predictions}
            mismatches = [replace(m, prediction=by_id[m.prediction.case_id]) for m in metrics.mismatches]
            notes = generate_explanations(
                client, {c.input.case_id: c.input for c in split_cases}, mismatches
            )
            batch.write_explanations(notes_path, settings.model, notes)
            inputs = replace(inputs, generated_path=notes_path)
            metrics = write_report(inputs, args.report)
    except (DataFileError, PredictionsFileError, ReportError) as exc:
        print(f"Error: the report could not be written: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote the {args.split} report to {args.report}: verdict accuracy "
        f"{_percent(metrics.verdict_accuracy)}, macro F1 {_decimal(metrics.macro_f1)}."
    )
    if not args.no_open and sys.stdout.isatty():
        webbrowser.open(args.report.resolve().as_uri())
    return 1 if summary.failed or skipped else 0


def _saved_explanations(predictions: Path) -> Path | None:
    """The explanations file run writes next to the predictions, if present."""
    path = batch.explanations_path(predictions)
    return path if path.exists() else None


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _report(args: argparse.Namespace) -> int:
    inputs = ReportInputs(
        data_path=args.input,
        predictions_path=args.predictions,
        split=args.split,
        explanations_path=args.explanations,
        generated_path=args.generated_explanations or _saved_explanations(args.predictions),
    )
    try:
        metrics = write_report(inputs, args.output)
    except (DataFileError, PredictionsFileError, ReportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote the {args.split} report to {args.output}: "
        f"{metrics.total} evaluated, {metrics.successful} successful, "
        f"{metrics.failed} failed."
    )
    return 0
