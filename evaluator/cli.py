"""Command-line entry point for the evaluator."""

import argparse
import sys
from pathlib import Path

from evaluator.loader import DEFAULT_DATA_PATH
from evaluator.models import SPLITS, DataFileError
from evaluator.predictions import PredictionsFileError
from evaluator.report import ReportError, ReportInputs, write_report

DEFAULT_PREDICTIONS_PATH = Path("outputs/predictions.json")
DEFAULT_REPORT_PATH = Path("outputs/report.html")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluator",
        description="Evaluate AI agent responses and report metrics.",
    )
    commands = parser.add_subparsers(dest="command", metavar="command")

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "report":
        return _report(args)
    return 0


def _report(args: argparse.Namespace) -> int:
    inputs = ReportInputs(
        data_path=args.input,
        predictions_path=args.predictions,
        split=args.split,
        explanations_path=args.explanations,
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
