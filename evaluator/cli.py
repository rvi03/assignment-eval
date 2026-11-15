"""Command-line entry point for the evaluator."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="python -m evaluator",
        description="Evaluate AI agent responses and report metrics.",
    )


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
