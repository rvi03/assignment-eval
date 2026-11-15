"""Record types shared by the loader, the evaluator and the report.

Each CSV row is split into two records on purpose:

- CaseInput holds only what the evaluator may see.
- GoldLabels holds gold_*, gold_rationale and trap, used only for scoring.

The evaluator receives CaseInput alone, so labels cannot leak into a
prediction by accident.
"""

from dataclasses import dataclass, field

VERDICTS = ("PASS", "FAIL", "ESCALATE")
QUALITY_SCORES = (0, 1, 2, 3, 4)
SPLITS = ("calibration", "test")
DIFFICULTIES = ("easy", "medium", "hard")

INPUT_COLUMNS = (
    "case_id",
    "split",
    "capability",
    "domain",
    "difficulty",
    "language",
    "task",
    "constraints",
    "available_tools",
    "agent_trajectory",
    "agent_final_answer",
    "state_json",
)
LABEL_COLUMNS = (
    "gold_verdict",
    "gold_quality",
    "gold_policy_violation",
    "gold_rationale",
    "trap",
)
REQUIRED_COLUMNS = INPUT_COLUMNS + LABEL_COLUMNS


@dataclass(frozen=True)
class CaseInput:
    """Everything the evaluator may use to predict a verdict."""

    case_id: str
    split: str
    capability: str
    domain: str
    difficulty: str
    language: str
    task: str
    constraints: str
    available_tools: tuple[str, ...]
    agent_trajectory: str
    agent_final_answer: str


@dataclass(frozen=True)
class GoldLabels:
    """Expected outcome for one case. Never passed to the evaluator."""

    case_id: str
    verdict: str
    quality: int
    policy_violation: bool
    rationale: str
    trap: str


@dataclass(frozen=True)
class Case:
    """One valid row: the evaluator input and its labels, kept apart."""

    input: CaseInput
    labels: GoldLabels


@dataclass(frozen=True)
class FieldError:
    field: str
    message: str


@dataclass(frozen=True)
class RowError:
    """One invalid row, with every problem found and the raw record.

    `row` is the row number in the file, counting the header as row 1,
    so it matches the row number a spreadsheet shows.
    """

    row: int
    case_id: str | None
    errors: tuple[FieldError, ...]
    record: dict


@dataclass
class LoadResult:
    cases: list[Case] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)


@dataclass(frozen=True)
class Prediction:
    """The evaluator's result for one case.

    A successful prediction has status "ok" and all four labels. A failed
    one has status "failed", no labels, the error in `reason` and its kind
    in `error_type`; it is counted as failed instead of being given a
    verdict nobody decided.
    """

    case_id: str
    status: str
    verdict: str | None
    quality_score: int | None
    policy_violation: bool | None
    reason: str
    error_type: str | None = None
    format_check: bool | None = None
    unverified_quotes: int = 0
    assessment: dict | None = None


class DataFileError(Exception):
    """The file as a whole cannot be read, so no row can be trusted."""
