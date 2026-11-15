"""Score predictions against gold labels.

Rules that decide how the numbers are read:

- Label metrics use successful predictions only. A failed prediction has
  no labels, and inventing one would bias every metric.
- Strict accuracy counts each failed prediction as wrong, so it is the
  lower bound when failures are present.
- A metric whose denominator is zero (for example, precision when no
  policy violation was predicted) is None and shown as "n/a", never 0.
- Macro F1 is the unweighted mean of the F1 for PASS, FAIL and ESCALATE.
  A class that appears in neither the gold labels nor the predictions
  scores 0 there, which lowers the mean; the report shows per-class F1
  so this stays visible.
- The quality baseline always predicts one score: the median gold score
  of the scored cases. No constant has a lower MAE, so an evaluator that
  does not beat it has learned nothing about quality. The baseline reads
  the gold labels; it is a yardstick, not a prediction.
"""

import math
import statistics
from dataclasses import dataclass, field

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from evaluator.models import VERDICTS, GoldLabels, Prediction


@dataclass(frozen=True)
class ClassScores:
    precision: float | None
    recall: float | None
    f1: float | None
    support: int


@dataclass(frozen=True)
class Mismatch:
    """A successful prediction that disagrees with gold on verdict or policy."""

    prediction: Prediction
    gold: GoldLabels

    @property
    def verdict_wrong(self) -> bool:
        return self.prediction.verdict != self.gold.verdict

    @property
    def policy_wrong(self) -> bool:
        return self.prediction.policy_violation != self.gold.policy_violation


@dataclass
class Metrics:
    total: int
    successful: int
    failed: int
    verdict_accuracy: float | None = None
    strict_verdict_accuracy: float | None = None
    macro_f1: float | None = None
    per_class: dict[str, ClassScores] = field(default_factory=dict)
    # Rows are gold verdicts and columns are predicted verdicts, both in
    # VERDICTS order.
    confusion: list[list[int]] = field(default_factory=list)
    policy_precision: float | None = None
    policy_recall: float | None = None
    # True/false positives and negatives, with "violation" as the positive class.
    policy_counts: dict[str, int] = field(default_factory=dict)
    quality_mae: float | None = None
    quality_baseline_score: int | None = None
    quality_baseline_mae: float | None = None
    mismatches: list[Mismatch] = field(default_factory=list)
    failures: list[Prediction] = field(default_factory=list)


def compute_metrics(
    predictions: list[Prediction], gold: dict[str, GoldLabels]
) -> Metrics:
    """Compute every report metric. `gold` maps case_id to its labels."""
    ok = [p for p in predictions if p.status == "ok"]
    failures = [p for p in predictions if p.status != "ok"]
    metrics = Metrics(
        total=len(predictions),
        successful=len(ok),
        failed=len(failures),
        failures=failures,
        confusion=[[0] * len(VERDICTS) for _ in VERDICTS],
        policy_counts={"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    )
    if not ok:
        return metrics

    labels = [gold[p.case_id] for p in ok]
    true_verdicts = [g.verdict for g in labels]
    pred_verdicts = [p.verdict for p in ok]

    metrics.verdict_accuracy = float(accuracy_score(true_verdicts, pred_verdicts))
    correct = sum(t == p for t, p in zip(true_verdicts, pred_verdicts))
    metrics.strict_verdict_accuracy = correct / len(predictions)
    metrics.macro_f1 = float(
        f1_score(
            true_verdicts,
            pred_verdicts,
            labels=list(VERDICTS),
            average="macro",
            zero_division=0,
        )
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        true_verdicts,
        pred_verdicts,
        labels=list(VERDICTS),
        zero_division=math.nan,
    )
    metrics.per_class = {
        verdict: ClassScores(
            precision=_defined(precision[i]),
            recall=_defined(recall[i]),
            f1=_defined(f1[i]),
            support=int(support[i]),
        )
        for i, verdict in enumerate(VERDICTS)
    }
    metrics.confusion = confusion_matrix(
        true_verdicts, pred_verdicts, labels=list(VERDICTS)
    ).tolist()

    true_policy = [g.policy_violation for g in labels]
    pred_policy = [p.policy_violation for p in ok]
    counts = {
        "tp": sum(t and p for t, p in zip(true_policy, pred_policy)),
        "fp": sum(p and not t for t, p in zip(true_policy, pred_policy)),
        "fn": sum(t and not p for t, p in zip(true_policy, pred_policy)),
        "tn": sum(not t and not p for t, p in zip(true_policy, pred_policy)),
    }
    metrics.policy_counts = counts
    metrics.policy_precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    metrics.policy_recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])

    true_quality = [g.quality for g in labels]
    metrics.quality_mae = float(
        mean_absolute_error(true_quality, [p.quality_score for p in ok])
    )
    # With an even count, every score between the two middle values gives
    # the same MAE; the lower one keeps the baseline a whole score.
    baseline = statistics.median_low(true_quality)
    metrics.quality_baseline_score = baseline
    metrics.quality_baseline_mae = float(
        mean_absolute_error(true_quality, [baseline] * len(true_quality))
    )

    mismatches = [Mismatch(p, gold[p.case_id]) for p in ok]
    mismatches = [m for m in mismatches if m.verdict_wrong or m.policy_wrong]
    # Verdict errors first, because the verdict is the main decision.
    mismatches.sort(key=lambda m: (not m.verdict_wrong, m.prediction.case_id))
    metrics.mismatches = mismatches
    return metrics


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _defined(value) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value
