"""Metric tests against values worked out by hand.

The shared example has six scored cases and one failed case:

    case  gold verdict  predicted   gold policy  predicted  gold q  predicted q
    C1    PASS          PASS        yes          yes        4       4
    C2    PASS          FAIL        no           yes        3       2
    C3    FAIL          FAIL        yes          no         1       1
    C4    FAIL          FAIL        no           no         0       1
    C5    ESCALATE      FAIL        no           no         2       1
    C6    FAIL          PASS        no           no         1       3
    C7    PASS          (failed)
"""

import pytest

from evaluator.metrics import compute_metrics
from evaluator.models import GoldLabels, Prediction


def gold(case_id, verdict, quality, policy):
    return GoldLabels(case_id, verdict, quality, policy, f"Rationale {case_id}.", "")


def ok(case_id, verdict, quality, policy):
    return Prediction(case_id, "ok", verdict, quality, policy, f"Reason {case_id}.")


def failed(case_id):
    return Prediction(case_id, "failed", None, None, None, "Timeout.", "timeout")


GOLD = {
    g.case_id: g
    for g in [
        gold("C1", "PASS", 4, True),
        gold("C2", "PASS", 3, False),
        gold("C3", "FAIL", 1, True),
        gold("C4", "FAIL", 0, False),
        gold("C5", "ESCALATE", 2, False),
        gold("C6", "FAIL", 1, False),
        gold("C7", "PASS", 4, False),
    ]
}
PREDICTIONS = [
    ok("C1", "PASS", 4, True),
    ok("C2", "FAIL", 2, True),
    ok("C3", "FAIL", 1, False),
    ok("C4", "FAIL", 1, False),
    ok("C5", "FAIL", 1, False),
    ok("C6", "PASS", 3, False),
    failed("C7"),
]


@pytest.fixture
def metrics():
    return compute_metrics(PREDICTIONS, GOLD)


def test_counts(metrics):
    assert (metrics.total, metrics.successful, metrics.failed) == (7, 6, 1)
    assert [f.case_id for f in metrics.failures] == ["C7"]


def test_verdict_accuracy_uses_scored_cases_only(metrics):
    # C1, C3 and C4 are right: 3 of 6 scored.
    assert metrics.verdict_accuracy == pytest.approx(3 / 6)


def test_strict_accuracy_counts_failed_as_wrong(metrics):
    assert metrics.strict_verdict_accuracy == pytest.approx(3 / 7)


def test_per_class_scores(metrics):
    # PASS: predicted for C1, C6; gold for C1, C2.
    assert metrics.per_class["PASS"].precision == pytest.approx(1 / 2)
    assert metrics.per_class["PASS"].recall == pytest.approx(1 / 2)
    assert metrics.per_class["PASS"].f1 == pytest.approx(1 / 2)
    # FAIL: predicted for C2..C5 (2 right); gold for C3, C4, C6.
    assert metrics.per_class["FAIL"].precision == pytest.approx(2 / 4)
    assert metrics.per_class["FAIL"].recall == pytest.approx(2 / 3)
    assert metrics.per_class["FAIL"].f1 == pytest.approx(4 / 7)
    # ESCALATE is never predicted: precision is undefined, recall is 0.
    assert metrics.per_class["ESCALATE"].precision is None
    assert metrics.per_class["ESCALATE"].recall == 0
    assert metrics.per_class["ESCALATE"].f1 == 0
    assert [s.support for s in metrics.per_class.values()] == [2, 3, 1]


def test_macro_f1_is_plain_mean_over_three_classes(metrics):
    assert metrics.macro_f1 == pytest.approx((1 / 2 + 4 / 7 + 0) / 3)


def test_confusion_matrix_rows_are_gold(metrics):
    assert metrics.confusion == [
        [1, 1, 0],  # gold PASS: C1 right, C2 called FAIL
        [1, 2, 0],  # gold FAIL: C6 called PASS, C3 and C4 right
        [0, 1, 0],  # gold ESCALATE: C5 called FAIL
    ]


def test_policy_precision_and_recall(metrics):
    assert metrics.policy_counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 3}
    assert metrics.policy_precision == pytest.approx(1 / 2)
    assert metrics.policy_recall == pytest.approx(1 / 2)


def test_quality_mae(metrics):
    # Absolute errors 0, 1, 0, 1, 1, 2.
    assert metrics.quality_mae == pytest.approx(5 / 6)


def test_quality_baseline_predicts_the_median_gold_score(metrics):
    # Scored gold scores 0, 1, 1, 2, 3, 4: the lower median is 1, and
    # always predicting 1 gives distances 3, 2, 0, 1, 1, 0.
    assert metrics.quality_baseline_score == 1
    assert metrics.quality_baseline_mae == pytest.approx(7 / 6)


def test_mismatches_list_verdict_errors_first(metrics):
    assert [m.prediction.case_id for m in metrics.mismatches] == ["C2", "C5", "C6", "C3"]
    c3 = metrics.mismatches[-1]
    assert not c3.verdict_wrong and c3.policy_wrong


def test_policy_metrics_are_none_when_denominator_is_zero():
    gold_labels = {"A": gold("A", "PASS", 4, False), "B": gold("B", "FAIL", 1, False)}
    predictions = [ok("A", "PASS", 4, False), ok("B", "FAIL", 1, False)]
    metrics = compute_metrics(predictions, gold_labels)
    assert metrics.policy_precision is None  # nothing predicted as a violation
    assert metrics.policy_recall is None  # no gold violation exists


def test_all_failed_gives_no_label_metrics():
    metrics = compute_metrics([failed("C7")], GOLD)
    assert (metrics.total, metrics.successful, metrics.failed) == (1, 0, 1)
    assert metrics.verdict_accuracy is None
    assert metrics.macro_f1 is None
    assert metrics.quality_mae is None
    assert metrics.quality_baseline_mae is None
    assert metrics.confusion == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
