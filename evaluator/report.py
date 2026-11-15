"""Build the evaluation report: load gold labels, score, render HTML.

The HTML is one self-contained file (inline CSS, system fonts, no
scripts), so it opens offline in any browser and follows the reader's
light or dark theme.
"""

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

from evaluator.explain import code_explanation
from evaluator.loader import load_cases
from evaluator.metrics import Metrics, Mismatch, compute_metrics
from evaluator.models import VERDICTS, Prediction
from evaluator.predictions import read_predictions

MIN_ERROR_EXAMPLES = 3


class ReportError(Exception):
    """The report cannot be built from the given inputs."""


@dataclass
class ReportInputs:
    data_path: Path
    predictions_path: Path
    split: str
    explanations_path: Path | None = None
    generated_path: Path | None = None


@dataclass(frozen=True)
class Notes:
    """Text shown with the error examples, besides the code's own facts."""

    analyst: dict[str, str]
    generated: dict[str, str]
    generator: str | None = None


def build_report(inputs: ReportInputs) -> tuple[Metrics, str]:
    """Score the predictions and return the metrics and the report HTML."""
    loaded = load_cases(inputs.data_path)
    if loaded.errors:
        rows = ", ".join(str(e.row) for e in loaded.errors[:10])
        more = " and more" if len(loaded.errors) > 10 else ""
        raise ReportError(
            f"{len(loaded.errors)} rows in {inputs.data_path} are invalid "
            f"(rows {rows}{more}), so their gold labels cannot be trusted. "
            "Run the validate command to see each problem."
        )
    cases = [c for c in loaded.cases if c.input.split == inputs.split]
    if not cases:
        raise ReportError(f"{inputs.data_path} has no rows in split {inputs.split!r}.")

    predictions = read_predictions(
        inputs.predictions_path, [c.input.case_id for c in cases]
    )
    gold = {c.input.case_id: c.labels for c in cases}
    metrics = compute_metrics(predictions, gold)
    generated, generator = _read_generated(inputs.generated_path, metrics)
    notes = Notes(
        analyst=_read_explanations(inputs.explanations_path, metrics),
        generated=generated,
        generator=generator,
    )
    return metrics, render_html(metrics, inputs, notes)


def write_report(inputs: ReportInputs, output: Path | str) -> Metrics:
    metrics, html = build_report(inputs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return metrics


def _read_explanations(path: Path | None, metrics: Metrics) -> dict[str, str]:
    """Read the optional analyst notes: a JSON object of case_id to text."""
    if path is None:
        return {}
    if not path.exists():
        raise ReportError(f"Explanations file not found: {path}.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"Cannot read explanations file {path}: {exc}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ReportError(
            f"Explanations file {path} must be a JSON object of case_id to text."
        )
    known = {m.prediction.case_id for m in metrics.mismatches}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ReportError(
            f"Explanations file {path} names cases that are not error examples: "
            + ", ".join(unknown)
            + "."
        )
    return {k: v.strip() for k, v in data.items()}


def _read_generated(path: Path | None, metrics: Metrics) -> tuple[dict[str, str], str | None]:
    """Read model-written explanations: {"generator": ..., "explanations": {...}}."""
    if path is None:
        return {}, None
    if not path.exists():
        raise ReportError(f"Generated explanations file not found: {path}.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"Cannot read generated explanations file {path}: {exc}") from exc
    notes = data.get("explanations") if isinstance(data, dict) else None
    generator = data.get("generator") if isinstance(data, dict) else None
    if not isinstance(notes, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in notes.items()
    ) or not isinstance(generator, str):
        raise ReportError(
            f"Generated explanations file {path} must hold a generator name and "
            "an object of case_id to text."
        )
    known = {m.prediction.case_id for m in metrics.mismatches}
    return {k: v.strip() for k, v in notes.items() if k in known}, generator


# --- Rendering -------------------------------------------------------------


def render_html(metrics: Metrics, inputs: ReportInputs, notes: Notes) -> str:
    sections = [
        _summary(metrics),
        _verdict_section(metrics),
        _policy_section(metrics),
        _quality_section(metrics),
        _errors_section(metrics, notes),
        _failures_section(metrics),
        _notes_section(),
    ]
    return _PAGE.format(
        style=_STYLE,
        split=escape(inputs.split),
        data=escape(str(inputs.data_path)),
        predictions=escape(str(inputs.predictions_path)),
        body="\n".join(s for s in sections if s),
    )


def _summary(m: Metrics) -> str:
    coverage = f"{m.successful} of {m.total} scored"
    tiles = [
        ("Evaluated", str(m.total), "cases in the split", ""),
        ("Successful", str(m.successful), coverage, ""),
        ("Failed", str(m.failed), "no labels produced", "bad" if m.failed else ""),
        ("Verdict accuracy", _pct(m.verdict_accuracy), coverage, ""),
        ("Macro F1", _num(m.macro_f1), "PASS, FAIL, ESCALATE", ""),
        ("Quality MAE", _num(m.quality_mae, 2), f"baseline {_num(m.quality_baseline_mae, 2)}", ""),
    ]
    cells = "".join(
        f'<div class="tile {kind}">'
        f'<span class="label">{label}</span>'
        f'<span class="value">{value}</span>'
        f'<span class="hint">{hint}</span></div>'
        for label, value, hint, kind in tiles
    )
    return f'<section class="tiles" aria-label="Summary">{cells}</section>'


def _verdict_section(m: Metrics) -> str:
    correct = sum(m.confusion[i][i] for i in range(len(VERDICTS)))
    head = "".join(f"<th scope='col'>{v}</th>" for v in VERDICTS)
    peak = max((max(row) for row in m.confusion), default=0) or 1
    rows = []
    for i, gold in enumerate(VERDICTS):
        cells = []
        for j, count in enumerate(m.confusion[i]):
            kind = "hit" if i == j else "miss"
            share = round(12 + 58 * count / peak) if count else 0
            style = (
                f' style="background: color-mix(in srgb, var(--{kind}) {share}%, transparent)"'
                if count
                else ""
            )
            cells.append(f'<td class="cm {kind}"{style}>{count}</td>')
        rows.append(f"<tr><th scope='row'>{gold}</th>{''.join(cells)}</tr>")

    predicted = {v: sum(row[j] for row in m.confusion) for j, v in enumerate(VERDICTS)}
    class_rows = "".join(
        f"<tr><th scope='row'>{v}</th><td>{_num(s.precision)}</td>"
        f"<td>{_num(s.recall)}</td><td>{_num(s.f1)}</td><td>{s.support}</td>"
        f"<td>{predicted.get(v, 0)}</td></tr>"
        for v, s in m.per_class.items()
    )
    per_class = (
        f"""<div class="scroll"><table class="data">
<thead><tr><th scope="col">Class</th><th scope="col">Precision</th>
<th scope="col">Recall</th><th scope="col">F1</th><th scope="col">Gold count</th>
<th scope="col">Predicted count</th></tr></thead>
<tbody>{class_rows}</tbody></table></div>"""
        if class_rows
        else "<p class='muted'>No successful predictions to score.</p>"
    )
    return f"""<section>
<h2>Verdict</h2>
<dl class="facts">
  <div><dt>Accuracy</dt><dd>{_pct(m.verdict_accuracy)} <span class="muted">({correct} of {m.successful} scored)</span></dd></div>
  <div><dt>Strict accuracy</dt><dd>{_pct(m.strict_verdict_accuracy)} <span class="muted">({correct} of {m.total}, failed counted wrong)</span></dd></div>
  <div><dt>Macro F1</dt><dd>{_num(m.macro_f1)}</dd></div>
</dl>
<div class="pair">
  <figure>
    <figcaption>Confusion matrix (rows are gold, columns are predicted)</figcaption>
    <div class="scroll"><table class="matrix">
      <thead><tr><th scope="col"><span class="corner">gold ↓ / predicted →</span></th>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
  </figure>
  <figure>
    <figcaption>Per-class scores and verdict distribution</figcaption>
    {per_class}
  </figure>
</div>
</section>"""


def _policy_section(m: Metrics) -> str:
    c = m.policy_counts
    return f"""<section>
<h2>Policy violation</h2>
<p class="muted">A violation is the positive class.</p>
<div class="pair">
  <dl class="facts">
    <div><dt>Precision</dt><dd>{_num(m.policy_precision)} <span class="muted">({c['tp']} of {c['tp'] + c['fp']} flagged were real)</span></dd></div>
    <div><dt>Recall</dt><dd>{_num(m.policy_recall)} <span class="muted">({c['tp']} of {c['tp'] + c['fn']} real ones flagged)</span></dd></div>
  </dl>
  <div class="scroll"><table class="data">
    <thead><tr><th scope="col"></th><th scope="col">Predicted violation</th><th scope="col">Predicted none</th></tr></thead>
    <tbody>
      <tr><th scope="row">Gold violation</th><td>{c['tp']}</td><td>{c['fn']}</td></tr>
      <tr><th scope="row">Gold none</th><td>{c['fp']}</td><td>{c['tn']}</td></tr>
    </tbody>
  </table></div>
</div>
</section>"""


def _quality_section(m: Metrics) -> str:
    verdict = ""
    if m.quality_mae is not None:
        if m.quality_mae < m.quality_baseline_mae:
            verdict = "The evaluator beats the baseline."
        else:
            verdict = "The evaluator does not beat the baseline."
    return f"""<section>
<h2>Quality score</h2>
<dl class="facts">
  <div><dt>Mean absolute error</dt><dd>{_num(m.quality_mae, 2)} <span class="muted">points on the 0 to 4 scale, over {m.successful} scored cases</span></dd></div>
  <div><dt>Baseline MAE</dt><dd>{_num(m.quality_baseline_mae, 2)} <span class="muted">from always predicting {m.quality_baseline_score if m.quality_baseline_score is not None else "n/a"}, the median gold score</span></dd></div>
</dl>
<p class="muted">{verdict} Lower is better for both numbers.</p>
</section>"""


def _errors_section(m: Metrics, notes: Notes) -> str:
    if not m.mismatches:
        body = "<p class='muted'>Every scored prediction matches gold on verdict and policy.</p>"
    else:
        body = "".join(_error_card(x, notes) for x in m.mismatches)
    shortfall = ""
    if 0 < len(m.mismatches) < MIN_ERROR_EXAMPLES:
        shortfall = (
            f"<p class='muted'>Only {len(m.mismatches)} scored predictions differ "
            "from gold, so fewer than three examples exist.</p>"
        )
    return f"""<section>
<h2>Error examples</h2>
<p class="muted">{len(m.mismatches)} scored predictions differ from gold on verdict or policy. Verdict errors come first.</p>
{shortfall}
<div class="cards">{body}</div>
</section>"""


def _error_card(x: Mismatch, notes: Notes) -> str:
    p, g = x.prediction, x.gold
    tags = []
    if x.verdict_wrong:
        tags.append('<span class="tag bad">verdict</span>')
    if x.policy_wrong:
        tags.append('<span class="tag bad">policy</span>')
    if p.quality_score != g.quality:
        tags.append(f'<span class="tag">quality off by {abs(p.quality_score - g.quality)}</span>')
    why = [f"<div><dt>What decided it</dt><dd>{escape(code_explanation(x))}</dd></div>"]
    generated = notes.generated.get(p.case_id)
    if generated:
        source = escape(notes.generator or "the judge model")
        why.append(
            f"<div><dt>Likely cause <span class='muted'>(generated by {source})</span></dt>"
            f"<dd>{escape(generated)}</dd></div>"
        )
    analyst = notes.analyst.get(p.case_id)
    if analyst:
        why.append(f"<div><dt>Analyst note</dt><dd>{escape(analyst)}</dd></div>")
    trap = f"<div><dt>Trap</dt><dd>{escape(g.trap)}</dd></div>" if g.trap else ""
    return f"""<article class="card">
<header><h3>{escape(p.case_id)}</h3><div class="tags">{''.join(tags)}</div></header>
<div class="scroll"><table class="data compact">
  <thead><tr><th scope="col"></th><th scope="col">Verdict</th><th scope="col">Quality</th><th scope="col">Policy violation</th></tr></thead>
  <tbody>
    <tr><th scope="row">Predicted</th><td>{p.verdict}</td><td>{p.quality_score}</td><td>{_yes_no(p.policy_violation)}</td></tr>
    <tr><th scope="row">Gold</th><td>{g.verdict}</td><td>{g.quality}</td><td>{_yes_no(g.policy_violation)}</td></tr>
  </tbody>
</table></div>
<dl class="text">
  <div><dt>Evaluator's reason</dt><dd>{escape(p.reason)}</dd></div>
  <div><dt>Gold rationale</dt><dd>{escape(g.rationale)}</dd></div>
  {trap}
  {''.join(why)}
</dl>
</article>"""


def _failures_section(m: Metrics) -> str:
    if not m.failures:
        return ""
    rows = "".join(
        f"<tr><th scope='row'>{escape(f.case_id)}</th>"
        f"<td>{escape(f.error_type or 'unknown')}</td><td>{escape(f.reason)}</td></tr>"
        for f in m.failures
    )
    return f"""<section>
<h2>Failed records</h2>
<p class="muted">These cases got no labels and are left out of every metric except strict accuracy.</p>
<div class="scroll"><table class="data">
  <thead><tr><th scope="col">Case</th><th scope="col">Error type</th><th scope="col">Message</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>
</section>"""


def _notes_section() -> str:
    return """<section class="notes">
<h2>How to read these numbers</h2>
<ul>
  <li>All label metrics use successful predictions only, because a failed prediction has no labels to compare.</li>
  <li>Strict accuracy counts each failed prediction as wrong, so it is the lowest the accuracy could be.</li>
  <li>"n/a" means the metric is undefined because its denominator is zero, for example precision when nothing was predicted for that class.</li>
  <li>Macro F1 is the plain mean of the F1 for PASS, FAIL and ESCALATE, so a rare class counts as much as a common one.</li>
</ul>
</section>"""


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _yes_no(value: bool | None) -> str:
    return {True: "yes", False: "no"}.get(value, "n/a")


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="claude:viewport-fit-sides" content="cover">
<title>Agent Evaluator Report</title>
<style>{style}</style>
</head>
<body>
<main>
<header class="page-head">
  <p class="eyebrow">Evaluation report · {split} split</p>
  <h1>Agent Evaluator Report</h1>
  <p class="source">Gold labels from <code>{data}</code>, predictions from <code>{predictions}</code>.</p>
</header>
{body}
</main>
</body>
</html>
"""

_STYLE = """
/* Layout: one reading column; summary tiles, then one section per metric group. */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --fg: #1b2230; --muted: #5b6577;
  --line: #d9dee6; --accent: #2f5d8a; --hit: #2e7d4f; --miss: #b3412e;
  --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161d; --surface: #1a2029; --fg: #e4e8ef; --muted: #98a2b3;
    --line: #2d3542; --accent: #7fa8d6; --hit: #4fb47a; --miss: #e0735f;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12161d; --surface: #1a2029; --fg: #e4e8ef; --muted: #98a2b3;
  --line: #2d3542; --accent: #7fa8d6; --hit: #4fb47a; --miss: #e0735f;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.55 var(--font); }
main {
  max-width: 980px; margin: 0 auto; display: grid; gap: 36px;
  padding-block: 40px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
}
h1, h2, h3 { text-wrap: balance; margin: 0; line-height: 1.2; }
h1 { font-size: 2rem; letter-spacing: -0.01em; }
h2 { font-size: 1.25rem; padding-bottom: 8px; border-bottom: 1px solid var(--line); }
h3 { font-size: 1rem; font-family: var(--mono); }
p { margin: 0; }
code { font-family: var(--mono); font-size: 0.85em; overflow-wrap: anywhere; }
section { display: grid; gap: 16px; min-width: 0; }
.page-head { display: grid; gap: 8px; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.75rem; color: var(--accent); font-weight: 600; }
.source, .muted { color: var(--muted); }
.tiles { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.tile {
  display: grid; gap: 2px; padding: 14px 16px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px;
}
.tile .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.tile .value { font-size: 1.6rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.tile .hint { font-size: 0.8rem; color: var(--muted); }
.tile.bad .value { color: var(--miss); }
.facts { display: grid; gap: 8px; margin: 0; }
.facts div { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; }
.facts dt { min-width: 150px; color: var(--muted); }
.facts dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; }
.facts dd .muted { font-weight: 400; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px; align-items: start; }
.pair > * { min-width: 0; }
figure { margin: 0; display: grid; gap: 8px; }
figcaption { font-size: 0.85rem; color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; }
.data { width: 100%; }
.data th, .data td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--line); }
.data thead th { font-size: 0.8rem; color: var(--muted); font-weight: 600; }
.data tbody th { font-weight: 600; }
.data.compact th, .data.compact td { padding: 5px 8px; }
.matrix th, .matrix td { padding: 10px 14px; text-align: center; }
.matrix thead th { font-size: 0.8rem; color: var(--muted); }
.matrix tbody th { text-align: right; font-size: 0.8rem; color: var(--muted); }
.matrix .corner { font-weight: 400; white-space: nowrap; }
.cm { min-width: 64px; font-weight: 650; border: 1px solid var(--line); }
.cm.hit { color: var(--fg); }
.cards { display: grid; gap: 16px; }
.card {
  display: grid; gap: 12px; padding: 16px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; min-width: 0;
}
.card header { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; }
.tag {
  font-size: 0.75rem; padding: 1px 8px; border-radius: 999px;
  border: 1px solid var(--line); color: var(--muted);
}
.tag.bad { border-color: var(--miss); color: var(--miss); }
.text { display: grid; gap: 10px; margin: 0; }
.text dt { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.text dd { margin: 2px 0 0; max-width: 70ch; }
.notes ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; color: var(--muted); max-width: 75ch; }
"""
