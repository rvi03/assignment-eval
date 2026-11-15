# AI Agent Response Evaluator

A command-line tool that evaluates the quality and safety of an AI agent's
work. For each case it judges whether the agent completed the task, stayed
within its constraints and authority limits, and used tools and evidence
appropriately, then returns `PASS`, `FAIL` or `ESCALATE` with a short
reason.

**Status:** The data loader and validator are implemented. Evaluation,
batch runs and reporting are in development.

## Contents

- [Getting started](#getting-started)
- [Usage](#usage)
- [Input data](#input-data)
- [Validation rules](#validation-rules)
- [Development](#development)
- [Assumptions and limitations](#assumptions-and-limitations)

## Getting started

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Any recent | Installs Python 3.12 and all dependencies. |
| `make` | Any | Preinstalled on macOS and Linux. On Windows, run the `uv` commands from the Makefile directly. |

### Installation

```bash
git clone <repository-url>
cd <repository-directory>
make install
```

`make install` creates `.venv/` with Python 3.12 and installs the
dependencies pinned in `uv.lock`. The tool has no runtime dependencies;
`pytest` is the only development dependency.

### Data

Place the benchmark CSV at:

```
.data/ai_agent_evaluator_benchmark.csv
```

`.data/` is listed in `.gitignore`, so datasets are never committed.

## Usage

Load and validate a CSV from Python:

```python
from evaluator.loader import load_cases, write_errors

result = load_cases(".data/ai_agent_evaluator_benchmark.csv")
print(f"{len(result.cases)} valid rows, {len(result.errors)} invalid rows")

write_errors(result.errors, "outputs/validation_errors.jsonl")
```

`load_cases` returns a `LoadResult` with two lists:

| Field | Type | Contents |
|---|---|---|
| `cases` | `list[Case]` | One `Case` per valid row, holding a `CaseInput` and its `GoldLabels`. |
| `errors` | `list[RowError]` | One `RowError` per invalid row, listing every problem found. |

`load_cases` raises `DataFileError` only when the file cannot be read as a
whole (see [Validation rules](#validation-rules)).

### Error file format

`write_errors` writes one JSON object per line (JSON Lines), one line per
invalid row:

```json
{
  "row": 14,
  "case_id": "AE-013",
  "errors": [
    {"field": "state_json", "message": "Malformed JSON in state_json: Expecting value (line 1, column 9)."}
  ],
  "record": {"case_id": "AE-013", "split": "calibration", "...": "..."}
}
```

| Key | Meaning |
|---|---|
| `row` | Row number in the file, counting the header as row 1. |
| `case_id` | The row's `case_id`, or `null` when it has none. |
| `errors` | Every field that failed and why. |
| `record` | The complete raw row, exactly as read. |

## Input data

Each CSV row describes one agent case. Required columns:

| Column | Description |
|---|---|
| `case_id` | Unique case identifier. |
| `split` | `calibration` or `test`. |
| `capability` | Primary behavior being evaluated. |
| `domain` | Business domain of the scenario. |
| `difficulty` | `easy`, `medium` or `hard`. |
| `language` | Language code of the case. |
| `task` | Work the agent was asked to perform. |
| `constraints` | Rules, limits or required output format. |
| `available_tools` | Comma-separated tool names, or `none`. |
| `agent_trajectory` | The agent's tool-use steps. May be empty. |
| `agent_final_answer` | The agent's final answer. May be empty. |
| `state_json` | JSON object holding the same case fields. |
| `gold_verdict` | Expected verdict: `PASS`, `FAIL` or `ESCALATE`. |
| `gold_quality` | Expected quality score, an integer from 0 to 4. |
| `gold_policy_violation` | Expected policy-violation flag: `true` or `false`. |
| `gold_rationale` | Reference explanation for the label. |
| `trap` | Main failure mode the case represents. |

### Labels never reach the evaluator

Each valid row is split into two records:

- `CaseInput` holds only what the evaluator may see: the case metadata,
  task, constraints, tools, trajectory and final answer.
- `GoldLabels` holds `gold_*`, `gold_rationale` and `trap`, and is used
  only for scoring.

The evaluator receives `CaseInput` alone, so labels cannot influence a
prediction.

## Validation rules

### File-level errors

These stop the load with `DataFileError`, because no row can be trusted:

- The file is missing or unreadable.
- The file is empty or has no header.
- A required column is missing from the header.

### Row-level errors

Any of these makes a row invalid. The row becomes a `RowError` and every
other row still loads.

| Field | Invalid when |
|---|---|
| `case_id` | Empty in both the column and `state_json`, or shared by more than one row. |
| `state_json` | Empty, malformed JSON, or not a JSON object. |
| `split` | Not `calibration` or `test`. |
| `difficulty` | Not `easy`, `medium` or `hard`. |
| `task`, `constraints`, `agent_trajectory`, `agent_final_answer` | Empty in both the column and `state_json`. |
| `available_tools` | Empty column and no list of strings in `state_json`. |
| `gold_verdict` | Not exactly `PASS`, `FAIL` or `ESCALATE`. |
| `gold_quality` | Not an integer from 0 to 4 (`3.0` is rejected). |
| `gold_policy_violation` | Not exactly `true` or `false`. |
| Whole row | Has a different number of fields than the header. |

## Development

| Command | Purpose |
|---|---|
| `make install` | Create `.venv/` and install dependencies. |
| `make test` | Run the test suite. |
| `make cli-help` | Show the command-line help. |
| `make clean` | Remove `.venv/` and generated caches. |
| `make help` | List all targets. |

The test suite includes one test that loads the real benchmark file. It
fails with a clear message if `.data/ai_agent_evaluator_benchmark.csv` is
missing.

### Project layout

```
evaluator/
  cli.py        Command-line entry point
  loader.py     CSV loading, validation and the error file writer
  models.py     Record types and allowed values
tests/
  test_cli.py
  test_loader.py
```

## Assumptions and limitations

### Column values take precedence

- For `case_id`, `task`, `constraints`, `available_tools`,
  `agent_trajectory` and `agent_final_answer`, a filled top-level column
  is used as is.
- `state_json` is read only when that column is empty.
- If a filled column and `state_json` disagree, the column wins and no
  error is reported.

### Validation scope

- Every row with a duplicated `case_id` is invalid, because no copy can be
  identified as the correct one.
- A row with an invalid gold label is invalid, because its prediction
  could not be scored.
- `capability`, `domain`, `language`, `gold_rationale` and `trap` are
  not validated.

### Batch mode and file size

- Batch mode means one non-interactive job that processes a whole CSV file
  and produces one result per selected row.
- The loader reads the whole file into memory.
- Files larger than available memory would need chunked processing with
  checkpoints so that a failed job can resume. This is not supported.
