"""Reporting Agent – experiment report generation.

Responsibilities (from playbook §3.13):
- Create concise Markdown reports.
- Include charts, metrics, and interpretation.
- Separate evidence from speculation.
- List what failed.
- Recommend next experiment.

Report template sections:
    Question, Hypothesis, Data, State Definitions, Model Setup,
    Backtest Rules, Results, Benchmark Comparison, What Worked,
    What Failed, Bias / Risk Checks, Next Experiment.
"""

from datetime import date
from pathlib import Path
import re


REPORT_TEMPLATE = """\
# Experiment Report: {name}

**Date:** {date}

## Question

{question}

## Hypothesis

{hypothesis}

## Data

{data_description}

## State Definitions

{state_definitions}

## Model Setup

{model_setup}

## Backtest Rules

{backtest_rules}

## Results

{results}

## Benchmark Comparison

{benchmark_comparison}

## What Worked

{what_worked}

## What Failed

{what_failed}

## Bias / Risk Checks

{bias_risk_checks}

## Next Experiment

{next_experiment}
"""


def generate_report(
    name: str,
    sections: dict[str, str],
    output_dir: str = "reports/runs",
) -> str:
    """Generate and save an experiment report as Markdown.

    Parameters
    ----------
    name : str
        Short experiment identifier.
    sections : dict[str, str]
        Content for each template section.
    output_dir : str
        Directory to save the .md file.

    Returns
    -------
    str
        Path to the saved report file.
    """
    defaults = {
        "question": "Not evaluated in this run.",
        "hypothesis": "Not evaluated in this run.",
        "data_description": "Not evaluated in this run.",
        "state_definitions": "Not evaluated in this run.",
        "model_setup": "Not evaluated in this run.",
        "backtest_rules": "Not evaluated in this run.",
        "results": "Not evaluated in this run.",
        "benchmark_comparison": "Not evaluated in this run.",
        "what_worked": "Not evaluated in this run.",
        "what_failed": "Not evaluated in this run.",
        "bias_risk_checks": "Not evaluated in this run.",
        "next_experiment": "Not evaluated in this run.",
    }
    content = defaults | sections
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-").lower()
    if not slug:
        raise ValueError("name must contain at least one filename-safe character")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"{slug}.md"
    output_path.write_text(
        REPORT_TEMPLATE.format(name=name, date=date.today().isoformat(), **content),
        encoding="utf-8",
    )
    return str(output_path)
