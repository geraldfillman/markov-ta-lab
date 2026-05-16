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
    raise NotImplementedError("Implement in Reporting Agent phase")
