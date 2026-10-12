from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from src.constants import (
    AGGREGATE_METRICS,
    DIAGNOSTIC_METRICS,
    EMOTIONS,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
)

if TYPE_CHECKING:
    from src.evaluation.compare import ComparisonSummary
    from src.evaluation.results import ResultsSummary
    from src.evaluation.scoring import ScoreSummary
    from src.finetune.selection import SelectionSummary
    from src.prompting.summary import PromptingSummary


def _metrics_table(title: str, columns: list[str], values: list[dict]) -> Table:
    """Build a table with one row per aggregate metric and one column per scored system.

    Args:
        title (str): The table title.
        columns (list[str]): The column headers, one per system and language.
        values (list[dict]): The metrics of each column.

    Returns:
        Table: A table.
    """
    from src.evaluation.metrics import metric_label

    table = Table(title=title)

    table.add_column("Metric")

    for column in columns:
        table.add_column(column, justify="right", style="cyan")

    for key in AGGREGATE_METRICS:
        table.add_row(
            metric_label(key),
            *(f"{metrics[key]:.4f}" for metrics in values),
            style="bold" if key in PRIMARY_METRICS else None,
        )

    table.add_row("Lines", *(str(int(metrics["count"])) for metrics in values))

    return table


def render_scores(console: Console, summary: ScoreSummary) -> None:
    """Render the scores of a fine-tuned run or a prompted model, overall and per emotion.

    Args:
        console (Console): The console.
        summary (ScoreSummary): The summary to render.
    """
    console.print(
        _metrics_table(
            f"Scores on {summary.split} ({summary.condition})",
            list(summary.scores),
            list(summary.scores.values()),
        )
    )

    for lang, metrics in summary.scores.items():
        emotions = Table(title=f"Per emotion, {lang} {summary.split}")

        for column in ("Emotion", "Precision", "Recall", "F1", "Support"):
            emotions.add_column(column, justify="left" if column == "Emotion" else "right")

        for emotion in EMOTIONS:
            row = metrics["per_emotion"][emotion]
            emotions.add_row(
                emotion,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1']:.4f}",
                str(row["support"]),
            )

        console.print(emotions)

    console.print(f"Scores saved to [cyan]{summary.scores_file}[/cyan]")


def render_selection(console: Console, summary: SelectionSummary) -> None:
    """Render the ranked candidates of the dev-set selection.

    Args:
        console (Console): The console.
        summary (SelectionSummary): The summary to render.
    """
    table = Table(title=f"Candidates ranked by dev {summary.metric}")

    table.add_column("Rank", justify="right")
    table.add_column("Run name", style="cyan")

    for lang in summary.langs:
        table.add_column(f"{lang} (mean)", justify="right")

    table.add_column("Selection score", justify="right", style="bold")

    for row in summary.ranking():
        table.add_row(
            str(row["rank"]),
            row["run_name"],
            *[f"{row[f'{lang}_mean'] or 0.0:.4f}" for lang in summary.langs],
            f"{row['selection_score']:.4f}",
        )

    console.print(table)
    console.print(f"Rule: {summary.rule}")
    console.print(f"Final configuration: [bold cyan]{summary.best.run_name}[/bold cyan]")


type Column = Callable[[dict[str, Any]], str]


def _field(name: str) -> Column:
    """Get a column that shows one field of a row.

    Args:
        name (str): The field.

    Returns:
        Column: A function from a row to its cell.
    """
    return lambda row: str(row[name])


def _rows_table(
    title: str,
    keys: dict[str, Column],
    metrics: Sequence[str],
    rows: list[dict[str, Any]],
    cell: Callable[[dict[str, Any], str], str] = lambda row, key: f"{row[key]:.4f}",
    extra: dict[str, Column] | None = None,
) -> Table:
    """Get a table with one row per result, one column per key, and one column per metric.

    Args:
        title (str): The table title.
        keys (dict[str, Column]): The key columns, by header.
        metrics (Sequence[str]): The metric keys, shown with their short labels.
        rows (list[dict[str, Any]]): The results.
        cell (Callable[[dict[str, Any], str], str]): The cell of a row and metric. Defaults to a
                                                     float with four decimals.
        extra (dict[str, Column] | None): The columns after the metrics, by header. Defaults to
                                          None.

    Returns:
        Table: A table.
    """
    from src.evaluation.metrics import metric_label

    extra = extra or {}
    table = Table(title=title)

    for header in keys:
        table.add_column(header)

    for key in metrics:
        table.add_column(metric_label(key), justify="right", style="cyan")

    for header in extra:
        table.add_column(header, justify="right")

    for row in rows:
        table.add_row(
            *(column(row) for column in keys.values()),
            *(cell(row, key) for key in metrics),
            *(column(row) for column in extra.values()),
        )

    return table


def render_results(console: Console, summary: ResultsSummary) -> None:
    """Render the final results, one row per system and language.

    Args:
        console (Console): The console.
        summary (ResultsSummary): The summary to render.
    """
    title = f"{summary.split} results, configuration '{summary.run_name}'"
    keys = {"System": _field("system"), "Lang": _field("eval_lang")}

    console.print(_rows_table(title, keys, (*PRIMARY_METRICS, *SECONDARY_METRICS), summary.rows))
    console.print(
        _rows_table(
            f"{title} (diagnostics)",
            keys,
            DIAGNOSTIC_METRICS,
            summary.rows,
            extra={"Lines": lambda row: str(int(row["count"]))},
        )
    )
    _experiment_tables(console, summary)

    for path in summary.files:
        console.print(f"Saved [cyan]{path}[/cyan]")


def render_prompting(console: Console, summary: PromptingSummary) -> None:
    """Render the prompting results, one row per configuration and language.

    Args:
        console (Console): The console.
        summary (PromptingSummary): The summary to render.
    """
    variant = {"Model": _field("model"), "Shots": _field("shots"), "Selection": _field("selection")}

    console.print(
        _rows_table(
            f"Prompting on {summary.split}",
            {**variant, "Template": _field("template"), "Lang": _field("eval_lang")},
            ("micro_f1", "jaccard_samples", "macro_f1", "hamming_loss", "malformed_rate"),
            summary.rows,
        )
    )
    console.print(
        _rows_table(
            f"Prompt sensitivity on {summary.split}: mean ± std (range) over templates",
            {
                **variant,
                "Lang": _field("eval_lang"),
                "Templates": lambda row: str(len(row["templates"])),
            },
            PRIMARY_METRICS,
            summary.sensitivity,
            cell=lambda row, key: (
                f"{row[f'{key}_mean']:.3f} ± {row[f'{key}_std']:.3f} ({row[f'{key}_range']:.3f})"
            ),
        )
    )

    for path in summary.files:
        console.print(f"Saved [cyan]{path}[/cyan]")


def render_comparison(console: Console, summary: ComparisonSummary) -> None:
    """Render the outcome counts of a comparison.

    Args:
        console (Console): The console.
        summary (ComparisonSummary): The summary to render.
    """
    table = Table(title=f"{summary.name} ({summary.lang})")

    table.add_column("Category", style="cyan")
    table.add_column("Lines", justify="right")
    table.add_column("Share", justify="right")

    total = len(summary.rows)

    for category, count in summary.counts.items():
        table.add_row(category, str(count), f"{count / total:.1%}")

    console.print(table)

    for path in summary.files:
        console.print(f"Saved [cyan]{path}[/cyan]")


def _experiment_tables(console: Console, summary: ResultsSummary) -> None:
    """Get the mean and standard deviation of every metric for every system and language, across the
    main run and its repetitions, and the learning curve of the selected configuration, across the
    main run and its capped runs.

    Args:
        console (Console): The console.
        summary (ResultsSummary): The summary to render.
    """
    from src.evaluation.metrics import metric_label

    if summary.seeds:
        seeds = ", ".join(str(seed) for seed in summary.seeds[0]["seeds"])
        table = Table(title=f"Mean ± sd on {summary.split} over seeds {seeds} ({summary.run_name})")

        for column in ("System", "Language", *map(metric_label, PRIMARY_METRICS)):
            table.add_column(
                column, justify="left" if column in ("System", "Language") else "right"
            )

        for row in summary.seeds:
            table.add_row(
                row["system"],
                row["eval_lang"],
                *(f"{row[key]['mean']:.4f} ± {row[key]['std']:.4f}" for key in PRIMARY_METRICS),
            )

        console.print(table)

    if summary.curve:
        table = Table(title=f"Learning curve on {summary.split} ({summary.run_name})")

        for column in ("System", "Lines", *map(metric_label, PRIMARY_METRICS)):
            table.add_column(column, justify="left" if column == "System" else "right")

        for row in sorted(summary.curve, key=lambda row: (row["system"], row["lines"])):
            table.add_row(
                row["system"],
                str(row["lines"]),
                *(
                    f"{row[key]:.4f} [{row[f'{key}_ci'][0]:.4f}, {row[f'{key}_ci'][1]:.4f}]"
                    for key in PRIMARY_METRICS
                ),
            )

        console.print(table)
