from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from src.constants import EMOTIONS

if TYPE_CHECKING:
    from src.evaluation.compare import ComparisonSummary
    from src.evaluation.results import ResultsSummary
    from src.evaluation.scoring import ScoreSummary
    from src.finetune.selection import SelectionSummary


def _metrics_table(title: str, columns: list[str], values: list[dict]) -> Table:
    """Build a table with one row per aggregate metric and one column per scored system.

    Args:
        title (str): The table title.
        columns (list[str]): The column headers, one per system and language.
        values (list[dict]): The metrics of each column.

    Returns:
        Table: A table.
    """
    from src.evaluation.metrics import AGGREGATE_METRICS, PRIMARY_METRICS, metric_label

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

    for rank, result in enumerate(summary.results, start=1):
        table.add_row(
            str(rank),
            result.run_name,
            *[f"{result.language_scores.get(lang, 0.0):.4f}" for lang in summary.langs],
            f"{result.selection_score:.4f}",
        )

    console.print(table)
    console.print(f"Rule: {summary.rule}")
    console.print(f"Final configuration: [bold cyan]{summary.best.run_name}[/bold cyan]")


def render_results(console: Console, summary: ResultsSummary) -> None:
    """Render the final results table, one column per system and language.

    Args:
        console (Console): The console.
        summary (ResultsSummary): The summary to render.
    """
    console.print(
        _metrics_table(
            f"{summary.split} results, configuration '{summary.run_name}'",
            [f"{row['system']}\n{row['eval_lang']}" for row in summary.rows],
            summary.rows,
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
