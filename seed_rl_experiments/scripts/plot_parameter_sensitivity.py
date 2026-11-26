# Wait until one of the experiments finishes to test if the code works well

#!/usr/bin/env python3
"""Plot per-seed evaluation scores for any Seed RL experiment summary."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# These identify the run rather than experimental settings.
IDENTITY_KEYS = {"env_id", "seed"}


def value_key(value):
    """Stable, sortable representation for config values."""
    return json.dumps(value, sort_keys=True)


def display_value(value):
    """Readable label for a config value."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def sort_values(values):
    """Sort numbers numerically and other values alphabetically."""
    def key(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (0, float(value))
        return (1, display_value(value).lower())

    return sorted(values, key=key)


def load_rows(summary_path, score_column):
    with summary_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {summary_path}")

    required = {"run_directory", "env_id", "seed", score_column}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"{summary_path} is missing required column(s): "
            f"{', '.join(sorted(missing))}. "
            f"Available columns: {', '.join(rows[0].keys())}"
        )

    joined = []
    all_config_keys = set()

    for row in rows:
        run_dir = Path(row["run_directory"])
        config_path = run_dir / "config.json"

        if not config_path.is_file():
            raise FileNotFoundError(f"Could not find {config_path}")

        with config_path.open() as f:
            config = json.load(f)

        score = float(row[score_column])
        if not math.isfinite(score):
            raise ValueError(
                f"Non-finite {score_column} for run {run_dir}: {score}"
            )

        item = dict(row)
        item["env_id"] = row["env_id"]
        item["seed"] = int(row["seed"])
        item[score_column] = score
        item["_config"] = config
        joined.append(item)
        all_config_keys.update(config)

    # Auto-detect config values that vary across these runs.
    varying = []
    for key in sorted(all_config_keys - IDENTITY_KEYS):
        values = {
            value_key(row["_config"].get(key))
            for row in joined
        }
        if len(values) > 1:
            varying.append(key)

    return joined, varying


def write_joined_csv(rows, varying, score_column, output_path):
    """Save the summary rows with selected config fields appended."""
    fields = [
        key for key in rows[0]
        if key != "_config"
    ]

    for key in varying:
        if key not in fields:
            fields.append(key)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            output_row = {key: row.get(key, "") for key in fields}
            for key in varying:
                output_row[key] = json.dumps(
                    row["_config"].get(key), sort_keys=True
                )
            writer.writerow(output_row)


def plot_baseline(rows, score_column, output_path):
    """Plot a baseline with no varying configuration parameters."""
    tasks = sorted({row["env_id"] for row in rows})
    fig, axes = plt.subplots(
        1, len(tasks),
        figsize=(4.5 * len(tasks), 4),
        squeeze=False,
    )

    for i, task in enumerate(tasks):
        ax = axes[0][i]
        task_rows = [r for r in rows if r["env_id"] == task]
        scores = [r[score_column] for r in task_rows]
        seeds = [r["seed"] for r in task_rows]

        ax.scatter(range(len(scores)), scores, s=55, zorder=3)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(seed) for seed in seeds])
        ax.set_xlabel("Seed")
        ax.set_title(task)
        ax.set_ylabel(score_column.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)

        if len(scores) > 1:
            mean = statistics.mean(scores)
            sd = statistics.stdev(scores)
            ax.axhline(mean, color="black", linestyle="--",
                       label=f"Mean ± sample SD: {mean:.2f} ± {sd:.2f}")
            ax.legend(fontsize="small")

    fig.suptitle("Baseline scores by task and seed")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parameter(rows, parameter, other_parameters, score_column,
                   output_path):
    """Plot one parameter, with other varying parameters as separate series."""
    tasks = sorted({row["env_id"] for row in rows})
    x_values = sort_values({
        value_key(row["_config"].get(parameter)): row["_config"].get(parameter)
        for row in rows
    }.values())
    x_positions = {value_key(value): i for i, value in enumerate(x_values)}

    # A series is a particular combination of the other varying parameters.
    series_by_key = {}
    for row in rows:
        key = tuple(
            value_key(row["_config"].get(param))
            for param in other_parameters
        )
        label = ", ".join(
            f"{param}={display_value(row['_config'].get(param))}"
            for param in other_parameters
        ) or "all runs"
        series_by_key[key] = label

    series_keys = sorted(series_by_key)
    colors = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        len(tasks), 1,
        figsize=(max(7, 1.25 * len(x_values)), 3.5 * len(tasks)),
        squeeze=False,
        sharex=True,
    )

    # Group rows by task, x-value, and the other parameter settings.
    grouped = defaultdict(list)
    for row in rows:
        series_key = tuple(
            value_key(row["_config"].get(param))
            for param in other_parameters
        )
        group_key = (
            row["env_id"],
            value_key(row["_config"].get(parameter)),
            series_key,
        )
        grouped[group_key].append(row)

    # Slightly offset series at the same x-position so seed dots don't overlap.
    if len(series_keys) == 1:
        series_offsets = {series_keys[0]: 0.0}
    else:
        spacing = min(0.18, 0.6 / len(series_keys))
        center = (len(series_keys) - 1) / 2
        series_offsets = {
            key: (i - center) * spacing
            for i, key in enumerate(series_keys)
        }

    for task_index, task in enumerate(tasks):
        ax = axes[task_index][0]

        for series_index, series_key in enumerate(series_keys):
            color = colors(series_index % 10)

            for x_value in x_values:
                group_key = (task, value_key(x_value), series_key)
                group_rows = grouped.get(group_key, [])
                if not group_rows:
                    continue

                x = (
                    x_positions[value_key(x_value)]
                    + series_offsets[series_key]
                )
                scores = [row[score_column] for row in group_rows]

                # Plot individual seed results.
                for row in group_rows:
                    ax.scatter(
                        x,
                        row[score_column],
                        color=color,
                        alpha=0.75,
                        s=42,
                        zorder=3,
                    )

                # Overlay mean and ±1 sample SD across available seeds.
                mean = statistics.mean(scores)
                sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
                ax.errorbar(
                    x, mean, yerr=sd,
                    fmt="D",
                    color=color,
                    markersize=5,
                    capsize=3,
                    linewidth=1.2,
                    label=series_by_key[series_key],
                    zorder=4,
                )

        ax.set_title(task)
        ax.set_ylabel(score_column.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)

        # Avoid repeating the same legend entry for every x-value.
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            ax.legend(
                unique.values(),
                unique.keys(),
                fontsize="small",
                loc="best",
            )

    axes[-1][0].set_xticks(range(len(x_values)))
    axes[-1][0].set_xticklabels(
        [display_value(value) for value in x_values],
        rotation=30,
        ha="right",
    )
    axes[-1][0].set_xlabel(parameter)

    fig.suptitle(
        f"Evaluation score by {parameter}\n"
        "Dots are individual seeds; diamonds/error bars are mean ± sample SD"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-seed",
        type=Path,
        required=True,
        help="Path to the summarizer's per_seed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for joined CSV and plots",
    )
    parser.add_argument(
        "--score-column",
        default="mean_evaluation_return",
        help="Score column in per_seed.csv (default: mean_evaluation_return)",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        help=(
            "Optional config keys to plot. By default, plots every config "
            "field that varies across the input runs."
        ),
    )
    args = parser.parse_args()

    rows, detected_parameters = load_rows(
        args.per_seed, args.score_column
    )

    parameters = args.parameters or detected_parameters
    unknown = set(parameters) - set(rows[0]["_config"])
    if unknown:
        raise ValueError(
            "Config key(s) not found: " + ", ".join(sorted(unknown))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    joined_path = args.output_dir / "per_seed_with_config.csv"
    write_joined_csv(
        rows, parameters, args.score_column, joined_path
    )
    print(f"Wrote {joined_path}")

    if not parameters:
        output_path = args.output_dir / "baseline_seed_scores.png"
        plot_baseline(rows, args.score_column, output_path)
        print(f"No varying config parameters found; wrote {output_path}")
        return

    for parameter in parameters:
        other_parameters = [
            key for key in parameters if key != parameter
        ]
        output_path = (
            args.output_dir / f"sensitivity_{parameter}.png"
        )
        plot_parameter(
            rows,
            parameter,
            other_parameters,
            args.score_column,
            output_path,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
