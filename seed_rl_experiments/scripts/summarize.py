"""Summarize measured Seed RL runs using the paper's checkpoint-mean metric."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seed_ppo.reporting import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    parser.add_argument("--no-plots", action="store_true", help="Write CSV files without importing matplotlib")
    args = parser.parse_args()
    try:
        rows = summarize(args.runs, args.output, plots=not args.no_plots)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, "Cannot summarize runs: %s\n" % error)
    for row in rows:
        print("%s/%s: median seed score=%.2f; median final return=%.2f [%s]" % (
            row["variant"], row["env_id"], row["median_seed_score"],
            row["median_final_return"], row["status"]))
    names = "per_seed.csv and aggregate.csv" if args.no_plots else "per_seed.csv, aggregate.csv, and learning_curves.png"
    print("Wrote %s to %s" % (names, args.output))


if __name__ == "__main__":
    main()
