#!/usr/bin/env python3
"""Diagnose whether Stable runtime follows sampled subset counts."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MODEL = "gradient_based_joint_edge_score_stable"


def latest_result_file(performance_dir: Path) -> Path:
    files = sorted(performance_dir.glob("multi_dataset_repeated_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No repeated-result JSON under {performance_dir}")
    return files[-1]


def read_benchmark_times(point_dir: Path, dataset: str) -> dict:
    result_file = latest_result_file(point_dir / "performance")
    data = json.loads(result_file.read_text())
    model_data = data["results"][dataset][MODEL]
    runs = [run for run in model_data["runs"] if run.get("success")]
    stats = model_data["statistics"]
    avg_summarization = sum(run.get("summarization_time", 0.0) for run in runs) / len(runs)
    avg_training = sum(run.get("training_time", 0.0) for run in runs) / len(runs)
    return {
        "avg_run_time": stats["avg_run_time"],
        "avg_training_time": avg_training,
        "avg_summarization_time": avg_summarization,
        "avg_other_time": stats["avg_run_time"] - avg_summarization,
        "performance_file": str(result_file),
    }


def read_log_subset_counts(log_dir: Path) -> pd.DataFrame:
    rows = []
    for log_file in sorted(log_dir.glob("gs_stable_hp_*.out")):
        text = log_file.read_text(errors="replace")
        subset_match = re.search(r"^Subset num: (\S+)", text, re.M)
        repeats_match = re.search(r"^Sampling repeats: (\S+)", text, re.M)
        if not subset_match or not repeats_match:
            continue
        counts = [int(value) for value in re.findall(r"Sampled (\d+) joint deletion subsets", text)]
        scoring_times = [
            float(value)
            for value in re.findall(r"Joint subset scoring completed in ([0-9.]+) seconds", text)
        ]
        rows.append(
            {
                "sampling_subset_num": subset_match.group(1),
                "sampling_repeats": int(repeats_match.group(1)),
                "actual_sampled_counts": "/".join(map(str, sorted(set(counts)))),
                "mean_sampled_subsets_per_step": sum(counts) / len(counts),
                "mean_joint_subset_scoring_s": sum(scoring_times) / len(scoring_times),
                "total_joint_subset_scoring_s": sum(scoring_times),
                "num_scoring_events": len(scoring_times),
                "log_file": str(log_file),
            }
        )
    return pd.DataFrame(rows)


def collect_runtime_rows(base_dir: Path, log_dir: Path, dataset: str) -> pd.DataFrame:
    rows = []
    log_df = read_log_subset_counts(log_dir)
    for subset_dir in sorted(base_dir.glob("subset_num_*")):
        subset = subset_dir.name.removeprefix("subset_num_")
        for repeats_dir in sorted(subset_dir.glob("sampling_repeats_*")):
            repeats = int(repeats_dir.name.removeprefix("sampling_repeats_"))
            row = {
                "sampling_subset_num": subset,
                "sampling_repeats": repeats,
                "expected_subsets_per_step": None if subset == "all" else int(subset) * repeats,
            }
            row.update(read_benchmark_times(repeats_dir, dataset))
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.merge(log_df, on=["sampling_subset_num", "sampling_repeats"], how="left")
    order = {"2": 0, "4": 1, "6": 2, "8": 3, "10": 4, "all": 5}
    df["subset_order"] = df["sampling_subset_num"].map(order)
    return df.sort_values(["sampling_repeats", "subset_order"])


def plot_runtime_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_df = df.dropna(subset=["expected_subsets_per_step"])
    metrics = [
        ("avg_run_time", "Benchmark total runtime per seed"),
        ("avg_summarization_time", "Summarization runtime per seed"),
        ("mean_joint_subset_scoring_s", "Mean joint subset scoring event"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        for repeats, group in numeric_df.groupby("sampling_repeats"):
            group = group.sort_values("expected_subsets_per_step")
            ax.plot(
                group["expected_subsets_per_step"],
                group[metric],
                marker="o",
                label=f"R={int(repeats)}",
            )
            for _, row in group.iterrows():
                ax.annotate(
                    f"N={row['sampling_subset_num']}",
                    (row["expected_subsets_per_step"], row[metric]),
                    fontsize=7,
                    xytext=(4, 3),
                    textcoords="offset points",
                )
        ax.set_title(title)
        ax.set_xlabel("Expected sampled subsets per pruning step (N x R)")
        ax.set_ylabel("Seconds")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_diagnostics_by_sampled_subsets.png", dpi=220)
    fig.savefig(output_dir / "runtime_diagnostics_by_sampled_subsets.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", type=Path, default=Path("results/stable_hparam_split_count_curves"))
    parser.add_argument("--log-dir", type=Path, default=Path("results/job_logs"))
    parser.add_argument("--dataset", default="Cora")
    args = parser.parse_args()

    output_dir = args.base_output_dir / "plots"
    df = collect_runtime_rows(args.base_output_dir, args.log_dir, args.dataset)
    csv_file = args.base_output_dir / "stable_hparam_runtime_diagnostics.csv"
    df.to_csv(csv_file, index=False)
    plot_runtime_diagnostics(df, output_dir)
    print(df.to_string(index=False))
    print(f"Wrote {csv_file}")
    print(f"Wrote runtime diagnostic plots to {output_dir}")


if __name__ == "__main__":
    main()
