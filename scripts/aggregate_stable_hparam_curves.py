#!/usr/bin/env python3
"""Aggregate and plot Stable hyperparameter runtime-performance curves."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def latest_json(directory: Path) -> Optional[Path]:
    files = sorted(directory.glob("multi_dataset_repeated_results_*.json"))
    return files[-1] if files else None


def read_performance(point_dir: Path, dataset: str, model: str) -> Dict:
    perf_file = latest_json(point_dir / "performance")
    if perf_file is None:
        return {"performance_missing": True}

    data = json.loads(perf_file.read_text())
    stats = data.get("results", {}).get(dataset, {}).get(model, {}).get("statistics", {})
    return {
        "performance_missing": False,
        "ic_auc_add_mean": stats.get("ic_auc_additive", {}).get("mean"),
        "ic_auc_add_stderr": stats.get("ic_auc_additive", {}).get("stderr"),
        "ic_auc_log_mean": stats.get("ic_auc_log_ratio", {}).get("mean"),
        "ic_auc_log_stderr": stats.get("ic_auc_log_ratio", {}).get("stderr"),
        "threshold_add_mean": stats.get("threshold_point_additive", {}).get("mean"),
        "threshold_log_mean": stats.get("threshold_point_log_ratio", {}).get("mean"),
        "total_time_s": stats.get("total_run_time"),
        "avg_time_s": stats.get("avg_run_time"),
        "num_successful": stats.get("num_successful"),
        "num_failed": stats.get("num_failed"),
        "performance_file": str(perf_file),
    }


def read_path_stability(point_dir: Path, subset_num: str, repeats: str, model: str) -> Dict:
    summary_file = (
        point_dir
        / "path_stability"
        / f"subset_num_{subset_num}"
        / f"sampling_repeats_{repeats}"
        / "summary.json"
    )
    if not summary_file.exists():
        return {"path_stability_missing": True}

    data = json.loads(summary_file.read_text())
    rows = data.get("summary", [])
    row = next((item for item in rows if item.get("model") == model), rows[0] if rows else {})
    return {
        "path_stability_missing": False,
        "scoring_switch_aupa_mid": row.get("aupa_mid_mean"),
        "scoring_switch_deleted_aupa_mid": row.get("deleted_aupa_mid_mean"),
        "scoring_switch_deletion_spearman": row.get("deletion_time_spearman_mean"),
        "scoring_switch_final_jaccard": row.get("final_non_empty_retained_jaccard_mean"),
        "gcn_path_time_mean": row.get("gcn_time_mean"),
        "gat_path_time_mean": row.get("gat_time_mean"),
        "path_summary_file": str(summary_file),
    }


def read_seed_stability(point_dir: Path, model: str) -> Dict:
    csv_file = point_dir / "seed_stability_summary.csv"
    if not csv_file.exists():
        return {"seed_stability_missing": True}

    df = pd.read_csv(csv_file)
    df = df[df["model"] == model]
    result = {"seed_stability_missing": False, "seed_stability_file": str(csv_file)}
    for scoring_model in ["gcn", "gat"]:
        subset = df[df["scoring_model"] == scoring_model]
        if subset.empty:
            continue
        row = subset.iloc[0]
        prefix = f"seed_{scoring_model}"
        result[f"{prefix}_aupa_mid"] = row.get("aupa_mid_mean")
        result[f"{prefix}_deleted_aupa_mid"] = row.get("deleted_aupa_mid_mean")
        result[f"{prefix}_deletion_corr"] = row.get("deletion_time_corr_mean")
        result[f"{prefix}_final_jaccard"] = row.get("final_non_empty_retained_jaccard_mean")
    return result


def collect_rows(base_output_dir: Path, dataset: str, model: str) -> List[Dict]:
    rows = []
    for subset_dir in sorted(base_output_dir.glob("subset_num_*")):
        subset_num = subset_dir.name.removeprefix("subset_num_")
        for repeats_dir in sorted(subset_dir.glob("sampling_repeats_*")):
            repeats = repeats_dir.name.removeprefix("sampling_repeats_")
            row = {
                "sampling_subset_num": subset_num,
                "sampling_repeats": int(repeats),
                "point_dir": str(repeats_dir),
            }
            if subset_num != "all":
                row["sampling_subset_num_numeric"] = int(subset_num)
            else:
                row["sampling_subset_num_numeric"] = None
            row.update(read_performance(repeats_dir, dataset, model))
            row.update(read_path_stability(repeats_dir, subset_num, repeats, model))
            row.update(read_seed_stability(repeats_dir, model))
            rows.append(row)
    return rows


def plot_curves(df: pd.DataFrame, output_dir: Path, log_x: bool = False) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("ic_auc_add_mean", "IC-AUC additive"),
        ("ic_auc_log_mean", "IC-AUC log-ratio"),
        ("scoring_switch_aupa_mid", "GCN/GAT path AUPA"),
        ("seed_gcn_aupa_mid", "Seed-to-seed path AUPA (GCN scoring)"),
        ("seed_gat_aupa_mid", "Seed-to-seed path AUPA (GAT scoring)"),
    ]
    plot_dir = output_dir / ("plots_log_x" if log_x else "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    xscale_label = "log " if log_x else ""

    for metric, label in metrics:
        if metric not in df.columns:
            continue
        plot_df = df.dropna(subset=["avg_time_s", metric])
        if plot_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for subset_num, group in plot_df.groupby("sampling_subset_num", dropna=False):
            group = group.sort_values("sampling_repeats")
            ax.plot(group["avg_time_s"], group[metric], marker="o", label=f"subset={subset_num}")
            for _, row in group.iterrows():
                ax.annotate(f"R={int(row['sampling_repeats'])}", (row["avg_time_s"], row[metric]), fontsize=8)
        ax.set_xlabel("Average run time per seed (s)")
        ax.set_ylabel(label)
        if log_x:
            ax.set_xscale("log")
        ax.set_title(f"Repeat sweep: {label} ({xscale_label}runtime)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"repeat_sweep_{metric}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        numeric_df = plot_df.dropna(subset=["sampling_subset_num_numeric"])
        for repeats, group in numeric_df.groupby("sampling_repeats"):
            group = group.sort_values("sampling_subset_num_numeric")
            ax.plot(group["avg_time_s"], group[metric], marker="o", label=f"repeat={repeats}")
            for _, row in group.iterrows():
                ax.annotate(f"N={int(row['sampling_subset_num_numeric'])}", (row["avg_time_s"], row[metric]), fontsize=8)
        ax.set_xlabel("Average run time per seed (s)")
        ax.set_ylabel(label)
        if log_x:
            ax.set_xscale("log")
        ax.set_title(f"Subset-num sweep: {label} ({xscale_label}runtime)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"subset_num_sweep_{metric}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", type=Path, default=Path("results/stable_hparam_split_count_curves"))
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--model", default="gradient_based_joint_edge_score_stable")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-x", action="store_true", help="Use a logarithmic runtime axis for plots.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.base_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.base_output_dir, args.dataset, args.model)
    df = pd.DataFrame(rows)
    df = df.sort_values(["sampling_subset_num", "sampling_repeats"])
    df.to_csv(output_dir / "stable_hparam_curve_summary.csv", index=False)
    df.to_csv(output_dir / "stable_hparam_curve_summary.tsv", sep="\t", index=False)
    (output_dir / "stable_hparam_curve_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    if not args.no_plots and not df.empty:
        plot_curves(df, output_dir, log_x=args.log_x)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
