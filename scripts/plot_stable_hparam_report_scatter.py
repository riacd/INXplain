#!/usr/bin/env python3
"""Create separate scatter plots for the Stable hyperparameter report."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[3] / "Projects" / "GS"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from analyze_sampling_repeats_seed_stability import load_path, path_metrics

METRICS = [
    ("ic_auc_add_mean", "Information-content area under curve (additive)"),
    ("ic_auc_log_mean", "Information-content area under curve (log-ratio)"),
    ("scoring_switch_aupa_mid", "Trajectory agreement"),
    ("seed_gcn_aupa_mid", "Trajectory agreement"),
    ("seed_gat_aupa_mid", "Trajectory agreement"),
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "p", "*"]
MORANDI_COLORS = [
    "#A8B3C6",
    "#86899F",
    "#D0D9E6",
    "#8CB49C",
    "#57A692",
    "#DEAB88",
    "#B19B88",
    "#D9C477",
    "#5E3E5B",
]
INXPLAIN_COLOR = "#D3A228"
PRI_GRAPHS_COLOR = "#718699"
PRI_GRAPHS_RUNTIME = 1136.49
PRI_GRAPHS_RUNTIME_STD = 2.62
PRI_GRAPHS_AGREEMENT = 0.4422
PRI_GRAPHS_AGREEMENT_STD = 0.0129


def load_inxplain(summary_path, paths_dir, seeds):
    rows = json.loads(summary_path.read_text())["summary"]
    times = next(row for row in rows if row["model"] == "gradient_based")
    result = {}
    for model in ("gcn", "gat"):
        paths = {
            seed: load_path(paths_dir / f"gradient_based_seed_{seed}_scoring_{model}.json")
            for seed in seeds
        }
        aupas = [
            path_metrics(paths[a], paths[b])["aupa_mid"]
            for a, b in itertools.combinations(seeds, 2)
        ]
        result[model] = {
            "runtime": float(times[f"{model}_time_mean"]),
            "aupa": float(sum(aupas) / len(aupas)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--log-x", action="store_true")
    args = parser.parse_args()

    baseline_root = PROJECT_ROOT / "results/path_stability_original_gradient_baseline/sampling_repeats_1"
    baseline = load_inxplain(
        baseline_root / "summary.json", baseline_root / "paths", [42, 43, 44]
    )
    df = pd.read_csv(args.summary_csv)
    df["sampling_subset_num"] = df["sampling_subset_num"].astype(str)
    df["sampling_repeats"] = pd.to_numeric(df["sampling_repeats"], errors="coerce")
    df = df[df["sampling_repeats"] >= args.min_repeats]
    subsets = sorted(df.sampling_subset_num.unique(), key=lambda x: int(x))
    repeats = sorted(df.sampling_repeats.dropna().unique())

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
    })
    colors = {n: MORANDI_COLORS[i % len(MORANDI_COLORS)] for i, n in enumerate(subsets)}
    markers = {r: MARKERS[i % len(MARKERS)] for i, r in enumerate(repeats)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric, label in METRICS:
        fig, ax = plt.subplots(figsize=(9.4, 6.1))
        plot_df = df.dropna(subset=["avg_time_s", metric])
        for n in subsets:
            group = plot_df[plot_df.sampling_subset_num == n]
            for _, row in group.iterrows():
                ax.scatter(row.avg_time_s, row[metric], marker=markers[row.sampling_repeats],
                           s=52, color=colors[n], edgecolors="black", linewidths=0.9)

        model = {"seed_gcn_aupa_mid": "gcn", "seed_gat_aupa_mid": "gat"}.get(metric)
        if model:
            point = baseline[model]
            ax.scatter(point["runtime"], point["aupa"], marker="D", s=120,
                       color=INXPLAIN_COLOR, edgecolors="black", linewidths=1.3, zorder=6)
            ax.annotate("INXPlain", (point["runtime"], point["aupa"]),
                        xytext=(-8, -10), textcoords="offset points", fontsize=13,
                        fontweight="bold", ha="right", va="top")
            ax.scatter(PRI_GRAPHS_RUNTIME, PRI_GRAPHS_AGREEMENT, marker="*", s=260,
                       color=PRI_GRAPHS_COLOR, edgecolors="black", linewidths=1.5,
                       zorder=6)
            ax.annotate("PRI-Graphs", (PRI_GRAPHS_RUNTIME, PRI_GRAPHS_AGREEMENT),
                        xytext=(12, -14), textcoords="offset points", fontsize=13,
                        fontweight="bold", ha="left", va="top",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                              "pad": 1.5})

        ax.set(xlabel="Run time (s)", ylabel=label)
        seed_title = {
            "seed_gcn_aupa_mid": "Seed-to-seed trajectory agreement (GCN)",
            "seed_gat_aupa_mid": "Seed-to-seed trajectory agreement (GAT)",
        }.get(metric)
        if seed_title:
            ax.set_title(seed_title)
        if args.log_x:
            ax.set_xscale("log")
        ax.grid(False)
        segmentation_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[n],
                   markeredgecolor="black", markersize=7, label=n)
            for n in subsets
        ]
        repetition_handles = [
            Line2D([0], [0], marker=markers[r], color="black", linestyle="None",
                   markersize=7, label=str(int(r)))
            for r in repeats
        ]
        segmentation_legend = fig.legend(
            handles=segmentation_handles, title="No. segmentation", loc="upper left",
            bbox_to_anchor=(0.70, 0.92), frameon=False, fontsize=13, title_fontsize=13,
            alignment="left",
        )
        segmentation_legend.get_title().set_ha("left")
        segmentation_legend._legend_box.align = "left"
        repetition_legend = fig.legend(
            handles=repetition_handles, title="No. repetition", loc="upper left",
            bbox_to_anchor=(0.70, 0.42), frameon=False, fontsize=13, title_fontsize=13,
            alignment="left",
        )
        repetition_legend.get_title().set_ha("left")
        repetition_legend._legend_box.align = "left"
        fig.subplots_adjust(right=0.68)
        suffix = "_log_x" if args.log_x else ""
        stem = f"report_{metric.removesuffix('_mean')}{suffix}"
        fig.savefig(args.output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    for model, point in baseline.items():
        print(f"INXPlain {model.upper()}: runtime={point['runtime']:.6f}s, seed AUPA={point['aupa']:.6f}")


if __name__ == "__main__":
    main()
