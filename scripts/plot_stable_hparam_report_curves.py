#!/usr/bin/env python3
"""Create multi-panel report curves for Stable hyperparameter sweeps."""

import argparse
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D


METRICS = [
    ("ic_auc_add_mean", "IC-AUC additive"),
    ("ic_auc_log_mean", "IC-AUC log-ratio"),
    ("scoring_switch_aupa_mid", "GCN/GAT path AUPA"),
    ("seed_gcn_aupa_mid", "Seed-to-seed AUPA (GCN)"),
    ("seed_gat_aupa_mid", "Seed-to-seed AUPA (GAT)"),
]

REPEAT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "p", "*"]


def subset_sort_key(value: str) -> tuple:
    if value == "all":
        return (1, 0)
    return (0, int(value))


def setup_axes(plt):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=False)
    axes = axes.flatten()
    for ax in axes[len(METRICS):]:
        ax.axis("off")
    return fig, axes


def mark_key_points(ax, df: pd.DataFrame, metric: str) -> None:
    recommended = df[(df["sampling_subset_num"] == "4") & (df["sampling_repeats"] == 5)]
    best_ic = df[(df["sampling_subset_num"] == "6") & (df["sampling_repeats"] == 5)]

    for row, marker, label in [
        (recommended, "*", "recommended N=4,R=5"),
        (best_ic, "X", "best IC N=6,R=5"),
    ]:
        if row.empty or metric not in row.columns:
            continue
        row = row.iloc[0]
        ax.scatter(
            [row["avg_time_s"]],
            [row[metric]],
            marker=marker,
            s=130,
            color="black",
            zorder=5,
            label=label,
        )


def prepare_plot_df(df: pd.DataFrame, min_repeats: int) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["sampling_repeats"] = pd.to_numeric(plot_df["sampling_repeats"], errors="coerce")
    plot_df["sampling_subset_num_numeric"] = pd.to_numeric(
        plot_df["sampling_subset_num_numeric"], errors="coerce"
    )
    plot_df = plot_df[plot_df["sampling_repeats"] >= min_repeats]
    return plot_df


def marker_for_repeats(repeats_values):
    return {
        repeats: REPEAT_MARKERS[idx % len(REPEAT_MARKERS)]
        for idx, repeats in enumerate(sorted(repeats_values))
    }


def plot_runtime_tradeoff(
    df: pd.DataFrame,
    output_dir: Path,
    log_x: bool = False,
    min_repeats: int = 3,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = setup_axes(plt)
    plot_base = prepare_plot_df(df, min_repeats)
    subsets = sorted(plot_base["sampling_subset_num"].astype(str).unique(), key=subset_sort_key)
    repeats_values = sorted(plot_base["sampling_repeats"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    colors = {subset: cmap(idx % 10) for idx, subset in enumerate(subsets)}
    markers = marker_for_repeats(repeats_values)

    for ax, (metric, label) in zip(axes, METRICS):
        plot_df = plot_base.dropna(subset=["avg_time_s", metric])
        for subset in subsets:
            group = plot_df[plot_df["sampling_subset_num"].astype(str) == subset]
            if group.empty:
                continue
            group = group.sort_values("sampling_repeats")
            ax.plot(
                group["avg_time_s"],
                group[metric],
                linewidth=1.8,
                color=colors[subset],
                label=f"N={subset}",
            )
            for _, row in group.iterrows():
                ax.scatter(
                    [row["avg_time_s"]],
                    [row[metric]],
                    marker=markers[row["sampling_repeats"]],
                    s=48,
                    color=colors[subset],
                    edgecolors="black",
                    linewidths=0.35,
                    zorder=4,
                )
        mark_key_points(ax, plot_df, metric)
        ax.set_title(label)
        ax.set_xlabel("RUN TIME (s)")
        ax.set_ylabel(label)
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.25)

    n_handles = [
        Line2D([0], [0], color=colors[subset], linewidth=2, label=f"N={subset}")
        for subset in subsets
    ]
    r_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[repeats],
            color="black",
            linestyle="None",
            markersize=7,
            label=f"R={int(repeats)}",
        )
        for repeats in repeats_values
    ]
    fig.legend(
        n_handles + r_handles,
        [handle.get_label() for handle in n_handles + r_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
    )
    xscale_label = "log-runtime" if log_x else "runtime"
    fig.suptitle(
        f"Stable hyperparameter curves (R >= {min_repeats}, {xscale_label})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    suffix = "_log_x" if log_x else ""
    fig.savefig(output_dir / f"report_runtime_tradeoff_all_metrics{suffix}.png", dpi=220)
    fig.savefig(output_dir / f"report_runtime_tradeoff_all_metrics{suffix}.pdf")
    plt.close(fig)


def plot_repeat_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    log_x: bool = False,
    min_repeats: int = 3,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = setup_axes(plt)
    plot_base = prepare_plot_df(df, min_repeats)
    subsets = sorted(plot_base["sampling_subset_num"].astype(str).unique(), key=subset_sort_key)
    repeats_values = sorted(plot_base["sampling_repeats"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    colors = {subset: cmap(idx % 10) for idx, subset in enumerate(subsets)}
    markers = marker_for_repeats(repeats_values)

    for ax, (metric, label) in zip(axes, METRICS):
        plot_df = plot_base.dropna(subset=["avg_time_s", metric])
        for subset in subsets:
            group = plot_df[plot_df["sampling_subset_num"].astype(str) == subset]
            if group.empty:
                continue
            group = group.sort_values("sampling_repeats")
            ax.plot(
                group["avg_time_s"],
                group[metric],
                linewidth=1.8,
                color=colors[subset],
                label=f"N={subset}",
            )
            for _, row in group.iterrows():
                ax.scatter(
                    [row["avg_time_s"]],
                    [row[metric]],
                    marker=markers[row["sampling_repeats"]],
                    s=48,
                    color=colors[subset],
                    edgecolors="black",
                    linewidths=0.35,
                    zorder=4,
                )
        mark_key_points(ax, plot_df, metric)
        ax.set_title(label)
        ax.set_xlabel("RUN TIME (s)")
        ax.set_ylabel(label)
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.25)

    n_handles = [
        Line2D([0], [0], color=colors[subset], linewidth=2, label=f"N={subset}")
        for subset in subsets
    ]
    r_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[repeats],
            color="black",
            linestyle="None",
            markersize=7,
            label=f"R={int(repeats)}",
        )
        for repeats in repeats_values
    ]
    fig.legend(
        n_handles + r_handles,
        [handle.get_label() for handle in n_handles + r_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
    )
    xscale_label = "log-runtime" if log_x else "runtime"
    fig.suptitle(
        f"Stable hyperparameter curves: repeat sweep (R >= {min_repeats}, {xscale_label})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    suffix = "_log_x" if log_x else ""
    fig.savefig(output_dir / f"report_repeat_sweep_all_metrics{suffix}.png", dpi=220)
    fig.savefig(output_dir / f"report_repeat_sweep_all_metrics{suffix}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/stable_hparam_split_count_curves/stable_hparam_curve_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-x", action="store_true", help="Use a logarithmic runtime axis.")
    parser.add_argument("--min-repeats", type=int, default=3, help="Minimum R value to plot.")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    df["sampling_subset_num"] = df["sampling_subset_num"].astype(str)
    output_dir = args.output_dir or args.summary_csv.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_runtime_tradeoff(df, output_dir, log_x=args.log_x, min_repeats=args.min_repeats)
    plot_repeat_sweep(df, output_dir, log_x=args.log_x, min_repeats=args.min_repeats)
    print(f"Wrote report curves to {output_dir}")


if __name__ == "__main__":
    main()
