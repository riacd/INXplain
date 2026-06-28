#!/usr/bin/env python3
"""Aggregate repeated-experiment JSON files into fixed TSV/Markdown summaries."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def metric_with_stderr(mean, stderr) -> str:
    if mean is None:
        return ""
    if stderr is None:
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {stderr:.6f}"


def markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(df: pd.DataFrame, output_path: Path, title: str):
    columns = [
        "Dataset", "Task", "Downstream", "Model", "Success_Rate",
        "IC_AUC_Add", "IC_AUC_Log", "Threshold_Add", "Threshold_Log",
        "Total_Time_s", "Avg_Time_s", "Config",
    ]
    with open(output_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for _, row in df[columns].iterrows():
            f.write("| " + " | ".join(markdown_cell(row[col]) for col in columns) + " |\n")


def row_from_result(
    dataset: str,
    model_name: str,
    model_results: Dict[str, Any],
    config: Dict[str, Any],
    source_file: str,
) -> Dict[str, Any]:
    stats = model_results.get("statistics", {})
    num_successful = stats.get("num_successful", 0)
    num_repeats = config.get("num_repeats", len(model_results.get("runs", [])))

    ic_auc_add = stats.get("ic_auc_additive", {})
    ic_auc_log = stats.get("ic_auc_log_ratio", {})
    threshold_add = stats.get("threshold_point_additive", {})
    threshold_log = stats.get("threshold_point_log_ratio", {})
    model_kwargs = config.get("model_kwargs", {})
    if model_name.startswith("networkit_") or model_name == "pri_graphs":
        model_kwargs = {}

    config_summary = {
        "num_steps": config.get("num_steps"),
        "epochs": config.get("epochs"),
        "seeds": config.get("seeds"),
        "device": config.get("device"),
        "downstream_kwargs": config.get("downstream_kwargs", {}),
        "model_kwargs": model_kwargs,
        "disable_adaptive_epochs": config.get("disable_adaptive_epochs"),
        "source_file": source_file,
    }

    return {
        "Dataset": dataset,
        "Task": config.get("task"),
        "Downstream": config.get("downstream"),
        "Model": model_name,
        "Success_Rate": f"{num_successful}/{num_repeats}",
        "IC_AUC_Add_Mean": ic_auc_add.get("mean"),
        "IC_AUC_Add_StdErr": ic_auc_add.get("stderr"),
        "IC_AUC_Log_Mean": ic_auc_log.get("mean"),
        "IC_AUC_Log_StdErr": ic_auc_log.get("stderr"),
        "Threshold_Add_Mean": threshold_add.get("mean"),
        "Threshold_Add_StdErr": threshold_add.get("stderr"),
        "Threshold_Log_Mean": threshold_log.get("mean"),
        "Threshold_Log_StdErr": threshold_log.get("stderr"),
        "Total_Time_s": stats.get("total_run_time"),
        "Avg_Time_s": stats.get("avg_run_time"),
        "IC_AUC_Add": metric_with_stderr(ic_auc_add.get("mean"), ic_auc_add.get("stderr")),
        "IC_AUC_Log": metric_with_stderr(ic_auc_log.get("mean"), ic_auc_log.get("stderr")),
        "Threshold_Add": metric_with_stderr(threshold_add.get("mean"), threshold_add.get("stderr")),
        "Threshold_Log": metric_with_stderr(threshold_log.get("mean"), threshold_log.get("stderr")),
        "Config": json.dumps(config_summary, sort_keys=True, default=str),
    }


def compute_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "stderr": None, "n": 0}
    series = pd.Series(values, dtype=float)
    std = float(series.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": float(series.mean()),
        "std": std,
        "stderr": std / math.sqrt(len(values)) if len(values) > 1 else 0.0,
        "n": len(values),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def combine_model_results(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [run for run in runs if run.get("success", False)]

    def values(name: str) -> List[float]:
        return [run[name] for run in successful if run.get(name) is not None]

    return {
        "runs": runs,
        "statistics": {
            "num_successful": len(successful),
            "num_failed": len(runs) - len(successful),
            "ic_auc_additive": compute_stats(values("ic_auc_additive")),
            "ic_auc_log_ratio": compute_stats(values("ic_auc_log_ratio")),
            "threshold_point_additive": compute_stats(values("threshold_point_additive")),
            "threshold_point_log_ratio": compute_stats(values("threshold_point_log_ratio")),
            "avg_run_time": float(pd.Series([run["run_time"] for run in runs]).mean()),
            "total_run_time": float(sum(run["run_time"] for run in runs)),
        },
    }


def aggregate(input_dir: Path) -> pd.DataFrame:
    combined: Dict[Tuple[str, str], Dict[str, Any]] = {}
    json_files = sorted(input_dir.glob("multi_dataset_repeated_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No multi_dataset_repeated_results_*.json files in {input_dir}")

    for json_file in json_files:
        with open(json_file) as f:
            payload = json.load(f)

        config = payload.get("config", {})
        for dataset, models_data in payload.get("results", {}).items():
            for model_name, model_results in models_data.items():
                key = (dataset, model_name)
                entry = combined.setdefault(key, {
                    "config": dict(config),
                    "runs": {},
                    "source_files": [],
                })
                entry["source_files"].append(str(json_file))
                for run in model_results.get("runs", []):
                    run_key = (run.get("seed"), run.get("repeat_idx"))
                    entry["runs"][run_key] = run

    rows = []
    for (dataset, model_name), entry in combined.items():
        runs = list(entry["runs"].values())
        runs.sort(key=lambda run: (run.get("seed", 0), run.get("repeat_idx", 0)))
        config = entry["config"]
        config["seeds"] = [run.get("seed") for run in runs]
        config["num_repeats"] = len(runs)
        rows.append(row_from_result(
            dataset,
            model_name,
            combine_model_results(runs),
            config,
            ";".join(entry["source_files"]),
        ))

    df = pd.DataFrame(rows)
    return df.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="results/ogbn_arxiv_original")
    parser.add_argument("--output-prefix", default="ogbn_arxiv_original_summary")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    df = aggregate(input_dir)

    tsv_path = input_dir / f"{args.output_prefix}.tsv"
    md_path = input_dir / f"{args.output_prefix}.md"
    df.to_csv(tsv_path, sep="\t", index=False)
    write_markdown(df, md_path, args.output_prefix)

    print(f"Wrote {tsv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
