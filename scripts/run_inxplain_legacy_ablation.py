#!/usr/bin/env python3
"""Reproduce the original INXplain benchmark while varying only GNN types."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from GS.datasets import DatasetLoader
from GS.metrics import ComplexityMetric, ICAnalysis
from GS.models.gradient_based import GradientBasedGraphSummarization
from GS.models.legacy_downstream import create_legacy_downstream_model


COMBINATIONS = {
    "gcn_gcn": ("gcn", "gcn"),
    "gat_gat": ("gat", "gat"),
    "gat_gcn": ("gat", "gcn"),
    "sage_sage": ("sage", "sage"),
    "sage_gcn": ("sage", "gcn"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_losses_legacy(
    graphs,
    model_type,
    input_dim,
    train_mask,
    val_mask,
    test_mask,
    labels,
    epochs,
    seed,
    device,
):
    losses = []
    for step, graph in enumerate(graphs):
        set_seed(seed + step * 1000)
        model = create_legacy_downstream_model(
            model_type, input_dim=input_dim, device=device
        )
        model.train_model(graph, train_mask, val_mask, labels, epochs=epochs)
        losses.append(model.evaluate(graph, test_mask, labels))
    return losses


def normalize_losses(losses, normalization):
    original_loss = losses[0]
    empty_loss = losses[-1]
    if normalization == "additive":
        denominator = empty_loss - original_loss
        if abs(denominator) <= 1e-10:
            return [1.0] * len(losses)
        return [(empty_loss - loss) / denominator for loss in losses]

    denominator = math.log(original_loss / empty_loss)
    if abs(denominator) <= 1e-10:
        return [1.0 if abs(loss - empty_loss) <= 1e-10 else 0.0 for loss in losses]
    return [math.log(loss / empty_loss) / denominator for loss in losses]


def run_one(args):
    scoring_model, evaluation_model = COMBINATIONS[args.combination]
    device = torch.device(args.device)
    set_seed(args.seed)
    loader = DatasetLoader(root_dir=args.data_dir)
    graph, train_mask, val_mask, test_mask = loader.load_dataset(
        args.dataset, task_type="original"
    )
    graph = loader.preprocess_for_summarization(graph).to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    labels = graph.y.to(device)

    model = GradientBasedGraphSummarization(
        input_dim=graph.x.size(1),
        downstream_model_type=scoring_model,
        hidden_dim=128,
        train_epochs=args.scoring_epochs,
        device=device,
    )
    model.train_mask = train_mask
    model.val_mask = val_mask
    model.labels = labels

    started = time.time()
    graphs = model.summarize(graph, num_steps=args.num_steps)
    summarization_time = time.time() - started
    complexities = ComplexityMetric.compute_list(graphs, graph)

    # The original benchmark trained independent evaluator sequences for Log/Add.
    log_losses = evaluate_losses_legacy(
        graphs, evaluation_model, graph.x.size(1), train_mask, val_mask,
        test_mask, labels, args.evaluation_epochs, args.seed, device
    )
    add_losses = evaluate_losses_legacy(
        graphs, evaluation_model, graph.x.size(1), train_mask, val_mask,
        test_mask, labels, args.evaluation_epochs, args.seed, device
    )
    log_information = normalize_losses(log_losses, "log_ratio")
    add_information = normalize_losses(add_losses, "additive")

    record = {
        "success": True,
        "protocol": "legacy_main_benchmark_2025",
        "combination": args.combination,
        "model": "GradientBasedGraphSummarization",
        "dataset": args.dataset,
        "seed": args.seed,
        "scoring_model": scoring_model,
        "evaluation_model": evaluation_model,
        "num_steps": args.num_steps,
        "scoring_epochs": args.scoring_epochs,
        "evaluation_epochs": args.evaluation_epochs,
        "hidden_dim": 128,
        "directed_edge_loo": True,
        "step_seed_offset": 1000,
        "separate_log_add_training": True,
        "complexities": complexities,
        "information_log_ratio": log_information,
        "information_additive": add_information,
        "ic_auc_log_ratio": ICAnalysis.compute_ic_auc(
            complexities, log_information
        ),
        "ic_auc_additive": ICAnalysis.compute_ic_auc(
            complexities, add_information
        ),
        "threshold_point_log_ratio": ICAnalysis.compute_information_threshold_point(
            complexities, log_information, threshold=0.8
        ),
        "threshold_point_additive": ICAnalysis.compute_information_threshold_point(
            complexities, add_information, threshold=0.8
        ),
        "summarization_time": summarization_time,
        "run_time": time.time() - started,
    }
    path = (
        args.output_dir / args.combination / args.dataset /
        f"seed_{args.seed}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(record, indent=2))
    return record


def aggregate(args):
    rows = []
    for combination, (scoring, evaluation) in COMBINATIONS.items():
        for dataset in args.datasets:
            records = []
            for seed in args.seeds:
                path = args.output_dir / combination / dataset / f"seed_{seed}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if record.get("success"):
                        records.append(record)
            row = {
                "Combination": combination,
                "Scoring_Model": scoring,
                "Evaluation_Model": evaluation,
                "Dataset": dataset,
                "Successful_Seeds": len(records),
                "Expected_Seeds": len(args.seeds),
            }
            for key, label in (
                ("ic_auc_additive", "IC_AUC_Add"),
                ("ic_auc_log_ratio", "IC_AUC_Log"),
                ("threshold_point_additive", "Threshold_Add"),
                ("threshold_point_log_ratio", "Threshold_Log"),
                ("run_time", "Runtime_s"),
            ):
                values = np.asarray([record[key] for record in records], dtype=float)
                row[f"{label}_Mean"] = float(values.mean()) if len(values) else None
                row[f"{label}_StdErr"] = (
                    float(values.std(ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1 else 0.0 if len(values) else None
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "legacy_ablation_summary.tsv", sep="\t", index=False)
    frame.to_csv(args.output_dir / "legacy_ablation_summary.csv", index=False)
    print(frame.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combination", choices=COMBINATIONS)
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer", "KarateClub"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--scoring-epochs", type=int, default=30)
    parser.add_argument("--evaluation-epochs", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/inxplain_legacy_ablation")
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--datasets", nargs="+", default=["Cora", "CiteSeer", "KarateClub"]
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024]
    )
    args = parser.parse_args()
    if not args.aggregate_only and (
        args.combination is None or args.dataset is None or args.seed is None
    ):
        parser.error("--combination, --dataset, and --seed are required")
    return args


def main():
    args = parse_args()
    if args.aggregate_only:
        aggregate(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
