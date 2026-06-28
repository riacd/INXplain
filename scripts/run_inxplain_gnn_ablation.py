#!/usr/bin/env python3
"""Run and aggregate the resumable INXplain GNN architecture ablation."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multi_dataset_repeated_experiments import run_single_experiment


COMBINATIONS = {
    'gcn_gcn': ('gcn', 'gcn'),
    'gat_gat': ('gat', 'gat'),
    'gat_gcn': ('gat', 'gcn'),
    'sage_sage': ('sage', 'sage'),
    'sage_gcn': ('sage', 'gcn'),
    'h2gcn_h2gcn': ('h2gcn', 'h2gcn'),
    'gcnii_gcnii': ('gcnii', 'gcnii'),
}
METRICS = (
    'ic_auc_additive',
    'ic_auc_log_ratio',
    'threshold_point_additive',
    'threshold_point_log_ratio',
    'original_accuracy',
    'empty_accuracy',
    'run_time',
)


def webkb_downstream_kwargs(model_type: str, dataset: str) -> Dict[str, Any]:
    if model_type in ('h2gcn', 'gcnii'):
        return {'dataset_name': dataset}
    return {}


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with open(temporary, 'w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
    os.replace(temporary, path)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def combination_config(args, combination: str) -> Dict[str, Any]:
    scoring_model, evaluation_model = COMBINATIONS[combination]
    return {
        'combination': combination,
        'model': 'gradient_based',
        'scoring_model': scoring_model,
        'evaluation_model': evaluation_model,
        'task': args.task,
        'num_steps': args.num_steps,
        'evaluation_epochs': args.epochs,
        'scoring_epochs': args.scoring_train_epochs,
        'device': args.device,
        'disable_adaptive_epochs': True,
    }


def validate_existing_config(path: Path, expected: Dict[str, Any]) -> None:
    if not path.exists():
        atomic_write_json(path, expected)
        return
    existing = load_json(path)
    # Older smoke runs stored the requested dataset/seed list in this
    # combination-level file. They are run-level dimensions, not config.
    existing.pop('datasets', None)
    existing.pop('seeds', None)
    if existing != expected:
        raise RuntimeError(
            f"Existing configuration differs from requested run: {path}. "
            "Use a different output directory."
        )


def run_combination(args, combination: str) -> bool:
    scoring_model, evaluation_model = COMBINATIONS[combination]
    combination_dir = args.output_dir / combination
    task_dir = combination_dir / 'runs' / args.task
    config = combination_config(args, combination)
    validate_existing_config(task_dir / 'config.json', config)
    all_successful = True

    for dataset in args.datasets:
        for seed in args.seeds:
            record_path = task_dir / dataset / f'seed_{seed}.json'
            if record_path.exists() and load_json(record_path).get('success'):
                print(
                    f"SKIP {combination} {args.task} {dataset} seed={seed}: "
                    "already successful"
                )
                continue

            print(f"RUN {combination} {args.task} {dataset} seed={seed}", flush=True)
            started_at = time.time()
            result = run_single_experiment(
                model_name='gradient_based',
                dataset=dataset,
                task=args.task,
                downstream=evaluation_model,
                num_steps=args.num_steps,
                epochs=args.epochs,
                seed=seed,
                device=args.device,
                model_kwargs={
                    'downstream_model_type': scoring_model,
                    'train_epochs': args.scoring_train_epochs,
                    'dataset_name': dataset,
                },
                downstream_kwargs=webkb_downstream_kwargs(evaluation_model, dataset),
                disable_adaptive_epochs=True,
                min_accuracy_over_majority=(
                    -1.0 if args.allow_noninformative_reference else 0.0
                ),
                require_informative_reference=(
                    not args.allow_noninformative_reference
                ),
                output_dir=str(combination_dir / 'benchmark'),
            )
            record = {
                **config,
                'dataset': dataset,
                'seed': seed,
                'started_at': started_at,
                'finished_at': time.time(),
                **result,
            }
            record['run_time'] = record['finished_at'] - started_at
            atomic_write_json(record_path, record)
            if not result.get('success'):
                all_successful = False
                print(
                    f"FAILED {combination} {args.task} {dataset} seed={seed}: "
                    f"{result.get('error')}"
                )

    if not args.no_aggregate:
        aggregate_results(
            args.output_dir,
            args.aggregate_datasets,
            args.aggregate_seeds,
            args.aggregate_combinations,
            args.aggregate_tasks,
        )
    return all_successful


def metric_stats(records: Iterable[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    values = [float(record[metric]) for record in records]
    if not values or not all(math.isfinite(value) for value in values):
        return {'mean': None, 'stderr': None}
    array = np.asarray(values, dtype=float)
    stderr = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    return {'mean': float(array.mean()), 'stderr': stderr}


def format_stats(stats: Dict[str, Any]) -> str:
    if stats['mean'] is None:
        return ''
    return f"{stats['mean']:.6f} +/- {stats['stderr']:.6f}"


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        'Combination', 'Scoring_Model', 'Evaluation_Model', 'Task', 'Dataset',
        'Successful_Seeds', 'IC_AUC_Add', 'IC_AUC_Log', 'Threshold_Add',
        'Threshold_Log', 'Original_Accuracy', 'Runtime_s', 'Complete',
    ]
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with open(temporary, 'w') as handle:
        handle.write('# INXplain GNN Architecture Ablation\n\n')
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for _, row in frame.iterrows():
            handle.write('| ' + ' | '.join(str(row[column]) for column in columns) + ' |\n')
    os.replace(temporary, path)


def atomic_write_frame(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    frame.to_csv(temporary, index=False, **kwargs)
    os.replace(temporary, path)


def aggregate_results(
    output_dir: Path,
    datasets: List[str],
    seeds: List[int],
    combinations: List[str] = None,
    tasks: List[str] = None,
) -> pd.DataFrame:
    rows = []
    selected = combinations or list(COMBINATIONS)
    selected_tasks = tasks or ['original']
    for combination in selected:
        scoring_model, evaluation_model = COMBINATIONS[combination]
        for task in selected_tasks:
            for dataset in datasets:
                records = []
                for seed in seeds:
                    path = (
                        output_dir / combination / 'runs' / task /
                        dataset / f'seed_{seed}.json'
                    )
                    if path.exists():
                        record = load_json(path)
                        if record.get('success'):
                            records.append(record)

                stats = {metric: metric_stats(records, metric) for metric in METRICS}
                finite = all(stats[metric]['mean'] is not None for metric in METRICS)
                row = {
                    'Combination': combination,
                    'Scoring_Model': scoring_model,
                    'Evaluation_Model': evaluation_model,
                    'Task': task,
                    'Dataset': dataset,
                    'Successful_Seeds': len(records),
                    'Expected_Seeds': len(seeds),
                    'Success_Rate': f"{len(records)}/{len(seeds)}",
                    'Seeds': ' '.join(str(record['seed']) for record in records),
                    'Complete': len(records) == len(seeds) and finite,
                }
                for metric in METRICS:
                    label = {
                        'ic_auc_additive': 'IC_AUC_Add',
                        'ic_auc_log_ratio': 'IC_AUC_Log',
                        'threshold_point_additive': 'Threshold_Add',
                        'threshold_point_log_ratio': 'Threshold_Log',
                        'original_accuracy': 'Original_Accuracy',
                        'empty_accuracy': 'Empty_Accuracy',
                        'run_time': 'Runtime_s',
                    }[metric]
                    row[f'{label}_Mean'] = stats[metric]['mean']
                    row[f'{label}_StdErr'] = stats[metric]['stderr']
                    row[label] = format_stats(stats[metric])
                rows.append(row)

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_frame(frame, output_dir / 'inxplain_gnn_ablation_summary.tsv', sep='\t')
    atomic_write_frame(frame, output_dir / 'inxplain_gnn_ablation_summary.csv')
    write_markdown(frame, output_dir / 'inxplain_gnn_ablation_summary.md')
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combination', choices=list(COMBINATIONS) + ['all'], default='all')
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer', 'PubMed', 'KarateClub'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument(
        '--task',
        choices=['original', 'degree', 'degree_centrality', 'pagerank',
                 'closeness_centrality'],
        default='original',
    )
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--scoring-train-epochs', type=int, default=200)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output-dir', type=Path, default=Path('results/inxplain_gnn_ablation'))
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--no-aggregate', action='store_true')
    parser.add_argument(
        '--require-single-run',
        action='store_true',
        help='Fail unless exactly one combination, dataset, and seed are requested.',
    )
    parser.add_argument(
        '--aggregate-datasets',
        nargs='+',
        default=['Cora', 'CiteSeer', 'PubMed', 'KarateClub'],
    )
    parser.add_argument(
        '--aggregate-seeds',
        nargs='+',
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument(
        '--aggregate-tasks',
        nargs='+',
        choices=['original', 'degree', 'degree_centrality', 'pagerank',
                 'closeness_centrality'],
        default=['original'],
    )
    parser.add_argument(
        '--aggregate-combinations',
        nargs='+',
        choices=list(COMBINATIONS),
    )
    parser.add_argument('--allow-noninformative-reference', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate_only:
        frame = aggregate_results(
            args.output_dir,
            args.aggregate_datasets,
            args.aggregate_seeds,
            args.aggregate_combinations,
            args.aggregate_tasks,
        )
        return 0 if frame['Complete'].all() else 1

    if args.require_single_run and (
        args.combination == 'all'
        or len(args.datasets) != 1
        or len(args.seeds) != 1
    ):
        raise SystemExit(
            '--require-single-run requires one combination, one dataset, and one seed'
        )

    combinations = COMBINATIONS if args.combination == 'all' else [args.combination]
    statuses = [run_combination(args, combination) for combination in combinations]
    success = all(statuses)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
