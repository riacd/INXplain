#!/usr/bin/env python3
"""Submit one PBS job per INXplain ablation combination, dataset, and seed."""

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path


DEFAULT_COMBINATIONS = ('gcn_gcn', 'gat_gat', 'gat_gcn', 'sage_sage', 'sage_gcn')
WEBKB_COMBINATIONS = ('h2gcn_h2gcn', 'gcnii_gcnii')
COMBINATIONS = DEFAULT_COMBINATIONS + WEBKB_COMBINATIONS
DEFAULT_DATASETS = ('Cora', 'CiteSeer', 'PubMed', 'KarateClub')
WEBKB_DATASETS = ('Cornell', 'Texas', 'Wisconsin')
DATASETS = DEFAULT_DATASETS + WEBKB_DATASETS
TASKS = ('original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality')
WEBKB_FOUR_LABEL_TASKS = (
    'degree',
    'degree_centrality',
    'pagerank',
    'closeness_centrality',
)
SEEDS = (42, 43, 44, 45, 46)


def successful_record(
    output_dir: Path,
    combination: str,
    task: str,
    dataset: str,
    seed: int,
) -> bool:
    path = output_dir / combination / 'runs' / task / dataset / f'seed_{seed}.json'
    if not path.exists():
        return False
    with open(path) as handle:
        return bool(json.load(handle).get('success'))


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {'submitted': [], 'skipped': [], 'batches': []}
    with open(path) as handle:
        manifest = json.load(handle)
    manifest.setdefault('submitted', [])
    manifest.setdefault('skipped', [])
    manifest.setdefault('batches', [])
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--combinations', nargs='+', choices=COMBINATIONS)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=['original'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--queues', nargs='+', choices=['ai4090', 'ai4090d'], default=['ai4090', 'ai4090d'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pbs-script', default='scripts/inxplain_gnn_ablation_4090d.pbs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--scoring-train-epochs', type=int, default=200)
    parser.add_argument('--extra-args', default='--allow-noninformative-reference')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resubmit-successful', action='store_true')
    parser.add_argument(
        '--webkb',
        action='store_true',
        help='Use the H2GCN/GCNII WebKB run matrix and output directory.',
    )
    parser.add_argument(
        '--webkb-gcnii-four-labels',
        action='store_true',
        help='Use only gcnii_gcnii on WebKB degree/centrality label tasks.',
    )
    parser.add_argument(
        '--pubmed-last',
        action='store_true',
        help='Submit non-PubMed jobs first and hold PubMed until all of them pass.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.webkb_gcnii_four_labels:
        args.combinations = args.combinations or ('gcnii_gcnii',)
        args.datasets = args.datasets or WEBKB_DATASETS
        args.tasks = args.tasks if args.tasks != ['original'] else WEBKB_FOUR_LABEL_TASKS
        args.output_dir = args.output_dir or Path(
            'results/inxplain_webkb_gcnii_four_labels/inxplain'
        )
        args.epochs = args.epochs if args.epochs != 30 else 200
    elif args.webkb:
        args.combinations = args.combinations or WEBKB_COMBINATIONS
        args.datasets = args.datasets or WEBKB_DATASETS
        args.output_dir = args.output_dir or Path(
            'results/inxplain_webkb_pytorch_gnn_ablation'
        )
    else:
        args.combinations = args.combinations or DEFAULT_COMBINATIONS
        args.datasets = args.datasets or DEFAULT_DATASETS
        args.output_dir = args.output_dir or Path('results/inxplain_gnn_ablation')
    output_dir = args.output_dir.resolve()
    submitted = []
    skipped = []
    queue_cycle = itertools.cycle(args.queues)

    run_specs = list(itertools.product(
        args.combinations, args.tasks, args.datasets, args.seeds
    ))
    if args.pubmed_last:
        run_specs.sort(key=lambda spec: spec[2] == 'PubMed')

    prerequisite_job_ids = []
    for combination, task, dataset, seed in run_specs:
        if not args.resubmit_successful and successful_record(
            output_dir, combination, task, dataset, seed
        ):
            skipped.append((combination, task, dataset, seed))
            continue

        queue = next(queue_cycle)
        variables = ','.join((
            f'COMBINATION={combination}',
            f'TASK={task}',
            f'DATASET={dataset}',
            f'SEED={seed}',
            f'OUTPUT_DIR={output_dir}',
            f'EPOCHS={args.epochs}',
            f'SCORING_TRAIN_EPOCHS={args.scoring_train_epochs}',
            f'EXTRA_ARGS={args.extra_args}',
        ))
        combination_tag = combination.replace('_', '')[:6]
        task_tag = ''.join(part[0] for part in task.split('_'))[:4]
        job_name = f"inx_{combination_tag}_{task_tag}_{dataset[:4]}_{seed}"
        command = [
            'qsub', '-q', queue, '-N', job_name,
            '-v', variables, args.pbs_script,
        ]
        if args.pubmed_last and dataset == 'PubMed' and prerequisite_job_ids:
            dependency = 'depend=afterok:' + ':'.join(prerequisite_job_ids)
            command[1:1] = ['-W', dependency]
        if args.dry_run:
            job_id = 'DRY_RUN'
            print(' '.join(command))
        else:
            job_id = subprocess.check_output(command, text=True).strip()
            print(f'{job_id}\t{queue}\t{combination}\t{task}\t{dataset}\t{seed}')
        submitted.append({
            'job_id': job_id,
            'queue': queue,
            'combination': combination,
            'task': task,
            'dataset': dataset,
            'seed': seed,
        })
        if dataset != 'PubMed':
            prerequisite_job_ids.append(
                job_id if job_id != 'DRY_RUN' else f'DRY_{len(prerequisite_job_ids)}'
            )

    manifest = output_dir / 'submission_manifest.json'
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_manifest = load_manifest(manifest)
        existing_manifest['submitted'].extend(submitted)
        existing_manifest['skipped'].extend(skipped)
        existing_manifest['batches'].append({
            'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'combinations': list(args.combinations),
            'tasks': list(args.tasks),
            'datasets': list(args.datasets),
            'seeds': list(args.seeds),
            'submitted_count': len(submitted),
            'skipped_count': len(skipped),
        })
        with open(manifest, 'w') as handle:
            json.dump(existing_manifest, handle, indent=2)
        print(f'Manifest: {manifest}')
    print(f'Submitted: {len(submitted)}; skipped successful: {len(skipped)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
