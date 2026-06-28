#!/usr/bin/env python3
"""Aggregate WebKB GCNII four-label INXplain and baseline summaries."""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd


TASKS = ('degree', 'degree_centrality', 'pagerank', 'closeness_centrality')
DATASETS = ('Cornell', 'Texas', 'Wisconsin')
METRIC_COLUMN = 'IC_AUC_Add_Mean'


def atomic_write_frame(frame: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    frame.to_csv(temporary, index=False, **kwargs)
    os.replace(temporary, path)


def markdown_cell(value) -> str:
    if pd.isna(value):
        return ''
    return str(value).replace('|', '\\|').replace('\n', ' ')


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    display_columns = [
        'Task', 'Dataset', 'Evaluation_Model', 'Rank', 'Method', 'Source',
        'Success_Rate', 'IC_AUC_Add', 'IC_AUC_Log', 'Threshold_Add',
        'Threshold_Log', 'Original_Accuracy', 'Empty_Accuracy', 'Runtime_s',
    ]
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with open(temporary, 'w') as handle:
        handle.write('# WebKB GCNII Four-Label Summary\n\n')
        handle.write('| ' + ' | '.join(display_columns) + ' |\n')
        handle.write('| ' + ' | '.join(['---'] * len(display_columns)) + ' |\n')
        for _, row in frame.iterrows():
            handle.write(
                '| ' + ' | '.join(markdown_cell(row[col]) for col in display_columns) + ' |\n'
            )
    os.replace(temporary, path)


def load_inxplain(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, sep='\t')
    if frame.empty:
        return frame
    frame = frame.copy()
    frame['Source'] = 'inxplain'
    frame['Method'] = frame['Combination']
    frame['Evaluation_Model'] = frame['Evaluation_Model'].str.lower()
    return frame


def load_baselines(root: Path, tasks: Iterable[str], datasets: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for task in tasks:
        for dataset in datasets:
            path = root / f'webkb_baselines_{task}_{dataset}_gcnii.tsv'
            if not path.exists():
                continue
            frame = pd.read_csv(path, sep='\t')
            if frame.empty:
                continue
            frame = frame.copy()
            frame['Source'] = 'baseline'
            frame['Method'] = frame['Model']
            frame['Evaluation_Model'] = frame['Downstream'].str.lower()
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    desired = [
        'Task', 'Dataset', 'Evaluation_Model', 'Method', 'Source',
        'Success_Rate', 'IC_AUC_Add_Mean', 'IC_AUC_Add_StdErr',
        'IC_AUC_Log_Mean', 'IC_AUC_Log_StdErr',
        'Threshold_Add_Mean', 'Threshold_Add_StdErr',
        'Threshold_Log_Mean', 'Threshold_Log_StdErr',
        'Original_Accuracy_Mean', 'Original_Accuracy_StdErr',
        'Empty_Accuracy_Mean', 'Empty_Accuracy_StdErr',
        'Runtime_s_Mean', 'Runtime_s_StdErr',
        'IC_AUC_Add', 'IC_AUC_Log', 'Threshold_Add', 'Threshold_Log',
        'Original_Accuracy', 'Empty_Accuracy', 'Runtime_s',
    ]
    for column in desired:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[desired].copy()


def aggregate(args) -> pd.DataFrame:
    inxplain = load_inxplain(args.inxplain_dir / 'inxplain_gnn_ablation_summary.tsv')
    baselines = load_baselines(args.baseline_dir, args.tasks, args.datasets)
    frames = [
        normalize_columns(frame)
        for frame in (inxplain, baselines)
        if not frame.empty
    ]
    if not frames:
        raise FileNotFoundError('No INXplain or baseline summary files found.')

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        combined['Task'].isin(args.tasks)
        & combined['Dataset'].isin(args.datasets)
        & (combined['Evaluation_Model'] == 'gcnii')
    ].copy()
    combined[METRIC_COLUMN] = pd.to_numeric(combined[METRIC_COLUMN], errors='coerce')
    combined.sort_values(
        ['Task', 'Dataset', 'Evaluation_Model', METRIC_COLUMN, 'Method'],
        ascending=[True, True, True, False, True],
        inplace=True,
        na_position='last',
    )
    combined['Rank'] = (
        combined.groupby(['Task', 'Dataset', 'Evaluation_Model'])[METRIC_COLUMN]
        .rank(method='min', ascending=False)
    )
    combined['Rank'] = combined['Rank'].astype('Int64')

    columns = [
        'Task', 'Dataset', 'Evaluation_Model', 'Rank', 'Method', 'Source',
        'Success_Rate', 'IC_AUC_Add_Mean', 'IC_AUC_Add_StdErr',
        'IC_AUC_Log_Mean', 'IC_AUC_Log_StdErr',
        'Threshold_Add_Mean', 'Threshold_Add_StdErr',
        'Threshold_Log_Mean', 'Threshold_Log_StdErr',
        'Original_Accuracy_Mean', 'Original_Accuracy_StdErr',
        'Empty_Accuracy_Mean', 'Empty_Accuracy_StdErr',
        'Runtime_s_Mean', 'Runtime_s_StdErr',
        'IC_AUC_Add', 'IC_AUC_Log', 'Threshold_Add', 'Threshold_Log',
        'Original_Accuracy', 'Empty_Accuracy', 'Runtime_s',
    ]
    return combined[columns]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--root',
        type=Path,
        default=Path('results/inxplain_webkb_gcnii_four_labels'),
    )
    parser.add_argument('--inxplain-dir', type=Path)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--tasks', nargs='+', default=list(TASKS), choices=TASKS)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=DATASETS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.inxplain_dir = args.inxplain_dir or args.root / 'inxplain'
    args.baseline_dir = args.baseline_dir or args.root / 'baselines'
    args.output_prefix = args.output_prefix or args.root / 'webkb_gcnii_four_labels_summary'

    frame = aggregate(args)
    atomic_write_frame(frame, args.output_prefix.with_suffix('.tsv'), sep='\t')
    write_markdown(frame, args.output_prefix.with_suffix('.md'))
    print(f'Wrote {len(frame)} rows to {args.output_prefix.with_suffix(".tsv")}')
    print(f'Wrote Markdown to {args.output_prefix.with_suffix(".md")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
