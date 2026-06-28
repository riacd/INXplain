# IGPrune: Information-Guided Graph Pruning

This repository contains the source code for the IGPrune graph pruning experiments, including the proposed gradient-based method and baseline integrations. The benchmark evaluates how much downstream node-classification information is retained as graph complexity is reduced.

## Repository Layout

```text
GS/
  models/        IGPrune models, downstream GNNs, and model registry
  datasets/      dataset loading, preprocessing, and task label generation
  benchmark/     unified benchmark orchestration
  metrics/       complexity, information retention, accuracy, and IC-AUC metrics
  utils/         summary-graph export and visualization helpers
baselines/       NetworKit, PRI-Graphs, and SparRL baseline integrations
scripts/         experiment entrypoints and result aggregation scripts
tests/           regression and behavior tests
third_party/     vendored baseline model code used by experiments
```

Local datasets, checkpoints, logs, generated figures, paper drafts, and result files are intentionally excluded from git.

## Installation

The code was developed with Python 3.9, PyTorch, and PyTorch Geometric. Install the Python dependencies with:

```bash
conda create -n GS python=3.9
conda activate GS
pip install -r requirements.txt
```

Some baselines and datasets require optional packages such as `networkit` or `ogb`. Install them when running the corresponding experiments.

## Main Reproduction Commands

List available registered models:

```bash
python scripts/run_unified_benchmark.py --list-models
```

Run a single IGPrune benchmark:

```bash
python scripts/run_unified_benchmark.py \
  --model gradient_based \
  --dataset Cora \
  --task original \
  --downstream gcn \
  --num-steps 10 \
  --epochs 30 \
  --device cuda
```

Run a multi-model benchmark against baselines:

```bash
python scripts/comprehensive_benchmark.py \
  --model-group gradient_and_baselines \
  --dataset-group citation \
  --task-group all \
  --downstream gcn \
  --steps 10 \
  --epochs 30 \
  --device cuda \
  --results-dir ./results/comprehensive_benchmark
```

Run repeated experiments across datasets:

```bash
python scripts/run_multi_dataset_repeated_experiments.py \
  --models gradient_based networkit_random_edge networkit_local_degree pri_graphs \
  --datasets Cora CiteSeer PubMed KarateClub \
  --task original \
  --downstream gcn \
  --num-repeats 5 \
  --num-steps 10 \
  --epochs 30 \
  --device cuda
```

Run the GNN ablation entrypoint:

```bash
python scripts/run_inxplain_gnn_ablation.py \
  --dataset Cora \
  --task original \
  --scoring-model gcn \
  --downstream gcn \
  --device cuda
```

## Models and Tasks

Main model identifiers:

- `gradient_based`: default IGPrune gradient-based undirected graph pruning model.
- `gradient_based_original`: original directed-edge deletion variant kept for comparison.
- `gradient_based_undirected`: explicit undirected variant.
- `neural_enhanced_main`: neural-enhanced gradient variant.
- `networkit_*`: NetworKit sparsification baselines.
- `pri_graphs`: PRI-Graphs baseline.

Common datasets:

- Citation networks: `Cora`, `CiteSeer`, `PubMed`
- Small graph: `KarateClub`
- Academic network: `WikiCS`
- OGB datasets such as `ogbn-arxiv` when `ogb` is installed
- Custom local datasets such as `SO_relation_ME`, `SO_relation_MT`, `HongL`, and `XYH` when the required files exist under `data/`

Supported task labels include `original`, `degree`, `degree_centrality`, `pagerank`, and `closeness_centrality`. Supported downstream models include `gcn` and `gat`, with additional GNN variants used by the ablation scripts.

## Outputs

Benchmark outputs are written under `results/` by default. Important reported quantities include:

- Complexity: normalized edge count.
- Information: downstream task information retention.
- IC-AUC: area under the information-complexity curve.
- Information threshold: minimum complexity needed to retain a target information level.
- Accuracy: downstream node classification accuracy.
- Runtime: graph pruning and downstream training time.

Generated outputs are ignored by git so experiments can be rerun without polluting the source tree.

## Tests

Run focused tests before publishing changes:

```bash
python baselines/tests/test_pri_graphs_simple.py
python baselines/tests/test_networkit_baselines.py --method random_edge --dataset Cora
python -m unittest discover baselines/SparRL/tests
python -m pytest tests
```
