import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data

from GS.metrics import InformationMetric
from scripts.aggregate_repeated_experiment_summary import aggregate
from scripts.run_multi_dataset_repeated_experiments import run_single_experiment


class FakeDownstreamModel:
    def __init__(self):
        self.reset_draws = []
        self.train_calls = 0
        self.labels = None

    def reset(self):
        self.reset_draws.append(torch.rand(1).item())

    def train_model(self, graph, train_mask, val_mask, labels, epochs=200):
        self.train_calls += 1
        self.labels = labels
        torch.rand(3)

    def evaluate(self, graph, test_mask, labels):
        return 3.0 - graph.edge_index.size(1)

    def predict(self, graph):
        logits = torch.zeros((graph.num_nodes, 2), device=graph.x.device)
        logits[torch.arange(graph.num_nodes), self.labels] = 1.0
        return logits


class FakeBenchmark:
    last_call = None

    def __init__(self, **kwargs):
        self.enable_memory_optimization = True

    def run_single_model(self, **kwargs):
        FakeBenchmark.last_call = kwargs
        return {
            'success': True,
            'ic_auc_additive': 0.5,
            'ic_auc_log_ratio': 0.5,
            'threshold_point_additive': 0.5,
            'threshold_point_log_ratio': 0.5,
        }


class RigorousEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.graphs = [
            Data(
                x=torch.ones((3, 2)),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                y=torch.tensor([0, 1, 0]),
                num_nodes=3,
            ),
            Data(
                x=torch.ones((3, 2)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                y=torch.tensor([0, 1, 0]),
                num_nodes=3,
            ),
        ]
        self.train_mask = torch.tensor([True, True, False])
        self.val_mask = torch.tensor([False, False, True])
        self.test_mask = torch.tensor([False, False, True])

    def test_graph_steps_share_initialization_and_train_once(self):
        downstream = FakeDownstreamModel()
        metric = InformationMetric(downstream, random_seed=17)

        losses, accuracies = metric.evaluate_list(
            self.graphs,
            self.train_mask,
            self.val_mask,
            self.test_mask,
            self.graphs[0].y,
            epochs=4,
        )

        self.assertEqual(downstream.train_calls, len(self.graphs))
        self.assertEqual(downstream.reset_draws[0], downstream.reset_draws[1])
        self.assertEqual(losses, [1.0, 3.0])
        self.assertEqual(accuracies, [1.0, 1.0])

    def test_both_normalizations_reuse_raw_losses(self):
        losses = [1.0, 2.0, 3.0]
        additive = InformationMetric.normalize_losses(losses, 'additive')
        log_ratio = InformationMetric.normalize_losses(losses, 'log_ratio')

        self.assertEqual(additive, [1.0, 0.5, 0.0])
        self.assertAlmostEqual(log_ratio[0], 1.0)
        self.assertAlmostEqual(log_ratio[-1], 0.0)

    def test_repeated_seed_reaches_networkit_baseline(self):
        with patch(
            'scripts.run_multi_dataset_repeated_experiments.UnifiedBenchmark',
            FakeBenchmark,
        ), patch(
            'scripts.run_multi_dataset_repeated_experiments.model_registry.get_model_info',
            return_value={'category': 'baseline'},
        ):
            result = run_single_experiment(
                model_name='networkit_random_edge',
                dataset='Cora',
                task='original',
                downstream='gcn',
                num_steps=2,
                epochs=2,
                seed=123,
                device='cpu',
            )

        self.assertTrue(result['success'])
        self.assertEqual(FakeBenchmark.last_call['model_kwargs']['seed'], 123)

    def test_repeated_seed_overrides_sampling_seed(self):
        with patch(
            'scripts.run_multi_dataset_repeated_experiments.UnifiedBenchmark',
            FakeBenchmark,
        ), patch(
            'scripts.run_multi_dataset_repeated_experiments.model_registry.get_model_info',
            return_value={'category': 'development'},
        ):
            run_single_experiment(
                model_name='gradient_based_joint_edge_score_stable',
                dataset='Cora',
                task='original',
                downstream='gcn',
                num_steps=2,
                epochs=2,
                seed=456,
                device='cpu',
                model_kwargs={'sampling_seed': 42},
            )

        self.assertEqual(
            FakeBenchmark.last_call['model_kwargs']['sampling_seed'], 456
        )

    def test_aggregate_combines_parallel_seed_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for seed, auc in ((42, 0.4), (43, 0.6)):
                payload = {
                    'config': {
                        'task': 'original',
                        'downstream': 'gcn',
                        'num_steps': 10,
                        'epochs': 200,
                        'num_repeats': 1,
                        'seeds': [seed],
                    },
                    'results': {
                        'ogbn-arxiv': {
                            'networkit_random_edge': {
                                'runs': [{
                                    'seed': seed,
                                    'repeat_idx': 1,
                                    'success': True,
                                    'ic_auc_additive': auc,
                                    'ic_auc_log_ratio': auc,
                                    'threshold_point_additive': 0.5,
                                    'threshold_point_log_ratio': 0.5,
                                    'run_time': 10.0,
                                }],
                                'statistics': {},
                            }
                        }
                    },
                }
                path = Path(tmpdir) / f'multi_dataset_repeated_results_{seed}.json'
                path.write_text(json.dumps(payload))

            result = aggregate(Path(tmpdir)).iloc[0]

        self.assertEqual(result['Success_Rate'], '2/2')
        self.assertAlmostEqual(result['IC_AUC_Add_Mean'], 0.5)
        self.assertGreater(result['IC_AUC_Add_StdErr'], 0.0)


if __name__ == '__main__':
    unittest.main()
