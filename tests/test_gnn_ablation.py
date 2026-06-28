import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from GS.datasets.loaders import DatasetLoader
from GS.models import (
    GATDownstreamModel,
    GCNDownstreamModel,
    GCNIIDownstreamModel,
    GraphSAGEDownstreamModel,
    H2GCNDownstreamModel,
    create_downstream_model,
    normalize_downstream_model_name,
)
from GS.models.gradient_based_undirected import GradientBasedUndirectedGraphSummarization
from scripts.run_inxplain_gnn_ablation import COMBINATIONS, aggregate_results


class GNNAbationTest(unittest.TestCase):
    def test_downstream_factory_preserves_models_and_defaults(self):
        gcn = create_downstream_model('gcn', input_dim=4)
        gat = create_downstream_model('gat', input_dim=4)
        sage = create_downstream_model('sage', input_dim=4)
        h2gcn = create_downstream_model('h2gcn', input_dim=4)
        gcnii = create_downstream_model('gcnii', input_dim=4)

        self.assertIsInstance(gcn, GCNDownstreamModel)
        self.assertIsInstance(gat, GATDownstreamModel)
        self.assertIsInstance(sage, GraphSAGEDownstreamModel)
        self.assertIsInstance(h2gcn, H2GCNDownstreamModel)
        self.assertIsInstance(gcnii, GCNIIDownstreamModel)
        self.assertEqual(gcn.hidden_dim, 16)
        self.assertEqual(gat.hidden_dim, 8)
        self.assertEqual(sage.hidden_dim, 64)
        self.assertEqual(h2gcn.hidden_dim, 64)
        self.assertEqual(h2gcn.k, 2)
        self.assertEqual(gcnii.hidden_dim, 64)
        self.assertEqual(gcnii.num_layers, 64)
        self.assertEqual(gcnii.alpha, 0.5)
        self.assertEqual(gcnii.lamda, 0.5)

        cornell_gcnii = create_downstream_model(
            'gcnii', input_dim=4, dataset_name='Cornell'
        )
        self.assertEqual(cornell_gcnii.num_layers, 16)
        self.assertEqual(cornell_gcnii.lamda, 1.0)
        self.assertEqual(cornell_gcnii.weight_decay, 1e-3)

    def test_sage_aliases_are_canonical_and_used_for_scoring(self):
        self.assertEqual(normalize_downstream_model_name('GraphSAGE'), 'sage')
        self.assertEqual(normalize_downstream_model_name('graph_sage'), 'sage')

        summarizer = GradientBasedUndirectedGraphSummarization(
            downstream_model_type='sage'
        )
        scorer = summarizer._create_scoring_downstream_model(4, 3)
        self.assertIsInstance(scorer, GraphSAGEDownstreamModel)

    def test_new_models_are_canonical_and_used_for_scoring(self):
        self.assertEqual(normalize_downstream_model_name('H2GCN'), 'h2gcn')
        self.assertEqual(normalize_downstream_model_name('GCN2'), 'gcnii')

        gcnii_summarizer = GradientBasedUndirectedGraphSummarization(
            downstream_model_type='gcnii',
            dataset_name='Texas',
        )
        gcnii_scorer = gcnii_summarizer._create_scoring_downstream_model(4, 3)
        self.assertIsInstance(gcnii_scorer, GCNIIDownstreamModel)
        self.assertEqual(gcnii_scorer.hidden_dim, 64)
        self.assertEqual(gcnii_scorer.output_dim, 3)
        self.assertEqual(gcnii_scorer.num_layers, 32)
        self.assertEqual(gcnii_scorer.lamda, 1.5)
        self.assertEqual(gcnii_scorer.weight_decay, 1e-4)

        h2gcn_summarizer = GradientBasedUndirectedGraphSummarization(
            downstream_model_type='h2gcn',
            dataset_name='Cornell',
        )
        h2gcn_scorer = h2gcn_summarizer._create_scoring_downstream_model(4, 3)
        self.assertIsInstance(h2gcn_scorer, H2GCNDownstreamModel)
        self.assertEqual(h2gcn_scorer.hidden_dim, 64)
        self.assertEqual(h2gcn_scorer.output_dim, 3)
        self.assertEqual(h2gcn_scorer.lr, 0.01)
        self.assertEqual(h2gcn_scorer.weight_decay, 5e-4)
        self.assertEqual(h2gcn_scorer.dropout, 0.5)
        self.assertFalse(h2gcn_scorer.use_relu)

    def test_webkb_uses_split_zero_masks(self):
        data = Data(
            x=torch.eye(4),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
            y=torch.tensor([0, 1, 0, 1]),
            train_mask=torch.tensor([
                [True, False], [False, True], [False, False], [False, False]
            ]),
            val_mask=torch.tensor([
                [False, False], [True, False], [False, True], [False, False]
            ]),
            test_mask=torch.tensor([
                [False, True], [False, False], [True, False], [True, True]
            ]),
        )

        class FakeDataset:
            def __getitem__(self, index):
                self.index = index
                return data

        with tempfile.TemporaryDirectory() as directory:
            with patch('GS.datasets.loaders.WebKB', return_value=FakeDataset()) as webkb:
                _, train_mask, val_mask, test_mask = DatasetLoader(
                    directory
                ).load_dataset('Cornell', normalize_features=False)

        webkb.assert_called_once_with(
            root=directory, name='Cornell', transform=None
        )
        self.assertTrue(torch.equal(train_mask, data.train_mask[:, 0]))
        self.assertTrue(torch.equal(val_mask, data.val_mask[:, 0]))
        self.assertTrue(torch.equal(test_mask, data.test_mask[:, 0]))

    def test_ablation_aggregation_requires_five_finite_seeds(self):
        datasets = ['Cora']
        seeds = [42, 43, 44, 45, 46]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for combination, models in COMBINATIONS.items():
                scoring_model, evaluation_model = models
                for seed in seeds:
                    record_dir = output_dir / combination / 'runs' / 'Cora'
                    record_dir.mkdir(parents=True, exist_ok=True)
                    record = {
                        'success': True,
                        'seed': seed,
                        'scoring_model': scoring_model,
                        'evaluation_model': evaluation_model,
                        'ic_auc_additive': 0.5,
                        'ic_auc_log_ratio': 0.4,
                        'threshold_point_additive': 0.3,
                        'threshold_point_log_ratio': 0.2,
                        'original_accuracy': 0.8,
                        'empty_accuracy': 0.2,
                        'run_time': 10.0,
                    }
                    with open(record_dir / f'seed_{seed}.json', 'w') as handle:
                        json.dump(record, handle)

            frame = aggregate_results(output_dir, datasets, seeds)

            self.assertEqual(len(frame), 6)
            self.assertTrue(frame['Complete'].all())
            self.assertTrue((frame['Successful_Seeds'] == 5).all())
            self.assertEqual(
                frame['Scoring_Model'].tolist(),
                [models[0] for models in COMBINATIONS.values()],
            )
            self.assertTrue((output_dir / 'inxplain_gnn_ablation_summary.tsv').exists())
            self.assertTrue((output_dir / 'inxplain_gnn_ablation_summary.csv').exists())
            self.assertTrue((output_dir / 'inxplain_gnn_ablation_summary.md').exists())


if __name__ == '__main__':
    unittest.main()
