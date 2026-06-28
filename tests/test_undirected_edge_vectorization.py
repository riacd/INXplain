import unittest

import torch

from GS.models.gradient_based_undirected import (
    GradientBasedUndirectedGraphSummarization,
    JointSubsetBestGradientSummarization,
)


class UndirectedEdgeVectorizationTest(unittest.TestCase):
    def setUp(self):
        self.model = GradientBasedUndirectedGraphSummarization(device="cpu")
        self.edge_index = torch.tensor(
            [
                [2, 0, 1, 1, 3, 2, 4],
                [0, 2, 3, 3, 1, 2, 4],
            ],
            dtype=torch.long,
        )

    def test_extracts_sorted_unique_pairs_and_excludes_self_loops(self):
        self.assertEqual(
            self.model._extract_undirected_edges(self.edge_index),
            [(0, 2), (1, 3)],
        )

    def test_single_pair_removal_removes_both_directions_and_duplicates(self):
        result = self.model._remove_undirected_edge_from_index(
            self.edge_index, 1, 3
        )
        self.assertTrue(
            torch.equal(
                result,
                torch.tensor(
                    [[2, 0, 2, 4], [0, 2, 2, 4]], dtype=torch.long
                ),
            )
        )

    def test_gradient_removal_uses_pair_order(self):
        gradients = torch.tensor([0.2, 0.1])
        pairs = self.model._extract_undirected_edges(self.edge_index)
        result = self.model._remove_undirected_edges_by_gradient(
            self.edge_index, gradients, pairs, 1
        )
        expected = self.model._remove_undirected_edge_from_index(
            self.edge_index, 1, 3
        )
        self.assertTrue(torch.equal(result, expected))

    def test_subset_removal_matches_repeated_single_pair_removal(self):
        subset = [(0, 2), (1, 3)]
        joint_model = JointSubsetBestGradientSummarization(device="cpu")
        vectorized = joint_model._remove_undirected_edge_subset(
            self.edge_index, subset
        )
        repeated = self.edge_index
        for src, dst in subset:
            repeated = self.model._remove_undirected_edge_from_index(
                repeated, src, dst
            )
        self.assertTrue(torch.equal(vectorized, repeated))


if __name__ == "__main__":
    unittest.main()
