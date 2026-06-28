import unittest

import torch
from torch_geometric.data import Data

from scripts.case_study_karate_edge_stability import (
    add_random_edges,
    aggregate_edge_probabilities,
    undirected_edge_set,
)


class KarateEdgeStabilityTest(unittest.TestCase):
    def test_add_random_edges_adds_twenty_percent_without_duplicates(self):
        edges = {(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 4)}
        directed = [(u, v) for edge in edges for u, v in (edge, edge[::-1])]
        graph = Data(
            x=torch.ones((5, 1)),
            edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
            num_nodes=5,
        )

        perturbed, added = add_random_edges(graph, fraction=0.20, seed=42)

        self.assertEqual(len(added), 2)
        self.assertTrue(added.isdisjoint(edges))
        self.assertEqual(undirected_edge_set(perturbed.edge_index), edges | added)

    def test_probability_aggregation_uses_union_of_run_edges(self):
        paths = [
            [{(0, 1), (1, 2)}, {(0, 1)}, set()],
            [{(0, 1), (0, 2)}, {(0, 2)}, set()],
        ]

        edges, probabilities = aggregate_edge_probabilities(paths)
        by_edge = {edge: probabilities[idx].tolist() for idx, edge in enumerate(edges)}

        self.assertEqual(by_edge[(0, 1)], [1.0, 0.5, 0.0])
        self.assertEqual(by_edge[(1, 2)], [0.5, 0.0, 0.0])
        self.assertEqual(by_edge[(0, 2)], [0.5, 0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
