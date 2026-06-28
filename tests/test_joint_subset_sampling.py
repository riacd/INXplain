import unittest

from GS.models.gradient_based_undirected import (
    JointSubsetBestGradientSummarization,
    JointSubsetEdgeScoreGradientSummarization,
    JointSubsetProductImportanceGradientSummarization,
    JointSubsetStabilityAwareEdgeScoreGradientSummarization,
)


class JointSubsetSamplingTest(unittest.TestCase):
    def make_model(self, model_cls, subset_num, repeats=1):
        model = model_cls.__new__(model_cls)
        model.sampling_subset_num = subset_num
        model.sampling_repeats = repeats
        model.sampling_seed = 42
        return model

    def test_edge_score_models_use_subset_num_as_split_count(self):
        edges = [(idx, idx + 1) for idx in range(10)]

        for model_cls in (
            JointSubsetEdgeScoreGradientSummarization,
            JointSubsetStabilityAwareEdgeScoreGradientSummarization,
        ):
            for subset_num in (2, 4):
                with self.subTest(model_cls=model_cls.__name__, subset_num=subset_num):
                    subsets = self.make_model(model_cls, subset_num)._sample_edge_subsets(
                        edges, subset_size=3, step=0
                    )

                    self.assertEqual(len(subsets), subset_num)
                    self.assertEqual(sum(map(len, subsets)), len(edges))
                    self.assertEqual(
                        sorted(edge for subset in subsets for edge in subset),
                        edges,
                    )

    def test_subset_selection_models_keep_deletion_size_chunking(self):
        edges = [(idx, idx + 1) for idx in range(10)]

        for model_cls in (
            JointSubsetBestGradientSummarization,
            JointSubsetProductImportanceGradientSummarization,
        ):
            with self.subTest(model_cls=model_cls.__name__):
                subsets = self.make_model(model_cls, subset_num=4)._sample_edge_subsets(
                    edges, subset_size=3, step=0
                )

                self.assertEqual(len(subsets), 3)
                self.assertTrue(all(len(subset) == 3 for subset in subsets))
                self.assertEqual(sum(map(len, subsets)), 9)

    def test_all_keeps_deletion_size_chunking(self):
        edges = [(idx, idx + 1) for idx in range(10)]
        subsets = self.make_model(
            JointSubsetStabilityAwareEdgeScoreGradientSummarization,
            subset_num=None,
        )._sample_edge_subsets(
            edges, subset_size=3, step=0
        )

        self.assertEqual(len(subsets), 3)
        self.assertTrue(all(len(subset) == 3 for subset in subsets))
        self.assertEqual(sum(map(len, subsets)), 9)


if __name__ == "__main__":
    unittest.main()
