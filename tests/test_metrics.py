import math
import unittest

from spatial_support_audit.metrics import (
    grid_cell_id,
    jaccard_similarity,
    overlap_min_fraction,
    summarize_support,
)


class MetricTests(unittest.TestCase):
    def test_jaccard_matches_manuscript_definition(self):
        self.assertAlmostEqual(jaccard_similarity({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(jaccard_similarity({"a"}, {"a"}), 1.0)
        self.assertEqual(jaccard_similarity({"a"}, {"b"}), 0.0)

    def test_empty_support_conventions(self):
        self.assertTrue(math.isnan(jaccard_similarity(set(), set())))
        self.assertTrue(math.isnan(overlap_min_fraction(set(), {"a"})))

    def test_min_fraction_and_counts(self):
        summary = summarize_support({"a", "b"}, {"b", "c", "d"})
        self.assertEqual(summary.early_cells, 2)
        self.assertEqual(summary.late_cells, 3)
        self.assertEqual(summary.intersection_cells, 1)
        self.assertEqual(summary.union_cells, 4)
        self.assertEqual(summary.jaccard, 0.25)
        self.assertEqual(summary.overlap_min_fraction, 0.5)

    def test_grid_uses_floor_for_positive_and_negative_coordinates(self):
        self.assertEqual(grid_cell_id(40.24, -75.01, 0.5), "80_-151")
        self.assertEqual(grid_cell_id(-0.01, 0.49, 0.5), "-1_0")

    def test_invalid_grid_arguments(self):
        with self.assertRaises(ValueError):
            grid_cell_id(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            grid_cell_id(0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
