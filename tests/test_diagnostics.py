import unittest

from spatial_support_audit.diagnostics import CellPeriodCount, diagnose_support


def row(period, cell, effort, focal):
    return CellPeriodCount(period, cell, effort, focal)


class DiagnosticTests(unittest.TestCase):
    def test_available_case_separates_effort_and_species_overlap(self):
        result = diagnose_support(
            [
                row("early", "a", 12, 3),
                row("early", "b", 15, 2),
                row("late", "a", 20, 4),
                row("late", "b", 5, 0),
                row("late", "c", 12, 1),
            ],
            group="maple",
            minimum_stable_cells=1,
        )
        self.assertAlmostEqual(result.effort_support.jaccard, 2 / 3)
        self.assertAlmostEqual(result.species_support.jaccard, 1 / 3)
        self.assertEqual(result.stable_cell_count, 1)
        self.assertTrue(result.raw_estimable)
        self.assertTrue(result.corrected_estimable)
        self.assertEqual(result.availability_reason, "available")

    def test_focal_support_outside_stable_frame_is_not_correctable(self):
        result = diagnose_support(
            [
                row("early", "a", 12, 1),
                row("late", "a", 12, 0),
                row("late", "b", 5, 2),
            ],
            minimum_stable_cells=1,
        )
        self.assertTrue(result.raw_estimable)
        self.assertFalse(result.corrected_estimable)
        self.assertIn(
            "no_late_focal_rate_on_stable_cells", result.availability_reason
        )

    def test_duplicate_cell_period_rows_are_summed(self):
        result = diagnose_support(
            [
                row("early", "a", 5, 1),
                row("early", "a", 5, 1),
                row("late", "a", 10, 2),
            ],
            minimum_stable_cells=1,
        )
        self.assertEqual(result.stable_cell_count, 1)
        self.assertTrue(result.corrected_estimable)
        self.assertAlmostEqual(result.early_relative_rate_sum, 0.2)

    def test_missing_period_is_reported(self):
        result = diagnose_support(
            [row("early", "a", 12, 1)], minimum_stable_cells=1
        )
        self.assertFalse(result.raw_estimable)
        self.assertFalse(result.corrected_estimable)
        self.assertIn("no_late_focal_records", result.availability_reason)

    def test_invalid_counts_and_thresholds_raise(self):
        with self.assertRaises(ValueError):
            row("early", "a", -1, 0)
        with self.assertRaises(ValueError):
            diagnose_support([], minimum_stable_cells=0)
        with self.assertRaises(ValueError):
            diagnose_support([], early_label="same", late_label="same")


if __name__ == "__main__":
    unittest.main()
