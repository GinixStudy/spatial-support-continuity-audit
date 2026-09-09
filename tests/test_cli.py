import csv
from pathlib import Path
import unittest

from spatial_support_audit.cli import main


EXAMPLE_ROWS = [
    ["maple", "early", "a", "12", "3"],
    ["maple", "early", "b", "15", "2"],
    ["maple", "late", "a", "20", "4"],
    ["maple", "late", "b", "5", "0"],
    ["maple", "late", "c", "12", "1"],
    ["oak", "early", "a", "12", "1"],
    ["oak", "late", "a", "12", "0"],
    ["oak", "late", "b", "5", "2"],
]


class CliTests(unittest.TestCase):
    def test_cli_writes_one_row_per_group(self):
        input_path = Path(__file__).with_name("_cli_input.csv")
        output_path = Path(__file__).with_name("_cli_output.csv")
        try:
            with input_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["species", "period", "cell_id", "effort_count", "focal_count"]
                )
                writer.writerows(EXAMPLE_ROWS)

            status = main(
                [
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--group-column",
                    "species",
                    "--minimum-stable-cells",
                    "1",
                ]
            )
            self.assertEqual(status, 0)
            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        self.assertEqual([item["group"] for item in rows], ["maple", "oak"])
        self.assertEqual(rows[0]["corrected_estimable"], "True")
        self.assertEqual(rows[1]["corrected_estimable"], "False")


if __name__ == "__main__":
    unittest.main()
