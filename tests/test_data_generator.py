"""Basic checks for synthetic data generation."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data_generator import generate_data


class DataGeneratorTests(unittest.TestCase):
    """Verify generated files have the expected rows and relationships."""

    def test_generate_data_writes_linked_csv_files(self) -> None:
        """Generate a small dataset and check file contents and foreign keys."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            generate_data(3, 2, 5, output_dir, 7)
            with (output_dir / "customers.csv").open(encoding="utf-8", newline="") as file_handle:
                customers = list(csv.DictReader(file_handle))
            with (output_dir / "products.csv").open(encoding="utf-8", newline="") as file_handle:
                products = list(csv.DictReader(file_handle))
            with (output_dir / "orders.csv").open(encoding="utf-8", newline="") as file_handle:
                orders = list(csv.DictReader(file_handle))
            self.assertEqual(len(customers), 3)
            self.assertEqual(len(products), 2)
            self.assertEqual(len(orders), 5)
            self.assertTrue({row["customer_id"] for row in orders} <= {row["customer_id"] for row in customers})
            self.assertTrue({row["product_id"] for row in orders} <= {row["product_id"] for row in products})


if __name__ == "__main__":
    unittest.main()
