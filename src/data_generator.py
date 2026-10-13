"""Create large, reproducible synthetic e-commerce datasets."""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate customer, product, and order pandas DataFrames.

    The default sizes are suitable for a larger local data pipeline. Set
    ``seed`` to reproduce the NumPy sampling and Faker data.
    """

    CATEGORIES: tuple[str, ...] = ("Electronics", "Clothing", "Home", "Sports", "Books")

    def __init__(self, seed: int | None = 42) -> None:
        """Initialize Faker and NumPy random number generators with a seed."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.faker = Faker()
        if seed is not None:
            self.faker.seed_instance(seed)
        LOGGER.info("Initialized synthetic data generator (seed=%s)", seed)

    def generate_customers(self, count: int = 100_000) -> pd.DataFrame:
        """Generate customers with demographic and registration fields.

        Ages follow a normal distribution centered at 40 years with a standard
        deviation of 12, then are clipped to the range 18–90.
        """
        if count < 1:
            raise ValueError("Customer count must be at least 1")

        LOGGER.info("Generating %d customers", count)
        ages = np.clip(np.rint(self.rng.normal(loc=40, scale=12, size=count)), 18, 90).astype(int)
        rows: list[dict[str, Any]] = []
        for customer_id in tqdm(range(1, count + 1), desc="Generating customers", unit="customer"):
            name = self.faker.name()
            rows.append(
                {
                    "customer_id": customer_id,
                    "name": name,
                    "email": self.faker.unique.email(),
                    "age": int(ages[customer_id - 1]),
                    "city": self.faker.city(),
                    "country": self.faker.country(),
                    "registration_date": self.faker.date_between(
                        start_date="-5y", end_date="today"
                    ).isoformat(),
                }
            )
        result = pd.DataFrame.from_records(rows)
        LOGGER.info("Generated customer DataFrame with %d rows", len(result))
        return result

    def generate_products(self, count: int = 10_000) -> pd.DataFrame:
        """Generate products with category, price, stock, and rating fields."""
        if count < 1:
            raise ValueError("Product count must be at least 1")

        LOGGER.info("Generating %d products", count)
        names = [f"{self.faker.word().title()} {self.faker.word().title()}" for _ in
                 tqdm(range(count), desc="Generating products", unit="product")]
        result = pd.DataFrame(
            {
                "product_id": np.arange(1, count + 1, dtype=np.int64),
                "name": names,
                "category": self.rng.choice(self.CATEGORIES, size=count),
                "price": np.round(self.rng.uniform(10, 500, size=count), 2),
                "stock": self.rng.integers(0, 1_001, size=count),
                "rating": np.round(self.rng.uniform(1, 5, size=count), 1),
            }
        )
        LOGGER.info("Generated product DataFrame with %d rows", len(result))
        return result

    def _sample_customer_ids(self, count: int, customer_count: int) -> np.ndarray:
        """Sample IDs so roughly 20% of customers receive 80% of orders.

        Pareto-distributed weights create a long tail within both the high- and
        low-order customer groups; the 80/20 split is exact in order count.
        """
        if customer_count < 1:
            raise ValueError("Customer count must be at least 1")
        if count == 0:
            return np.empty(0, dtype=np.int64)

        heavy_count = max(1, int(np.ceil(customer_count * 0.20)))
        heavy_ids = np.arange(1, heavy_count + 1, dtype=np.int64)
        tail_ids = np.arange(heavy_count + 1, customer_count + 1, dtype=np.int64)
        heavy_order_count = int(np.ceil(count * 0.80)) if tail_ids.size else count
        tail_order_count = count - heavy_order_count

        heavy_weights = self.rng.pareto(a=1.5, size=heavy_count) + 1
        heavy_probabilities = heavy_weights / heavy_weights.sum()
        heavy_samples = self.rng.choice(heavy_ids, size=heavy_order_count, p=heavy_probabilities)
        if tail_order_count == 0:
            return heavy_samples

        tail_weights = self.rng.pareto(a=1.5, size=tail_ids.size) + 1
        tail_probabilities = tail_weights / tail_weights.sum()
        tail_samples = self.rng.choice(tail_ids, size=tail_order_count, p=tail_probabilities)
        return np.concatenate((heavy_samples, tail_samples))

    def generate_orders(
        self,
        customers: pd.DataFrame | int = 100_000,
        products: pd.DataFrame | int = 10_000,
        count: int = 1_000_000,
    ) -> pd.DataFrame:
        """Generate orders referencing customer and product identifiers.

        Pass DataFrames to derive the available IDs from generated inputs, or
        pass integer customer/product counts when only their sequential IDs
        are needed. Order dates cover the previous two years.
        """
        customer_ids = (
            customers["customer_id"].to_numpy(dtype=np.int64)
            if isinstance(customers, pd.DataFrame)
            else np.arange(1, customers + 1, dtype=np.int64)
        )
        product_ids = (
            products["product_id"].to_numpy(dtype=np.int64)
            if isinstance(products, pd.DataFrame)
            else np.arange(1, products + 1, dtype=np.int64)
        )
        if count < 0:
            raise ValueError("Order count cannot be negative")
        if customer_ids.size == 0 or product_ids.size == 0:
            raise ValueError("At least one customer and one product are required")

        LOGGER.info("Generating %d orders", count)
        sampled_customers = self._sample_customer_ids(count, int(customer_ids.size))
        sampled_customers = customer_ids[sampled_customers - 1]
        sampled_products = self.rng.choice(product_ids, size=count)
        quantities = self.rng.integers(1, 11, size=count)
        start = np.datetime64(date.today() - timedelta(days=730), "D")
        end = np.datetime64(date.today(), "D")
        day_offsets = self.rng.integers(0, int((end - start).astype(int)) + 1, size=count)
        order_dates = (start + day_offsets).astype("datetime64[D]").astype(str)

        # tqdm gives visible progress while materializing the million-row frame.
        rows = (
            {
                "order_id": order_id,
                "customer_id": int(sampled_customers[order_id - 1]),
                "product_id": int(sampled_products[order_id - 1]),
                "quantity": int(quantities[order_id - 1]),
                "order_date": order_dates[order_id - 1],
            }
            for order_id in tqdm(range(1, count + 1), desc="Generating orders", unit="order")
        )
        result = pd.DataFrame.from_records(rows, columns=[
            "order_id", "customer_id", "product_id", "quantity", "order_date"
        ])
        LOGGER.info("Generated order DataFrame with %d rows", len(result))
        return result

    def generate_all(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customer, product, and order DataFrames."""
        customers = self.generate_customers(customer_count)
        products = self.generate_products(product_count)
        orders = self.generate_orders(customers, products, order_count)
        return customers, products, orders


def main() -> None:
    """Generate the default-sized datasets and save them as CSV files."""
    import argparse

    from src.config import LOG_FORMAT, RAW_DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    generator = SyntheticDataGenerator(args.seed)
    customers, products, orders = generator.generate_all(args.customers, args.products, args.orders)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("customers", customers), ("products", products), ("orders", orders)):
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
        LOGGER.info("Saved %s.csv to %s", name, args.output_dir)


if __name__ == "__main__":
    main()
