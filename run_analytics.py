"""Run and time the e-commerce PySpark analytics interactively."""

import logging
import time
from pathlib import Path

from pyspark.sql import DataFrame

from src.config import LOG_FORMAT, RAW_DATA_DIR
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def time_operation(label: str, operation: object) -> DataFrame:
    """Execute a DataFrame-producing operation and print its elapsed time."""
    if not callable(operation):
        raise TypeError(f"Operation for {label} must be callable")
    started_at = time.perf_counter()
    result = operation()
    print(f"\n{label}")
    result.show(truncate=False)
    elapsed_seconds = time.perf_counter() - started_at
    print(f"Execution time: {elapsed_seconds:.2f} seconds")
    return result


def run(input_dir: Path = RAW_DATA_DIR) -> None:
    """Load the three Parquet datasets, show all analytics, then stop Spark."""
    analytics = SalesAnalytics()
    try:
        analytics.create_spark_session()
        customers_df = analytics.load_parquet(input_dir / "customers.parquet")
        products_df = analytics.load_parquet(input_dir / "products.parquet")
        orders_df = analytics.load_parquet(input_dir / "orders.parquet")

        print("Customers sample")
        customers_df.show(5, truncate=False)

        time_operation(
            "Top 10 customers by revenue",
            lambda: analytics.top_customers_by_revenue(orders_df, products_df),
        )
        time_operation(
            "Sales by category",
            lambda: analytics.sales_by_category(orders_df, products_df),
        )
        time_operation(
            "Monthly revenue trends",
            lambda: analytics.monthly_trends(orders_df, products_df),
        )
    except Exception:
        LOGGER.exception("Analytics run failed")
        raise
    finally:
        analytics.stop()


def main() -> None:
    """Configure logging and run analytics using an optional data directory."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    run(args.input_dir)


if __name__ == "__main__":
    main()
