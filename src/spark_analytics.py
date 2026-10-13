"""PySpark sales analytics for e-commerce Parquet datasets."""

import argparse
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import LOG_FORMAT, PROCESSED_DATA_DIR, RAW_DATA_DIR

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Load e-commerce Parquet data and calculate sales insights."""

    def __init__(self) -> None:
        """Initialize an analytics helper without starting Spark immediately."""
        self.spark: SparkSession | None = None

    def create_spark_session(self, app_name: str = "EcommerceSalesAnalytics") -> SparkSession:
        """Create a local Spark session configured with 4 GB and Kryo serialization."""
        self.spark = (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .getOrCreate()
        )
        LOGGER.info("Created Spark session in local mode with adaptive execution and Kryo")
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame."""
        if self.spark is None:
            raise RuntimeError("Create the Spark session before loading Parquet data")
        parquet_path = Path(path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet path does not exist: {parquet_path}")
        LOGGER.info("Loading Parquet data from %s", parquet_path)
        return self.spark.read.parquet(str(parquet_path))

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top N customers by total spend, using product prices."""
        if n < 1:
            raise ValueError("n must be at least 1")
        order_lines = orders_df.join(
            products_df.select("product_id", F.col("price").alias("unit_price")),
            on="product_id",
            how="inner",
        ).withColumn("revenue", F.col("quantity") * F.col("unit_price"))
        return (
            order_lines.groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Aggregate gross revenue and units sold for each product category."""
        order_lines = orders_df.join(
            products_df.select("product_id", "category", F.col("price").alias("unit_price")),
            on="product_id",
            how="inner",
        ).withColumn("revenue", F.col("quantity") * F.col("unit_price"))
        return (
            order_lines.groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return monthly revenue and month-over-month revenue growth percent."""
        order_lines = orders_df.join(
            products_df.select("product_id", F.col("price").alias("unit_price")),
            on="product_id",
            how="inner",
        ).withColumn("revenue", F.col("quantity") * F.col("unit_price"))
        monthly_revenue = order_lines.groupBy(
            F.date_trunc("month", F.to_date("order_date")).alias("month")
        ).agg(F.sum("revenue").alias("total_revenue"))
        month_window = Window.orderBy("month")
        return (
            monthly_revenue.withColumn("previous_month_revenue", F.lag("total_revenue").over(month_window))
            .withColumn(
                "revenue_growth_pct",
                F.when(
                    F.col("previous_month_revenue").isNull() | (F.col("previous_month_revenue") == 0),
                    F.lit(None).cast("double"),
                ).otherwise(
                    F.round(
                        (F.col("total_revenue") - F.col("previous_month_revenue"))
                        / F.col("previous_month_revenue")
                        * 100,
                        2,
                    )
                ),
            )
            .orderBy("month")
        )

    def stop(self) -> None:
        """Stop this helper's Spark session when one has been created."""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
            LOGGER.info("Stopped Spark session")


def write_report(data: DataFrame, output_dir: Path, name: str) -> None:
    """Write an aggregated Spark DataFrame as a CSV report directory."""
    destination = output_dir / name
    data.coalesce(1).write.mode("overwrite").option("header", True).csv(str(destination))
    LOGGER.info("Saved report %s", destination)


def run_analytics(input_dir: Path = RAW_DATA_DIR, output_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Load generated Parquet data, calculate insights, and write report CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analytics = SalesAnalytics()
    try:
        analytics.create_spark_session()
        orders = analytics.load_parquet(input_dir / "orders.parquet")
        products = analytics.load_parquet(input_dir / "products.parquet")
        reports: dict[str, DataFrame] = {
            "top_customers": analytics.top_customers_by_revenue(orders, products),
            "sales_by_category": analytics.sales_by_category(orders, products),
            "monthly_trends": analytics.monthly_trends(orders, products),
        }
        for name, report in reports.items():
            write_report(report, output_dir, name)
        LOGGER.info("Analytics completed; reports are in %s", output_dir)
    finally:
        analytics.stop()


def main() -> None:
    """Parse data paths and execute the Parquet analytics pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    run_analytics(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
