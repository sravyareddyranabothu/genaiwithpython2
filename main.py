"""Generate e-commerce datasets and save them as Parquet files."""

import logging
import sys
import time
from pathlib import Path

from src.config import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_ORDER_COUNT,
    DEFAULT_PRODUCT_COUNT,
    LOG_FORMAT,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Convert a byte count to a readable binary unit."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def generate_and_save(
    output_dir: Path = RAW_DATA_DIR,
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    seed: int = RANDOM_SEED,
) -> dict[str, Path]:
    """Generate all datasets and save each DataFrame as a Parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = SyntheticDataGenerator(seed=seed)
    customers, products, orders = generator.generate_all(
        customer_count=customer_count,
        product_count=product_count,
        order_count=order_count,
    )
    files: dict[str, Path] = {}
    for name, frame in (
        ("customers", customers),
        ("products", products),
        ("orders", orders),
    ):
        path = output_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        files[name] = path
        LOGGER.info("Saved %s rows to %s", len(frame), path)
    return files


def main() -> int:
    """Run generation, report elapsed time and file sizes, and handle failures."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    started_at = time.perf_counter()
    try:
        files = generate_and_save()
        elapsed_seconds = time.perf_counter() - started_at
        print(f"Generation and save time: {elapsed_seconds:.2f} seconds")
        for dataset_name, path in files.items():
            print(f"{dataset_name}: {path} ({format_file_size(path.stat().st_size)})")
        return 0
    except ImportError as error:
        LOGGER.exception("A required package or Parquet engine is unavailable")
        print(
            "Install project dependencies with `python -m pip install -r requirements.txt` "
            "and try again.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as error:
        LOGGER.exception("Could not generate or save the datasets: %s", error)
        return 1
    except Exception:
        LOGGER.exception("Unexpected error while generating datasets")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
