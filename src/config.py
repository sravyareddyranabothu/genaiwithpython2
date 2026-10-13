"""Project paths and default settings."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
DEFAULT_CUSTOMER_COUNT: int = 100_000
DEFAULT_PRODUCT_COUNT: int = 10_000
DEFAULT_ORDER_COUNT: int = 1_000_000
RANDOM_SEED: int = 42
LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
