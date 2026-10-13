"""Benchmark common tabular file formats for size, speed, memory, and CPU."""

import logging
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
ROW_COUNT = 500_000
TDP_WATTS = 65.0


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics measured for a single file format and serialization engine."""

    name: str
    size_mb: float
    write_seconds: float
    read_seconds: float
    peak_memory_mb: float
    cpu_seconds: float
    energy_wh: float


def create_dataframe(row_count: int = ROW_COUNT, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible DataFrame with the requested benchmark columns."""
    if row_count < 1:
        raise ValueError("row_count must be positive")
    rng = np.random.default_rng(seed)
    first_names = np.array(["Avery", "Jordan", "Morgan", "Riley", "Taylor", "Casey"])
    last_names = np.array(["Smith", "Patel", "Garcia", "Chen", "Brown", "Singh"])
    categories = np.array(["Electronics", "Clothing", "Home", "Sports", "Books"])
    first = rng.choice(first_names, size=row_count)
    last = rng.choice(last_names, size=row_count)
    ids = np.arange(1, row_count + 1, dtype=np.int64)
    return pd.DataFrame(
        {
            "id": ids,
            "name": np.char.add(np.char.add(first, " "), last),
            "email": np.char.add(np.char.add("user", ids.astype(str)), "@example.com"),
            "amount": np.round(rng.uniform(1, 10_000, size=row_count), 2),
            "date": pd.Timestamp("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 2_400, size=row_count), unit="D"),
            "category": rng.choice(categories, size=row_count),
        }
    )


def measure_operation(operation: Callable[[], object]) -> tuple[float, float, object]:
    """Run an operation and return wall time, peak traced memory, and result."""
    tracemalloc.start()
    started_wall = time.perf_counter()
    try:
        value = operation()
        elapsed = time.perf_counter() - started_wall
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return elapsed, peak_bytes / (1024 * 1024), value


def benchmark_format(
    frame: pd.DataFrame,
    name: str,
    path: Path,
    writer: Callable[[pd.DataFrame, Path], None],
    reader: Callable[[Path], pd.DataFrame],
) -> BenchmarkResult:
    """Measure serialization, deserialization, CPU time, and output size."""
    cpu_started = time.process_time()
    write_seconds, write_peak_mb, _ = measure_operation(lambda: writer(frame, path))
    read_seconds, read_peak_mb, loaded_frame = measure_operation(lambda: reader(path))
    cpu_seconds = time.process_time() - cpu_started
    if not isinstance(loaded_frame, pd.DataFrame) or len(loaded_frame) != len(frame):
        raise RuntimeError(f"{name} read-back did not preserve the DataFrame row count")
    return BenchmarkResult(
        name=name,
        size_mb=path.stat().st_size / (1024 * 1024),
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        peak_memory_mb=max(write_peak_mb, read_peak_mb),
        cpu_seconds=cpu_seconds,
        energy_wh=cpu_seconds * TDP_WATTS / 3600,
    )


def format_bytes_table(results: list[BenchmarkResult], baseline_size_mb: float) -> str:
    """Format measured results and file-size savings compared with CSV."""
    columns = ("Format", "Size MB", "Write s", "Read s", "Peak MB", "CPU s", "Energy Wh", "Size saved")
    rows: list[tuple[str, ...]] = []
    for result in results:
        savings = (baseline_size_mb - result.size_mb) / baseline_size_mb * 100
        rows.append(
            (
                result.name,
                f"{result.size_mb:.2f}",
                f"{result.write_seconds:.2f}",
                f"{result.read_seconds:.2f}",
                f"{result.peak_memory_mb:.2f}",
                f"{result.cpu_seconds:.2f}",
                f"{result.energy_wh:.4f}",
                f"{savings:.1f}%",
            )
        )
    widths = [max(len(columns[index]), *(len(row[index]) for row in rows)) for index in range(len(columns))]
    header = " | ".join(columns[index].ljust(widths[index]) for index in range(len(columns)))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) for row in rows]
    return "\n".join([header, divider, *body])


def run_benchmarks(row_count: int = ROW_COUNT) -> list[BenchmarkResult]:
    """Create a dataset and benchmark CSV, XLSX, Parquet, ORC, and Feather."""
    frame = create_dataframe(row_count)
    # A temporary directory avoids leaving large benchmark files in the project.
    with tempfile.TemporaryDirectory(prefix="format_benchmark_") as directory:
        root = Path(directory)
        cases: list[tuple[str, str, Callable[[pd.DataFrame, Path], None], Callable[[Path], pd.DataFrame]]] = [
            (
                "CSV",
                ".csv",
                lambda data, path: data.to_csv(path, index=False),
                lambda path: pd.read_csv(path),
            ),
            (
                "XLSX",
                ".xlsx",
                lambda data, path: data.to_excel(path, index=False, engine="openpyxl"),
                lambda path: pd.read_excel(path, engine="openpyxl"),
            ),
            (
                "Parquet (fastparquet)",
                ".parquet",
                lambda data, path: data.to_parquet(path, index=False, engine="fastparquet"),
                lambda path: pd.read_parquet(path, engine="fastparquet"),
            ),
            (
                "ORC (pyarrow)",
                ".orc",
                lambda data, path: data.to_orc(path, index=False),
                lambda path: pd.read_orc(path),
            ),
            (
                "Feather (pyarrow)",
                ".feather",
                lambda data, path: data.to_feather(path),
                lambda path: pd.read_feather(path),
            ),
        ]
        results: list[BenchmarkResult] = []
        for name, extension, writer, reader in cases:
            LOGGER.info("Benchmarking %s", name)
            results.append(benchmark_format(frame, name, root / f"benchmark{extension}", writer, reader))
        return results


def main() -> None:
    """Run all benchmarks and print the formatted results table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        results = run_benchmarks()
        csv_baseline = next(result.size_mb for result in results if result.name == "CSV")
        print("\nFile format benchmark (500,000 rows)")
        print(format_bytes_table(results, csv_baseline))
        print("\nSize saved is calculated against the CSV file size; negative values mean a larger file.")
        print("Energy is an estimate based on 65 W CPU TDP multiplied by measured process CPU time.")
        print("Peak memory is Python allocation peak measured by tracemalloc during the larger of write/read.")
    except (ImportError, ModuleNotFoundError) as error:
        LOGGER.exception("A required file-format dependency is missing")
        raise SystemExit("Install requirements.txt before running this benchmark.") from error
    except Exception as error:
        LOGGER.exception("File format benchmark failed")
        raise SystemExit(f"Benchmark failed: {error}") from error


if __name__ == "__main__":
    main()
