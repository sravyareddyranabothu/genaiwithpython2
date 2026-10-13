# E-commerce Data Pipeline

A small Python project that creates reproducible synthetic customer, product, and order data, then uses PySpark to produce business insight reports.

## Project structure

```text
.
├── data/
│   ├── raw/                 # Generated source CSV files
│   └── processed/           # Spark report output directories
├── notebooks/               # Jupyter exploration notebooks
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Python 3.10 or newer and Java 8 or newer are recommended for PySpark. From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate default-sized data files under `data/raw/`:

```bash
python -m src.data_generator
```

Run PySpark analytics and write reports under `data/processed/`:

```bash
python -m src.spark_analytics
```

To customize the data volume and paths:

```bash
python -m src.data_generator --customers 1000 --products 200 --orders 10000 --seed 123
python -m src.spark_analytics --input-dir data/raw --output-dir data/processed
```

The analytics job produces `revenue_by_category`, `monthly_sales`, `top_customers`, and `orders_by_status` directories. Each contains a Spark CSV part file and a header. Revenue is calculated as quantity multiplied by unit price. Data is synthetic and intended for development and analysis examples.

## Tests

Run the standard-library test suite from the project root:

```bash
python -m unittest discover -s tests -v
```
