from pathlib import Path
import json
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

REQUIRED_COLUMNS = [
    "year",
    "institution_name",
    "unitid",
    "carnegie_classification",
    "department_name",
    "cip_code",
    "cip_title",
    "total_sch",
    "faculty_fte",
    "direct_instructional_expenditures",
]

NUMERIC_COLUMNS = [
    "total_sch",
    "faculty_fte",
    "direct_instructional_expenditures",
]


def clean_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].round(2)
    return df


def write_csv_and_json(df: pd.DataFrame, base_name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED_DIR / f"{base_name}.csv"
    json_path = PROCESSED_DIR / f"{base_name}.json"

    df.to_csv(csv_path, index=False)

    records = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("**/*.csv"))

    if not raw_files:
        raise FileNotFoundError("No CSV files found in data/raw/.")

    frames = []
    quality_rows = []

    for raw_file in raw_files:
        df = pd.read_csv(
            raw_file,
            dtype={
                "year": str,
                "unitid": str,
                "cip_code": str,
            },
        )

        df.columns = [clean_column_name(column) for column in df.columns]

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{raw_file} is missing required columns: {missing_columns}"
            )

        df["source_file"] = str(raw_file)

        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in [
            "year",
            "institution_name",
            "unitid",
            "carnegie_classification",
            "department_name",
            "cip_code",
            "cip_title",
        ]:
            df[column] = df[column].astype(str).str.strip()

        df["valid_for_metrics"] = (
            (df["total_sch"] > 0)
            & (df["faculty_fte"] > 0)
            & (df["direct_instructional_expenditures"] >= 0)
        )

        quality_rows.append(
            {
                "source_file": str(raw_file),
                "rows": len(df),
                "valid_metric_rows": int(df["valid_for_metrics"].sum()),
                "invalid_metric_rows": int((~df["valid_for_metrics"]).sum()),
            }
        )

        frames.append(df)

    all_raw = pd.concat(frames, ignore_index=True)

    metrics = all_raw[all_raw["valid_for_metrics"]].copy()

    metrics["instructional_cost_per_faculty"] = (
        metrics["direct_instructional_expenditures"] / metrics["faculty_fte"]
    )

    metrics["cost_per_credit_hour"] = (
        metrics["direct_instructional_expenditures"] / metrics["total_sch"]
    )

    public_metrics_columns = [
        "year",
        "institution_name",
        "unitid",
        "carnegie_classification",
        "department_name",
        "cip_code",
        "cip_title",
        "total_sch",
        "faculty_fte",
        "direct_instructional_expenditures",
        "instructional_cost_per_faculty",
        "cost_per_credit_hour",
        "source_file",
    ]

    public_metrics = metrics[public_metrics_columns].copy()
    public_metrics = round_numeric_columns(public_metrics)
    public_metrics = public_metrics.sort_values(
        [
            "year",
            "carnegie_classification",
            "cip_code",
            "institution_name",
            "department_name",
        ]
    )

    group_columns = [
        "year",
        "cip_code",
        "carnegie_classification",
    ]

    benchmark = (
        metrics
        .groupby(group_columns)
        .agg(
            cip_title=("cip_title", "first"),
            n_institutions=("unitid", "nunique"),
            n_records=("institution_name", "count"),

            mean_instructional_cost_per_faculty=(
                "instructional_cost_per_faculty",
                "mean",
            ),
            p10_instructional_cost_per_faculty=(
                "instructional_cost_per_faculty",
                lambda x: x.quantile(0.10),
            ),
            p25_instructional_cost_per_faculty=(
                "instructional_cost_per_faculty",
                lambda x: x.quantile(0.25),
            ),
            median_instructional_cost_per_faculty=(
                "instructional_cost_per_faculty",
                "median",
            ),
            p75_instructional_cost_per_faculty=(
                "instructional_cost_per_faculty",
                lambda x: x.quantile(0.75),
            ),
            p90_instructional_cost_per_faculty=(
                "instructional_cost_per_faculty",
                lambda x: x.quantile(0.90),
            ),

            mean_cost_per_credit_hour=(
                "cost_per_credit_hour",
                "mean",
            ),
            p10_cost_per_credit_hour=(
                "cost_per_credit_hour",
                lambda x: x.quantile(0.10),
            ),
            p25_cost_per_credit_hour=(
                "cost_per_credit_hour",
                lambda x: x.quantile(0.25),
            ),
            median_cost_per_credit_hour=(
                "cost_per_credit_hour",
                "median",
            ),
            p75_cost_per_credit_hour=(
                "cost_per_credit_hour",
                lambda x: x.quantile(0.75),
            ),
            p90_cost_per_credit_hour=(
                "cost_per_credit_hour",
                lambda x: x.quantile(0.90),
            ),
        )
        .reset_index()
    )

    benchmark = round_numeric_columns(benchmark)
    benchmark = benchmark.sort_values(
        [
            "year",
            "carnegie_classification",
            "cip_code",
        ]
    )

    quality_report = pd.DataFrame(quality_rows)

    write_csv_and_json(public_metrics, "public-institution-metrics")
    write_csv_and_json(benchmark, "benchmark-database")
    write_csv_and_json(quality_report, "data-quality-report")

    print("Built processed public database files:")
    print("- data/processed/public-institution-metrics.csv")
    print("- data/processed/public-institution-metrics.json")
    print("- data/processed/benchmark-database.csv")
    print("- data/processed/benchmark-database.json")
    print("- data/processed/data-quality-report.csv")
    print("- data/processed/data-quality-report.json")


if __name__ == "__main__":
    main()
