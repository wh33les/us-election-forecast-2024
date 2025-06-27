# csv_analysis.py
"""Analyze the comprehensive CSV and suggest optimizations."""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_csv_size():
    """Analyze what's making the CSV so large."""

    print("📊 ANALYZING CSV SIZE AND CONTENT")
    print("=" * 50)

    # Load the data
    csv_path = "data/election_forecast_2024_comprehensive.csv"

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
        print(f"📏 File size: {Path(csv_path).stat().st_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

    # Analyze by record type
    print(f"\n📋 BREAKDOWN BY RECORD TYPE:")
    record_counts = df["record_type"].value_counts()
    for record_type, count in record_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {record_type}: {count:,} rows ({percentage:.1f}%)")

    # Analyze by forecast run date
    print(f"\n📅 BREAKDOWN BY FORECAST DATE:")
    date_counts = df["forecast_run_date"].value_counts().sort_index()
    for date, count in date_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {date}: {count:,} rows ({percentage:.1f}%)")

    # Analyze columns with mostly null values
    print(f"\n🔍 COLUMNS WITH MANY NULL VALUES:")
    null_percentages = (df.isnull().sum() / len(df)) * 100
    high_null_cols = null_percentages[null_percentages > 80].sort_values(
        ascending=False
    )

    if len(high_null_cols) > 0:
        for col, null_pct in high_null_cols.items():
            print(f"   {col}: {null_pct:.1f}% null")
    else:
        print("   ✅ No columns with >80% null values")

    # Memory usage by column
    print(f"\n💾 MEMORY USAGE BY COLUMN:")
    memory_usage = df.memory_usage(deep=True).sort_values(ascending=False)
    for col, memory in memory_usage.head(10).items():
        mb = memory / 1024 / 1024
        print(f"   {col}: {mb:.2f} MB")

    return df


def create_optimized_versions(df):
    """Create optimized versions of the dataset."""

    print(f"\n🎯 CREATING OPTIMIZED VERSIONS")
    print("=" * 40)

    # Version 1: Forecasts Only (smallest)
    forecasts_only = df[df["is_forecast"] == True].copy()

    # Keep only essential columns for forecasts
    essential_forecast_cols = [
        "date",
        "candidate",
        "forecast_run_date",
        "model_prediction",
        "baseline_prediction",
        "days_to_election",
        "alpha",
        "beta",
        "mase_score",
        "electoral_winner_model",
        "electoral_votes_trump_model",
        "electoral_votes_harris_model",
    ]

    forecasts_only = forecasts_only[essential_forecast_cols]

    print(f"📊 Version 1 - Forecasts Only:")
    print(f"   Rows: {len(forecasts_only):,} (was {len(df):,})")
    print(f"   Columns: {len(forecasts_only.columns)} (was {len(df.columns)})")
    print(
        f"   Size reduction: {((len(df) - len(forecasts_only)) / len(df)) * 100:.1f}% fewer rows"
    )

    # Version 2: Summary by Date (tiny)
    summary_by_date = (
        df[df["is_forecast"] == True]
        .groupby(["forecast_run_date", "candidate"])
        .agg(
            {
                "model_prediction": "first",
                "baseline_prediction": "first",
                "days_to_election": "first",
                "alpha": "first",
                "beta": "first",
                "mase_score": "first",
                "electoral_winner_model": "first",
                "electoral_votes_trump_model": "first",
                "electoral_votes_harris_model": "first",
            }
        )
        .reset_index()
    )

    print(f"\n📊 Version 2 - Daily Summary:")
    print(f"   Rows: {len(summary_by_date):,} (was {len(df):,})")
    print(f"   Columns: {len(summary_by_date.columns)} (was {len(df.columns)})")
    print(
        f"   Size reduction: {((len(df) - len(summary_by_date)) / len(df)) * 100:.1f}% fewer rows"
    )

    # Version 3: Election Day Only (minimal)
    election_day_only = df[
        (df["date"] == "2024-11-05") & (df["is_forecast"] == True)
    ].copy()

    print(f"\n📊 Version 3 - Election Day Only:")
    print(f"   Rows: {len(election_day_only):,} (was {len(df):,})")
    print(
        f"   Size reduction: {((len(df) - len(election_day_only)) / len(df)) * 100:.1f}% fewer rows"
    )

    return {
        "forecasts_only": forecasts_only,
        "daily_summary": summary_by_date,
        "election_day_only": election_day_only,
    }


def save_optimized_versions(versions):
    """Save the optimized versions."""

    print(f"\n💾 SAVING OPTIMIZED VERSIONS")
    print("-" * 30)

    output_dir = Path("data/optimized")
    output_dir.mkdir(exist_ok=True)

    for name, df in versions.items():
        filename = f"election_forecast_2024_{name}.csv"
        filepath = output_dir / filename

        try:
            df.to_csv(filepath, index=False)
            file_size = filepath.stat().st_size / 1024

            print(f"✅ Saved {name}:")
            print(f"   File: {filepath}")
            print(f"   Size: {file_size:.1f} KB")
            print(f"   Rows: {len(df):,}")

        except Exception as e:
            print(f"❌ Error saving {name}: {e}")


def suggest_data_manager_optimization():
    """Suggest changes to reduce data collection."""

    print(f"\n🔧 SUGGESTED DATA_MANAGER.PY OPTIMIZATIONS")
    print("=" * 50)

    suggestions = [
        "1. REDUCE HISTORICAL POLLING STORAGE:",
        "   - Currently storing all training data for each forecast",
        "   - Consider storing only unique dates (no duplicates)",
        "   - Or skip historical polling entirely (keep only forecasts)",
        "",
        "2. ELIMINATE FITTED VALUES:",
        "   - 'model_fitted' records are mainly for diagnostics",
        "   - Skip these in production runs",
        "   - Saves ~50% of records",
        "",
        "3. STORE ONLY ELECTION DAY FORECASTS:",
        "   - Keep only final prediction for each forecast date",
        "   - Skip intermediate forecast dates",
        "   - Reduces from ~400 to ~28 records per run",
        "",
        "4. CONSOLIDATE ELECTORAL DATA:",
        "   - Combine multiple electoral columns into JSON",
        "   - Or store electoral data in separate file",
        "",
        "5. USE EFFICIENT DATA TYPES:",
        "   - Convert string dates to datetime",
        "   - Use categorical for repeated strings",
        "   - Use float32 instead of float64",
    ]

    for suggestion in suggestions:
        print(suggestion)


if __name__ == "__main__":
    print("🔍 COMPREHENSIVE CSV OPTIMIZATION ANALYSIS")
    print("=" * 60)

    # Analyze current CSV
    df = analyze_csv_size()

    if df is not None:
        # Create optimized versions
        versions = create_optimized_versions(df)

        # Save them
        save_optimized_versions(versions)

        # Suggest code optimizations
        suggest_data_manager_optimization()

        print(f"\n🎯 RECOMMENDATIONS:")
        print("=" * 20)
        print("• Use 'daily_summary.csv' for most analysis (tiny file)")
        print("• Use 'forecasts_only.csv' for detailed forecast analysis")
        print("• Use 'election_day_only.csv' for final predictions only")
        print("• Modify data_manager.py to reduce future data collection")

    print(f"\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
