# check_csv_for_plotting.py
"""Check which CSV files have the data needed for plotting."""

import pandas as pd
from pathlib import Path


def check_csv_files():
    """Check which CSV files work for different plot types."""

    print("📊 CHECKING CSV FILES FOR PLOTTING REQUIREMENTS")
    print("=" * 60)

    # List of CSV files to check
    csv_files = [
        "data/election_forecast_2024_comprehensive.csv",
        "data/election_forecast_2024_SMALL.csv",
        "data/election_forecast_2024_TINY.csv",
        "data/optimized/election_forecast_2024_forecasts_only.csv",
        "data/optimized/election_forecast_2024_daily_summary.csv",
        "data/optimized/election_forecast_2024_election_day_only.csv",
    ]

    results = {}

    for csv_file in csv_files:
        path = Path(csv_file)
        if path.exists():
            try:
                df = pd.read_csv(path)
                analysis = analyze_csv_for_plotting(df, path.name)
                results[path.name] = analysis

            except Exception as e:
                print(f"❌ Error reading {path.name}: {e}")
        else:
            print(f"❌ File not found: {path}")

    # Summary recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 20)

    best_for_historical = None
    smallest_working = None

    for filename, analysis in results.items():
        if analysis["works_for_historical"]:
            if best_for_historical is None:
                best_for_historical = filename
            if (
                smallest_working is None
                or analysis["size_mb"] < results[smallest_working]["size_mb"]
            ):
                smallest_working = filename

    if best_for_historical:
        print(f"✅ BEST FOR HISTORICAL PLOTS: {best_for_historical}")
        print(
            f"   - Has {results[best_for_historical]['forecast_dates']} forecast dates"
        )
        print(f"   - Size: {results[best_for_historical]['size_mb']:.1f} MB")
        print(f"   - Use this for: plot_historical_forecasts()")

    if smallest_working:
        print(f"✅ SMALLEST WORKING FILE: {smallest_working}")
        print(f"   - Size: {results[smallest_working]['size_mb']:.2f} MB")

    print(f"\n💡 REMEMBER:")
    print(f"   - Main forecast plots (straight lines) don't need any CSV")
    print(f"   - They're generated during: python main.py --date XX-XX")
    print(f"   - CSV is only needed for historical comparison plots")

    return results


def analyze_csv_for_plotting(df, filename):
    """Analyze if a CSV file has the data needed for plotting."""

    print(f"\n📋 {filename}")
    print("-" * 40)

    # Basic info
    file_size = Path("data") / filename
    if file_size.exists():
        size_mb = file_size.stat().st_size / 1024 / 1024
    else:
        # Try other locations
        for parent in ["data/optimized", "data"]:
            test_path = Path(parent) / filename
            if test_path.exists():
                size_mb = test_path.stat().st_size / 1024 / 1024
                break
        else:
            size_mb = 0

    print(f"📊 Rows: {len(df):,}, Columns: {len(df.columns)}, Size: {size_mb:.2f} MB")

    # Check required columns for plotting
    required_cols = ["date", "candidate", "forecast_run_date", "model_prediction"]
    has_required = all(col in df.columns for col in required_cols)

    print(f"📋 Has required columns: {has_required}")

    if not has_required:
        missing = [col for col in required_cols if col not in df.columns]
        print(f"   Missing: {missing}")
        return {
            "works_for_historical": False,
            "size_mb": size_mb,
            "forecast_dates": 0,
            "election_day_forecasts": 0,
        }

    # Check for forecast data
    if "is_forecast" in df.columns:
        forecast_data = df[df["is_forecast"] == True]
    else:
        # Assume all data is forecast data if no is_forecast column
        forecast_data = df

    # Check for Election Day forecasts from multiple dates
    if "2024-11-05" in df["date"].values:
        election_day_forecasts = df[df["date"] == "2024-11-05"]
        unique_forecast_dates = election_day_forecasts["forecast_run_date"].nunique()
    else:
        unique_forecast_dates = 0
        election_day_forecasts = pd.DataFrame()

    print(f"📅 Unique forecast run dates: {df['forecast_run_date'].nunique()}")
    print(f"🗳️ Election Day forecasts: {len(election_day_forecasts)}")
    print(f"📈 Election Day from different dates: {unique_forecast_dates}")

    # Determine if works for historical plotting
    works_for_historical = (
        has_required
        and unique_forecast_dates >= 2  # Need at least 2 different forecast dates
        and len(election_day_forecasts)
        >= 4  # Need forecasts for both candidates from multiple dates
    )

    if works_for_historical:
        print(f"✅ WORKS for historical forecasts plot")
    else:
        print(f"❌ NOT suitable for historical forecasts plot")
        if unique_forecast_dates < 2:
            print(f"   Reason: Only {unique_forecast_dates} forecast date(s)")
        if len(election_day_forecasts) < 4:
            print(
                f"   Reason: Only {len(election_day_forecasts)} Election Day forecast(s)"
            )

    return {
        "works_for_historical": works_for_historical,
        "size_mb": size_mb,
        "forecast_dates": unique_forecast_dates,
        "election_day_forecasts": len(election_day_forecasts),
    }


if __name__ == "__main__":
    results = check_csv_files()

    print(f"\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE")
    print("=" * 60)
