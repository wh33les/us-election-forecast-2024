# quick_csv_fix.py
"""Immediately reduce CSV size without code changes."""

import pandas as pd
from pathlib import Path


def quick_reduce_csv():
    """Create smaller versions of the existing CSV."""

    print("🚀 QUICK CSV SIZE REDUCTION")
    print("=" * 40)

    # Load original
    original_path = Path("data/election_forecast_2024_comprehensive.csv")

    if not original_path.exists():
        print(f"❌ File not found: {original_path}")
        return

    df = pd.read_csv(original_path)
    original_size = original_path.stat().st_size / 1024 / 1024

    print(f"📊 Original: {len(df):,} rows, {original_size:.1f} MB")

    # Create forecasts-only version (90% smaller)
    forecasts_df = df[df["is_forecast"] == True].copy()

    # Drop unnecessary columns
    essential_cols = [
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
        "prediction_correct",
    ]

    # Keep only columns that exist
    available_cols = [col for col in essential_cols if col in forecasts_df.columns]
    forecasts_clean = forecasts_df[available_cols].copy()

    # Save reduced version
    reduced_path = Path("data/election_forecast_2024_SMALL.csv")
    forecasts_clean.to_csv(reduced_path, index=False)

    new_size = reduced_path.stat().st_size / 1024 / 1024
    reduction = ((original_size - new_size) / original_size) * 100

    print(f"✅ Reduced: {len(forecasts_clean):,} rows, {new_size:.1f} MB")
    print(f"🎯 Size reduction: {reduction:.1f}% smaller!")
    print(f"📁 Saved as: {reduced_path}")

    # Create tiny summary version
    summary_df = (
        forecasts_clean.groupby(["forecast_run_date", "candidate"])
        .first()
        .reset_index()
    )
    summary_path = Path("data/election_forecast_2024_TINY.csv")
    summary_df.to_csv(summary_path, index=False)

    tiny_size = summary_path.stat().st_size / 1024
    print(f"✅ Tiny version: {len(summary_df):,} rows, {tiny_size:.1f} KB")
    print(f"📁 Saved as: {summary_path}")

    return {
        "original_mb": original_size,
        "small_mb": new_size,
        "tiny_kb": tiny_size,
        "reduction_pct": reduction,
    }


if __name__ == "__main__":
    result = quick_reduce_csv()

    if result:
        print(f"\n🎉 SUCCESS!")
        print(f"💡 Use 'election_forecast_2024_SMALL.csv' for analysis")
        print(f"💡 Use 'election_forecast_2024_TINY.csv' for quick summaries")
        print(f"💡 Original file kept as backup")
