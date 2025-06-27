# test_gap_handling.py
"""Test what happens when there are gaps in the comprehensive dataset."""

import pandas as pd
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime


def create_gap_scenario():
    """Create a comprehensive CSV with gaps to test incremental loading."""

    print("🔍 TESTING GAP HANDLING IN INCREMENTAL LOADING")
    print("=" * 60)

    # Load current comprehensive dataset
    csv_path = Path("data/election_forecast_2024_comprehensive.csv")

    if not csv_path.exists():
        print(f"❌ Comprehensive CSV not found: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    print(f"📊 Original dataset: {len(df)} rows")

    # Check what forecast dates we have
    forecast_dates = sorted(df["forecast_run_date"].unique())
    print(f"📅 Available forecast dates: {forecast_dates}")

    # Create a version with gaps (simulate your scenario)
    # Remove 10-24 data and anything after 10-27
    gap_df = df[
        (df["forecast_run_date"] != "2024-10-24")  # Remove 10-24
        & (df["forecast_run_date"] <= "2024-10-27")  # Remove anything after 10-27
    ].copy()

    remaining_dates = sorted(gap_df["forecast_run_date"].unique())
    print(f"📅 After creating gaps: {remaining_dates}")
    print(f"📊 Gap dataset: {len(gap_df)} rows")

    # Save the gap version
    gap_path = Path("data/election_forecast_2024_comprehensive_WITH_GAPS.csv")
    gap_df.to_csv(gap_path, index=False)

    print(f"✅ Created gap scenario: {gap_path}")
    return gap_path


def test_gap_loading():
    """Test what happens when we request data beyond the gaps."""

    print(f"\n🧪 TESTING GAP LOADING BEHAVIOR")
    print("-" * 40)

    # Create the gap scenario
    gap_path = create_gap_scenario()
    if not gap_path:
        return

    # Backup original file
    original_path = Path("data/election_forecast_2024_comprehensive.csv")
    backup_path = Path("data/election_forecast_2024_comprehensive_BACKUP.csv")

    print(f"💾 Backing up original to: {backup_path}")
    import shutil

    shutil.copy2(original_path, backup_path)

    # Replace with gap version
    print(f"🔄 Replacing comprehensive CSV with gap version...")
    shutil.copy2(gap_path, original_path)

    try:
        # Test loading for 10-30 (beyond available data)
        print(f"\n🚀 Testing: python main.py --date 10-30 --verbose")
        print(f"Expected behavior: Should detect missing data and fill gaps")

        start_time = time.time()

        result = subprocess.run(
            [sys.executable, "main.py", "--date", "10-30", "--verbose"],
            capture_output=True,
            text=True,
            timeout=180,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"⏱️ Duration: {duration:.1f} seconds")

        if result.returncode == 0:
            print(f"✅ Pipeline completed successfully!")

            # Check for key indicators in the output
            output_lines = result.stdout.split("\n")

            # Look for incremental vs full loading indicators
            incremental_msgs = [
                line for line in output_lines if "incremental" in line.lower()
            ]
            loading_msgs = [
                line
                for line in output_lines
                if any(
                    word in line
                    for word in ["loading", "found", "missing", "need data"]
                )
            ]

            print(f"\n📊 LOADING MESSAGES:")
            for msg in loading_msgs[:10]:  # Show first 10
                print(f"   {msg}")

            # Determine what happened based on duration
            if duration < 10:
                print(
                    f"\n⚡ FAST EXECUTION ({duration:.1f}s) - Likely incremental loading"
                )
            elif duration > 30:
                print(
                    f"\n🐌 SLOW EXECUTION ({duration:.1f}s) - Likely full reprocessing"
                )
            else:
                print(f"\n⚖️ MEDIUM EXECUTION ({duration:.1f}s) - Likely gap filling")

        else:
            print(f"❌ Pipeline failed!")
            print(f"Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print(f"⏱️ Pipeline timed out (>3 minutes)")
        print(f"This suggests full reprocessing occurred")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Restore original file
        print(f"\n🔄 Restoring original comprehensive CSV...")
        shutil.copy2(backup_path, original_path)
        print(f"✅ Original file restored")

        # Clean up
        gap_path.unlink()
        backup_path.unlink()


def check_current_data_coverage():
    """Check what data is currently available."""

    print(f"\n📋 CURRENT DATA COVERAGE")
    print("-" * 30)

    csv_path = Path("data/election_forecast_2024_comprehensive.csv")

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        forecast_dates = sorted(df["forecast_run_date"].unique())
        print(f"📅 Available forecast dates:")
        for date in forecast_dates:
            count = len(df[df["forecast_run_date"] == date])
            print(f"   {date}: {count} records")

        print(f"\n💡 To test gap handling:")
        print(f"   - You have data up to: {max(forecast_dates)}")
        print(f"   - Try running: python main.py --date 11-01 --verbose")
        print(f"   - This will test loading beyond available data")

        return forecast_dates
    else:
        print(f"❌ No comprehensive CSV found")
        return []


if __name__ == "__main__":
    # First check current coverage
    available_dates = check_current_data_coverage()

    if available_dates:
        print(f"\n" + "=" * 60)
        response = input(
            "Run gap handling test? This will temporarily modify your CSV (y/n): "
        )

        if response.lower() == "y":
            test_gap_loading()
        else:
            print(f"💡 To test manually:")
            print(f"   python main.py --date 11-01 --verbose")
            print(f"   (This requests data beyond your current coverage)")

    print(f"\n" + "=" * 60)
    print("🎯 GAP HANDLING TEST COMPLETE")
    print("=" * 60)
