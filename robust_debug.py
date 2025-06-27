# robust_debug.py
"""Robust debug script with error handling."""

import sys


def main():
    print("🚀 STARTING ROBUST DEBUG SCRIPT")
    print("=" * 50)

    try:
        import numpy as np

        print("✅ numpy imported")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return

    try:
        import matplotlib.pyplot as plt

        print("✅ matplotlib imported")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return

    try:
        from pathlib import Path

        print("✅ pathlib imported")
    except ImportError as e:
        print(f"❌ pathlib import failed: {e}")
        return

    try:
        import os

        print(f"📁 Current directory: {os.getcwd()}")

        # Check if outputs directory exists
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            print(f"✅ outputs directory exists")
        else:
            print(f"❌ outputs directory missing, creating...")
            outputs_dir.mkdir(exist_ok=True)

        # Create debug directory
        debug_dir = Path("outputs/debug_plots")
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created debug directory: {debug_dir}")

        # Test if we can write to it
        test_file = debug_dir / "test.txt"
        with open(test_file, "w") as f:
            f.write("test")

        if test_file.exists():
            print(f"✅ Can write to debug directory")
            test_file.unlink()  # Clean up
        else:
            print(f"❌ Cannot write to debug directory")
            return

    except Exception as e:
        print(f"❌ Directory setup failed: {e}")
        return

    # Create simple linear test
    print(f"\n🔍 CREATING SIMPLE LINEAR TEST")
    print("-" * 30)

    try:
        # Your diagnostic slopes
        trump_slope = 0.02273504
        harris_slope = -0.00327424

        # Create 13 periods of data
        periods = 13
        trump_start = 47.5
        harris_start = 48.5

        # Generate linear data
        trump_forecast = []
        harris_forecast = []

        for i in range(periods):
            trump_val = trump_start + i * trump_slope
            harris_val = harris_start + i * harris_slope
            trump_forecast.append(trump_val)
            harris_forecast.append(harris_val)

        print(f"✅ Generated forecast data")
        print(f"Trump first 3 values: {trump_forecast[:3]}")
        print(f"Harris first 3 values: {harris_forecast[:3]}")

        # Check if data is linear
        trump_diffs = []
        harris_diffs = []

        for i in range(1, len(trump_forecast)):
            trump_diffs.append(trump_forecast[i] - trump_forecast[i - 1])
            harris_diffs.append(harris_forecast[i] - harris_forecast[i - 1])

        print(f"Trump differences: {trump_diffs[:3]} (should all be {trump_slope:.6f})")
        print(
            f"Harris differences: {harris_diffs[:3]} (should all be {harris_slope:.6f})"
        )

        # Test if all differences are the same
        trump_linear = all(abs(d - trump_slope) < 1e-10 for d in trump_diffs)
        harris_linear = all(abs(d - harris_slope) < 1e-10 for d in harris_diffs)

        print(f"Trump is linear: {trump_linear}")
        print(f"Harris is linear: {harris_linear}")

    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return

    # Create plot
    print(f"\n🎨 CREATING PLOT")
    print("-" * 20)

    try:
        x_values = list(range(periods))

        plt.figure(figsize=(10, 6))

        # Plot Trump
        plt.subplot(2, 1, 1)
        plt.plot(x_values, trump_forecast, "r-", linewidth=2, marker="o", markersize=4)
        plt.title(f"Trump Forecast (slope={trump_slope:.6f})")
        plt.xlabel("Period")
        plt.ylabel("Polling %")
        plt.grid(True, alpha=0.3)

        # Plot Harris
        plt.subplot(2, 1, 2)
        plt.plot(x_values, harris_forecast, "b-", linewidth=2, marker="o", markersize=4)
        plt.title(f"Harris Forecast (slope={harris_slope:.6f})")
        plt.xlabel("Period")
        plt.ylabel("Polling %")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        print("✅ Plot created")

    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
        return

    # Save plot
    try:
        save_path = debug_dir / "linear_test_robust.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if save_path.exists():
            print(f"✅ Plot saved to: {save_path}")
            print(f"📊 File size: {save_path.stat().st_size} bytes")
        else:
            print(f"❌ Plot file not found after saving")

    except Exception as e:
        print(f"❌ Plot saving failed: {e}")
        return

    print(f"\n🎉 DEBUG SCRIPT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Check this file: outputs/debug_plots/linear_test_robust.png")

    # List all files in debug directory
    try:
        debug_files = list(debug_dir.glob("*"))
        print(f"\nFiles in debug directory:")
        for file in debug_files:
            print(f"  - {file.name}")
    except Exception as e:
        print(f"Error listing debug files: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ SCRIPT FAILED: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n⏸️  Press Enter to exit...")
    input()
