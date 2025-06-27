# simple_forecast_debug.py
"""Create simple debug plots showing raw forecast data without connection artifacts."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta


def create_simple_forecast_debug():
    """Create simple plots showing raw forecast linearity."""

    print("🔍 CREATING SIMPLE FORECAST DEBUG PLOTS")
    print("=" * 50)

    # Create output directory
    debug_dir = Path("outputs/debug_plots")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Test with simple linear data that matches your diagnostics
    forecast_periods = 13

    # Trump: positive slope (+0.02273504 from diagnostics)
    trump_start = 47.5
    trump_slope = 0.02273504
    trump_forecast = [trump_start + i * trump_slope for i in range(forecast_periods)]

    # Harris: negative slope (-0.00327424 from diagnostics)
    harris_start = 48.5
    harris_slope = -0.00327424
    harris_forecast = [harris_start + i * harris_slope for i in range(forecast_periods)]

    # Create simple x-axis (just period numbers)
    periods = list(range(forecast_periods))

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Trump plot
    ax1.plot(periods, trump_forecast, "r-", linewidth=2, marker="o", markersize=4)
    ax1.set_title("Trump Forecast - Should Be Perfectly Linear")
    ax1.set_xlabel("Forecast Period")
    ax1.set_ylabel("Polling %")
    ax1.grid(True, alpha=0.3)

    # Add trend line to verify
    coeffs = np.polyfit(periods, trump_forecast, 1)
    trend_line = np.polyval(coeffs, periods)
    ax1.plot(
        periods, trend_line, "b--", alpha=0.7, label=f"Trend (slope={coeffs[0]:.6f})"
    )
    ax1.legend()

    # Check if perfectly linear
    diffs = np.diff(trump_forecast)
    is_linear = np.allclose(diffs, diffs[0], rtol=1e-10)
    ax1.text(
        0.02,
        0.98,
        f"Linear: {is_linear}\nSlope: {diffs[0]:.6f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Harris plot
    ax2.plot(periods, harris_forecast, "b-", linewidth=2, marker="o", markersize=4)
    ax2.set_title("Harris Forecast - Should Be Perfectly Linear")
    ax2.set_xlabel("Forecast Period")
    ax2.set_ylabel("Polling %")
    ax2.grid(True, alpha=0.3)

    # Add trend line to verify
    coeffs = np.polyfit(periods, harris_forecast, 1)
    trend_line = np.polyval(coeffs, periods)
    ax2.plot(
        periods, trend_line, "r--", alpha=0.7, label=f"Trend (slope={coeffs[0]:.6f})"
    )
    ax2.legend()

    # Check if perfectly linear
    diffs = np.diff(harris_forecast)
    is_linear = np.allclose(diffs, diffs[0], rtol=1e-10)
    ax2.text(
        0.02,
        0.98,
        f"Linear: {is_linear}\nSlope: {diffs[0]:.6f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    # Save plot
    simple_path = debug_dir / "simple_linear_test.png"
    plt.savefig(simple_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved simple linear test: {simple_path}")

    # Now create a plot showing the CONNECTION ARTIFACT issue
    create_connection_artifact_demo()


def create_connection_artifact_demo():
    """Demonstrate how connection logic can make linear forecasts appear curved."""

    print("\n🎨 CREATING CONNECTION ARTIFACT DEMO")
    print("=" * 40)

    debug_dir = Path("outputs/debug_plots")

    # Create linear forecast data
    forecast_periods = 13
    trump_start = 47.5
    trump_slope = 0.02273504
    trump_forecast = [trump_start + i * trump_slope for i in range(forecast_periods)]

    # Simulate your plotting code's connection logic
    last_historical_value = 46.8  # Different from forecast start
    last_historical_date = 0
    forecast_dates = list(range(1, forecast_periods + 1))

    # Method 1: Pure linear forecast (should be straight)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(
        forecast_dates,
        trump_forecast,
        "r--",
        linewidth=2,
        marker="o",
        label="Pure Linear Forecast",
    )
    ax1.set_title("Method 1: Pure Linear Forecast\n(Should be straight)")
    ax1.set_xlabel("Forecast Period")
    ax1.set_ylabel("Polling %")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Method 2: Connected forecast (like your plotting code)
    connected_dates = [last_historical_date] + forecast_dates
    connected_values = [last_historical_value] + trump_forecast

    ax2.plot(
        connected_dates,
        connected_values,
        "r--",
        linewidth=2,
        marker="o",
        label="Connected Forecast",
    )
    ax2.set_title(
        "Method 2: Connected Forecast\n(Can appear curved due to connection point)"
    )
    ax2.set_xlabel("Forecast Period")
    ax2.set_ylabel("Polling %")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Highlight the connection point
    ax2.plot(
        last_historical_date,
        last_historical_value,
        "go",
        markersize=8,
        label="Connection Point",
    )
    ax2.legend()

    plt.tight_layout()

    # Save plot
    connection_path = debug_dir / "connection_artifact_demo.png"
    plt.savefig(connection_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved connection artifact demo: {connection_path}")

    # Print analysis
    print(f"\n📊 ANALYSIS:")
    print(
        f"Pure forecast differences: {np.diff(trump_forecast)[:3]} (should be constant)"
    )
    print(
        f"Connected forecast differences: {np.diff(connected_values)[:3]} (first diff is different!)"
    )

    pure_diffs = np.diff(trump_forecast)
    connected_diffs = np.diff(connected_values)

    print(
        f"Pure forecast is linear: {np.allclose(pure_diffs, pure_diffs[0], rtol=1e-10)}"
    )
    print(
        f"Connected forecast is linear: {np.allclose(connected_diffs, connected_diffs[0], rtol=1e-10)}"
    )


def create_date_spacing_test():
    """Test if non-uniform date spacing could cause visual artifacts."""

    print(f"\n📅 TESTING DATE SPACING ARTIFACTS")
    print("=" * 40)

    debug_dir = Path("outputs/debug_plots")

    # Create forecast data
    forecast_periods = 13
    trump_start = 47.5
    trump_slope = 0.02273504
    trump_forecast = [trump_start + i * trump_slope for i in range(forecast_periods)]

    # Test with different date spacings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Uniform spacing
    uniform_dates = pd.date_range("2024-10-24", periods=forecast_periods, freq="D")
    ax1.plot(uniform_dates, trump_forecast, "r--", linewidth=2, marker="o")
    ax1.set_title("Uniform Date Spacing\n(Should be straight)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Polling %")
    ax1.grid(True, alpha=0.3)

    # Non-uniform spacing (weekends skipped)
    non_uniform_dates = pd.bdate_range("2024-10-24", periods=forecast_periods, freq="B")
    ax2.plot(non_uniform_dates, trump_forecast, "r--", linewidth=2, marker="o")
    ax2.set_title("Non-uniform Date Spacing\n(Business days only)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Polling %")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    spacing_path = debug_dir / "date_spacing_test.png"
    plt.savefig(spacing_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved date spacing test: {spacing_path}")


if __name__ == "__main__":
    create_simple_forecast_debug()
    create_date_spacing_test()

    print(f"\n🎉 DEBUG PLOTS CREATED!")
    print("=" * 30)
    print("Check these files:")
    print("- outputs/debug_plots/simple_linear_test.png")
    print("- outputs/debug_plots/connection_artifact_demo.png")
    print("- outputs/debug_plots/date_spacing_test.png")
    print("\nThese will show:")
    print("1. Whether pure linear data plots as straight lines")
    print("2. How connection logic can create curved appearance")
    print("3. Whether date spacing affects visual linearity")
