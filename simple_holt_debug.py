# simple_holt_debug.py
"""Simple standalone Holt linearity test."""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
import matplotlib.pyplot as plt


def test_holt_linearity():
    """Test if Holt exponential smoothing produces linear forecasts."""

    print("🧪 TESTING HOLT LINEARITY - STANDALONE TEST")
    print("=" * 60)

    # Create synthetic polling-like data (similar to your election data)
    np.random.seed(42)

    # Trump-like data: starts around 47%, slight downward trend with noise
    trump_base = 47.5
    trump_trend = -0.01  # Slight decline
    trump_noise = 0.5

    # Harris-like data: starts around 48%, slight upward trend with noise
    harris_base = 48.2
    harris_trend = 0.008  # Slight increase
    harris_noise = 0.4

    n_days = 90  # 90 days of training data

    # Generate synthetic data
    trump_data = []
    harris_data = []

    for i in range(n_days):
        trump_val = trump_base + trump_trend * i + np.random.normal(0, trump_noise)
        harris_val = harris_base + harris_trend * i + np.random.normal(0, harris_noise)
        trump_data.append(trump_val)
        harris_data.append(harris_val)

    trump_series = np.array(trump_data)
    harris_series = np.array(harris_data)

    print(f"📊 Generated {n_days} days of synthetic polling data")
    print(f"Trump: {trump_series[-5:]} (last 5 values)")
    print(f"Harris: {harris_series[-5:]} (last 5 values)")

    # Test different hyperparameter combinations
    test_params = [
        {"alpha": 0.1, "beta": 0.1},
        {"alpha": 0.2, "beta": 0.1},
        {"alpha": 0.3, "beta": 0.1},
        {"alpha": 0.1, "beta": 0.2},
        {"alpha": 0.3, "beta": 0.4},  # Higher values from your log
    ]

    forecast_horizon = 13  # Days to election

    for i, params in enumerate(test_params):
        print(f"\n{'='*50}")
        print(f"TEST {i+1}: α={params['alpha']}, β={params['beta']}")
        print(f"{'='*50}")

        # Test Trump model
        print("\n🔴 TRUMP MODEL:")
        trump_result = test_single_model(
            trump_series, params, forecast_horizon, "Trump"
        )

        # Test Harris model
        print("\n🔵 HARRIS MODEL:")
        harris_result = test_single_model(
            harris_series, params, forecast_horizon, "Harris"
        )

        # Summary for this parameter set
        print(f"\n📋 SUMMARY FOR TEST {i+1}:")
        print(f"Trump linear: {trump_result['is_linear']}")
        print(f"Harris linear: {harris_result['is_linear']}")

        if trump_result["is_linear"] and harris_result["is_linear"]:
            print("✅ Both models produce linear forecasts")
        else:
            print("❌ Non-linear forecasts detected!")

    return True


def test_single_model(data, params, horizon, name):
    """Test a single Holt model for linearity."""

    try:
        # Fit Holt model
        model = Holt(data).fit(
            smoothing_level=params["alpha"],
            smoothing_trend=params["beta"],
            optimized=False,
        )

        # Get model components
        final_level = model.level
        final_trend = model.trend

        print(f"Final Level (L_t): {float(final_level):.6f}")
        print(f"Final Trend (T_t): {float(final_trend):.6f}")

        # Generate forecast
        forecast = model.forecast(horizon)

        # Manual calculation for comparison
        manual_forecast = []
        for h in range(1, horizon + 1):
            manual_val = final_level + h * final_trend
            manual_forecast.append(manual_val)

        manual_forecast = np.array(manual_forecast)

        # Compare manual vs model forecast
        max_diff = np.max(np.abs(forecast - manual_forecast))
        print(f"Max difference (manual vs model): {max_diff:.8f}")

        # Check if forecast is linear
        if len(forecast) > 2:
            diffs = np.diff(forecast)
            is_linear = np.allclose(diffs, diffs[0], rtol=1e-10)

            print(f"Expected difference: {final_trend:.8f}")
            print(f"Actual differences: {diffs[:3]} ... (showing first 3)")
            print(f"Is linear: {is_linear}")

            if not is_linear:
                print(f"⚠️  NON-LINEAR FORECAST DETECTED!")
                print(f"Difference range: {np.min(diffs):.8f} to {np.max(diffs):.8f}")
        else:
            is_linear = True

        return {
            "is_linear": is_linear,
            "final_level": final_level,
            "final_trend": final_trend,
            "forecast": forecast,
            "manual_forecast": manual_forecast,
            "max_diff": max_diff,
        }

    except Exception as e:
        print(f"❌ Error fitting model: {e}")
        return {"is_linear": False, "error": str(e)}


def create_visualization():
    """Create a simple visualization to show the linearity concept."""

    print(f"\n📈 CREATING LINEARITY VISUALIZATION")
    print("=" * 40)

    # Simple example data
    data = np.array([47.0, 47.2, 46.8, 47.1, 47.3, 47.0, 47.4])

    # Fit model
    model = Holt(data).fit(smoothing_level=0.3, smoothing_trend=0.1, optimized=False)
    forecast = model.forecast(5)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot historical data
    plt.plot(range(len(data)), data, "o-", label="Historical Data", linewidth=2)

    # Plot forecast
    forecast_x = range(len(data), len(data) + len(forecast))
    plt.plot(
        forecast_x, forecast, "r--", label="Holt Forecast", linewidth=2, marker="s"
    )

    # Add vertical line at forecast start
    plt.axvline(
        x=len(data) - 0.5,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="Forecast Start",
    )

    # Verify linearity visually by adding trend line
    x_trend = np.arange(len(forecast))
    trend_slope = np.diff(forecast)[0] if len(forecast) > 1 else 0
    trend_line = forecast[0] + trend_slope * x_trend
    plt.plot(
        forecast_x,
        trend_line,
        "b:",
        alpha=0.7,
        label=f"Linear Trend (slope={trend_slope:.4f})",
    )

    plt.title("Holt Forecast Linearity Test")
    plt.xlabel("Time Period")
    plt.ylabel("Polling %")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    import os

    os.makedirs("outputs/debug_plots", exist_ok=True)
    plt.savefig(
        "outputs/debug_plots/holt_linearity_test.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    print("💾 Saved visualization: outputs/debug_plots/holt_linearity_test.png")

    # Print forecast details
    print(f"Forecast values: {forecast}")
    if len(forecast) > 1:
        diffs = np.diff(forecast)
        print(f"Forecast differences: {diffs}")
        print(f"All differences equal: {np.allclose(diffs, diffs[0], rtol=1e-10)}")


if __name__ == "__main__":
    print("🚀 STANDALONE HOLT LINEARITY DIAGNOSTICS")
    print("=" * 60)

    # Run the main test
    test_holt_linearity()

    # Create visualization
    create_visualization()

    print(f"\n🎉 DIAGNOSTIC COMPLETE!")
    print("=" * 30)
    print("Key findings:")
    print("- If all tests show 'linear: True', then Holt is working correctly")
    print(
        "- The 'curved' appearance in your main plot is likely from overlaying multiple days"
    )
    print("- Check the visualization in outputs/debug_plots/holt_linearity_test.png")
    print("\nNext step: Run your main pipeline to see if it completes without errors")
