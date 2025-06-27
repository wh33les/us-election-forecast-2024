# debug_holt_linearity.py
"""Quick debug script to test Holt linearity on a single day."""

import pandas as pd
import numpy as np
from datetime import date
from src.models.holt_forecaster import HoltElectionForecaster
from src.config.model_config import ModelConfig
from src.data.collectors import DataCollector
from src.data.processors import DataProcessor


def debug_single_day():
    """Debug Holt linearity for a single forecast day."""

    print("🔍 DEBUGGING HOLT LINEARITY - SINGLE DAY TEST")
    print("=" * 60)

    # Initialize components
    config = ModelConfig()
    data_collector = DataCollector()
    data_processor = DataProcessor()
    forecaster = HoltElectionForecaster(config)

    # Test with Oct 23, 2024 (first day from your log)
    test_date = date(2024, 10, 23)
    print(f"Testing forecast for: {test_date}")

    try:
        # Load data up to test date
        daily_averages = data_collector.load_incremental_data(test_date)
        print(f"Loaded {len(daily_averages)} polling records")

        # Split by candidate
        trump_data, harris_data = data_processor.split_by_candidate(daily_averages)
        print(f"Trump records: {len(trump_data)}, Harris records: {len(harris_data)}")

        # Create train/test split (similar to your pipeline)
        train_size = len(trump_data) - 7  # 7-day holdout
        trump_train = trump_data.iloc[:train_size]
        harris_train = harris_data.iloc[:train_size]

        print(f"Training size: {len(trump_train)} days")

        # Run hyperparameter optimization
        print("\n🔧 Running hyperparameter optimization...")
        best_params = forecaster.grid_search_hyperparameters(
            trump_train, harris_train, trump_train.index
        )

        # Fit final models
        print("\n🎯 Fitting final models...")
        models = forecaster.fit_final_models(trump_train, harris_train)

        # Test forecast with diagnostics
        forecast_horizon = 13  # Days until election from Oct 23
        print(f"\n📊 Testing forecast for {forecast_horizon} periods...")

        # Test Trump model
        print("\n" + "=" * 50)
        print("TRUMP MODEL DIAGNOSTICS")
        print("=" * 50)
        trump_diagnostics = forecaster.diagnose_holt_linearity(
            models["trump"], forecast_horizon
        )

        # Test Harris model
        print("\n" + "=" * 50)
        print("HARRIS MODEL DIAGNOSTICS")
        print("=" * 50)
        harris_diagnostics = forecaster.diagnose_holt_linearity(
            models["harris"], forecast_horizon
        )

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Trump forecast is linear: {trump_diagnostics['is_linear']}")
        print(f"Harris forecast is linear: {harris_diagnostics['is_linear']}")
        print(f"Trump max difference: {trump_diagnostics['max_difference']:.8f}")
        print(f"Harris max difference: {harris_diagnostics['max_difference']:.8f}")

        if trump_diagnostics["is_linear"] and harris_diagnostics["is_linear"]:
            print("\n✅ SUCCESS: Both forecasts are perfectly linear!")
            print("The issue is likely in visualization or data overlay.")
        else:
            print("\n❌ ISSUE FOUND: Forecasts are not linear!")
            print("This indicates a problem with the Holt implementation.")

        return {
            "trump": trump_diagnostics,
            "harris": harris_diagnostics,
            "models": models,
        }

    except Exception as e:
        print(f"\n❌ ERROR during debugging: {e}")
        import traceback

        traceback.print_exc()
        return None


def quick_manual_test():
    """Quick manual test with synthetic data."""
    print("\n🧪 QUICK MANUAL TEST WITH SYNTHETIC DATA")
    print("=" * 50)

    # Create simple synthetic data
    np.random.seed(42)
    data = np.array([45.0, 45.1, 45.2, 45.0, 45.3, 45.1, 45.4, 45.2, 45.5])

    from statsmodels.tsa.holtwinters import Holt

    # Fit Holt model
    model = Holt(data).fit(smoothing_level=0.3, smoothing_trend=0.1, optimized=False)

    # Generate forecast
    forecast = model.forecast(5)

    print(f"Data: {data}")
    print(f"Final level: {model.level:.6f}")
    print(f"Final trend: {model.trend:.6f}")
    print(f"Forecast: {forecast}")

    # Check linearity
    diffs = np.diff(forecast)
    is_linear = np.allclose(diffs, diffs[0], rtol=1e-10)

    print(f"Forecast differences: {diffs}")
    print(f"Is linear: {is_linear}")
    print(f"Expected difference: {model.trend:.8f}")

    if is_linear:
        print("✅ Manual test passed - Holt should produce linear forecasts")
    else:
        print("❌ Manual test failed - Something is wrong with implementation")


if __name__ == "__main__":
    # Run quick manual test first
    quick_manual_test()

    # Then run full debug
    results = debug_single_day()

    if results:
        print("\n🎉 Debug completed successfully!")
        print("Check the outputs/debug_plots/ folder for visualization.")
    else:
        print("\n💥 Debug failed - check the error messages above.")
