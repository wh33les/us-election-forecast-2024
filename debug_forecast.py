#!/usr/bin/env python3
"""
Debug script to identify and fix election forecast directory and plotting issues.
"""

import os
from pathlib import Path
import pandas as pd


def debug_forecast_setup():
    """Debug the forecast setup and directory issues."""
    print("🔍 DEBUGGING ELECTION FORECAST SETUP")
    print("=" * 50)

    # 1. Check current working directory
    print(f"📁 Current working directory: {os.getcwd()}")

    # 2. Try to import and check your config
    try:
        from src.config import DataConfig

        data_config = DataConfig()
        forecast_dir = data_config.forecast_images_dir
        print(f"📂 Expected forecast directory: {forecast_dir}")

        # Check if directory exists
        forecast_path = Path(forecast_dir)
        if forecast_path.exists():
            print(f"✅ Directory exists: {forecast_path.absolute()}")
            # List contents
            contents = list(forecast_path.glob("*"))
            print(f"📋 Directory contents: {len(contents)} files")
            for file in contents[:10]:  # Show first 10 files
                print(f"   - {file.name}")
        else:
            print(f"❌ Directory does not exist: {forecast_path.absolute()}")

            # Try to create it
            try:
                forecast_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Successfully created directory: {forecast_path.absolute()}")
            except Exception as e:
                print(f"❌ Failed to create directory: {e}")

    except ImportError as e:
        print(f"❌ Cannot import DataConfig: {e}")
        print("   Try creating directories manually:")

        # Create standard directories
        standard_dirs = [
            "outputs/forecast_images",
            "outputs/previous_forecasts",
            "data",
        ]

        for dir_path in standard_dirs:
            path = Path(dir_path)
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created: {path.absolute()}")
            except Exception as e:
                print(f"❌ Failed to create {dir_path}: {e}")

    # 3. Check for existing data files
    print(f"\n📊 CHECKING DATA FILES")
    print("-" * 30)

    data_files = [
        "data/president_polls.csv",
        "data/election_forecast_2024_comprehensive.csv",
    ]

    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Missing: {file_path}")


def fix_plotting_error():
    """Provide fix for the plotting error."""
    print(f"\n🔧 FIXING PLOTTING ERROR")
    print("-" * 30)

    print("The error is in this function call (around line 1095 in main.py):")
    print(
        """
    plotter.plot_main_forecast(
        all_dates,
        test_dates,
        trump_train,
        harris_train,
        forecasts,
        baselines,
        fitted_values,
        best_params,
        days_till_then,
        holdout_dates=holdout_dates,  # ← THIS LINE CAUSES THE ERROR
        holdout_predictions=holdout_predictions,
        train_cutoff_date=train_cutoff_date,
        forecast_date=forecast_date,
        save_path=forecast_plot_path,
    )
    """
    )

    print("\n🔨 SOLUTION OPTIONS:")
    print("1. Remove the problematic parameters:")
    print(
        """
    plotter.plot_main_forecast(
        all_dates,
        test_dates,
        trump_train,
        harris_train,
        forecasts,
        baselines,
        fitted_values,
        best_params,
        days_till_then,
        forecast_date=forecast_date,
        save_path=forecast_plot_path,
    )
    """
    )

    print("2. OR update your ElectionPlotter class to accept these parameters.")


def quick_test_run():
    """Test if we can run a minimal version."""
    print(f"\n🧪 QUICK TEST")
    print("-" * 20)

    # Test basic imports
    try:
        from src.config import ModelConfig, DataConfig

        print("✅ Config imports working")

        from src.data.collectors import PollingDataCollector

        print("✅ Data collector import working")

        from src.data.processors import PollingDataProcessor

        print("✅ Data processor import working")

        from src.models.holt_forecaster import HoltElectionForecaster

        print("✅ Forecaster import working")

        from src.visualization.plotting import ElectionPlotter

        print("✅ Plotter import working")

        # Check if we can inspect the plot function
        plotter = ElectionPlotter(DataConfig())
        plot_method = getattr(plotter, "plot_main_forecast", None)
        if plot_method:
            import inspect

            sig = inspect.signature(plot_method)
            print(f"📋 plot_main_forecast parameters: {list(sig.parameters.keys())}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the project root directory")
        print("   and all src/ modules are properly structured")


def create_minimal_fix():
    """Create a minimal script to test directory creation and basic functionality."""
    print(f"\n📝 CREATING MINIMAL FIX SCRIPT")
    print("-" * 35)

    fix_script = '''#!/usr/bin/env python3
"""
Minimal fix script for election forecast project.
Run this to create directories and test basic functionality.
"""

from pathlib import Path
import pandas as pd
from datetime import date

def setup_directories():
    """Create all necessary directories."""
    directories = [
        "data",
        "outputs",
        "outputs/forecast_images", 
        "outputs/previous_forecasts",
        "src",
        "src/data",
        "src/models", 
        "src/visualization"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified: {dir_path}")

def test_basic_run():
    """Test a basic forecast run without plotting."""
    try:
        # Import your modules
        from src.config import ModelConfig, DataConfig
        from src.data.collectors import PollingDataCollector
        from src.data.processors import PollingDataProcessor
        
        print("✅ All imports successful")
        
        # Test data loading
        data_config = DataConfig()
        collector = PollingDataCollector(data_config)
        
        # Check if data file exists
        if Path("data/president_polls.csv").exists():
            raw_data = collector.load_raw_data()
            print(f"✅ Loaded {len(raw_data)} polling records")
        else:
            print("❌ Missing data/president_polls.csv")
            
    except Exception as e:
        print(f"❌ Error in basic test: {e}")

if __name__ == "__main__":
    setup_directories()
    test_basic_run()
'''

    # Save the fix script
    with open("fix_forecast.py", "w") as f:
        f.write(fix_script)

    print("✅ Created fix_forecast.py")
    print("   Run: python fix_forecast.py")


if __name__ == "__main__":
    debug_forecast_setup()
    fix_plotting_error()
    quick_test_run()
    create_minimal_fix()

    print(f"\n🎯 NEXT STEPS:")
    print("1. Run the created fix_forecast.py script")
    print("2. Fix the plotting error by removing holdout_dates parameter")
    print("3. Test with: python main.py --date 10-27 --verbose")
    print("4. Check that outputs/forecast_images/ directory gets created")
