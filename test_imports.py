# Create a test file called test_imports.py in your project root
try:
    from src.data.collectors import PollingDataCollector

    print("✅ PollingDataCollector import successful")
except Exception as e:
    print(f"❌ PollingDataCollector import failed: {e}")

try:
    from src.data.processors import PollingDataProcessor

    print("✅ PollingDataProcessor import successful")
except Exception as e:
    print(f"❌ PollingDataProcessor import failed: {e}")

try:
    from src.models.holt_forecaster import HoltElectionForecaster

    print("✅ HoltElectionForecaster import successful")
except Exception as e:
    print(f"❌ HoltElectionForecaster import failed: {e}")

try:
    from src.pipeline.runner import ForecastRunner

    print("✅ ForecastRunner import successful")
except Exception as e:
    print(f"❌ ForecastRunner import failed: {e}")
