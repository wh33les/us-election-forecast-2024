# plotting_diagnostic.py
"""Diagnose what's happening in the plotting process."""

import sys
import subprocess
from pathlib import Path


def run_with_detailed_output():
    """Run the pipeline and capture detailed output."""

    print("🔍 RUNNING PIPELINE WITH DETAILED DIAGNOSTICS")
    print("=" * 60)

    try:
        # Run with maximum verbosity
        result = subprocess.run(
            [sys.executable, "main.py", "--date", "10-26", "--verbose", "--debug"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        print(f"📊 Return code: {result.returncode}")
        print(f"📏 STDOUT length: {len(result.stdout)} chars")
        print(f"📏 STDERR length: {len(result.stderr)} chars")

        # Look for specific plotting-related messages
        stdout_lines = result.stdout.split("\n")
        stderr_lines = result.stderr.split("\n")

        print(f"\n🎨 PLOTTING-RELATED MESSAGES:")
        print("-" * 40)

        plotting_keywords = [
            "plot",
            "visualization",
            "save",
            "forecast_images",
            "matplotlib",
            "plt",
            "figure",
            "error",
            "fail",
            "exception",
        ]

        print("📤 STDOUT plotting messages:")
        for line in stdout_lines:
            if any(keyword in line.lower() for keyword in plotting_keywords):
                print(f"   {line}")

        print("\n📤 STDERR plotting messages:")
        for line in stderr_lines:
            if any(keyword in line.lower() for keyword in plotting_keywords):
                print(f"   {line}")

        # Look for specific success/failure indicators
        print(f"\n✅ SUCCESS INDICATORS:")
        success_indicators = [
            "Saved main forecast plot",
            "Creating main forecast plot",
            "Plot created",
            "Pipeline completed successfully",
        ]

        for indicator in success_indicators:
            if indicator in result.stdout:
                print(f"   ✅ Found: {indicator}")
            else:
                print(f"   ❌ Missing: {indicator}")

        # Look for error indicators
        print(f"\n❌ ERROR INDICATORS:")
        error_indicators = [
            "Error",
            "Exception",
            "Failed",
            "Traceback",
            "matplotlib",
            "plt",
            "figure",
        ]

        found_errors = []
        for line in stdout_lines + stderr_lines:
            if any(error in line for error in error_indicators):
                found_errors.append(line)

        if found_errors:
            print("   Found errors:")
            for error in found_errors[:10]:  # Show first 10 errors
                print(f"     {error}")
        else:
            print("   ✅ No obvious errors found")

        # Show last 20 lines of output
        print(f"\n📋 LAST 20 LINES OF OUTPUT:")
        print("-" * 30)
        for line in stdout_lines[-20:]:
            if line.strip():
                print(f"   {line}")

        return result

    except subprocess.TimeoutExpired:
        print("⏱️ Pipeline timed out")
        return None
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return None


def check_plotting_imports():
    """Test if the plotting code can be imported without errors."""

    print(f"\n🔍 TESTING PLOTTING CODE IMPORTS")
    print("-" * 40)

    try:
        # Test basic import
        from src.visualization.plotting import ElectionPlotter

        print("✅ ElectionPlotter imported successfully")

        # Test instantiation
        from src.config.data_config import DataConfig

        config = DataConfig()
        plotter = ElectionPlotter(config)
        print("✅ ElectionPlotter instantiated successfully")

        # Test if the method exists
        if hasattr(plotter, "_plot_future_forecasts_connected"):
            print("✅ _plot_future_forecasts_connected method exists")
        else:
            print("❌ _plot_future_forecasts_connected method missing")

        # Test matplotlib
        import matplotlib.pyplot as plt

        print("✅ matplotlib imported successfully")

        # Test if we can create a figure
        fig = plt.figure()
        plt.close(fig)
        print("✅ matplotlib figure creation works")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_output_directory():
    """Check the output directory status."""

    print(f"\n📁 OUTPUT DIRECTORY STATUS")
    print("-" * 30)

    forecast_dir = Path("outputs/forecast_images")

    if forecast_dir.exists():
        print(f"✅ Directory exists: {forecast_dir}")

        # Check permissions
        test_file = forecast_dir / "test_write.tmp"
        try:
            with open(test_file, "w") as f:
                f.write("test")
            test_file.unlink()
            print("✅ Directory is writable")
        except Exception as e:
            print(f"❌ Directory not writable: {e}")

        # List recent files
        files = list(forecast_dir.glob("*.png"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        print(f"📊 Recent files:")
        for file in files[:5]:
            from datetime import datetime

            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            print(f"   {file.name} - {mtime}")

    else:
        print(f"❌ Directory missing: {forecast_dir}")


if __name__ == "__main__":
    print("🚀 COMPREHENSIVE PLOTTING DIAGNOSTIC")
    print("=" * 60)

    # Test 1: Check imports
    imports_ok = check_plotting_imports()

    # Test 2: Check output directory
    check_output_directory()

    # Test 3: Run pipeline with detailed logging
    if imports_ok:
        run_with_detailed_output()
    else:
        print("❌ Skipping pipeline run due to import errors")

    print(f"\n" + "=" * 60)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("=" * 60)
