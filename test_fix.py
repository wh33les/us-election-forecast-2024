# test_fix.py
"""Test if the plotting fix works."""

print("🔧 TESTING PLOTTING FIX")
print("=" * 30)

# After you replace the plotting method, run this:
import subprocess
import sys


def test_plotting_fix():
    """Test the plotting fix by generating a new forecast."""

    print("1. Testing single day forecast with fixed plotting...")

    try:
        # Run your pipeline with the fixed plotting code
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--date",
                "10-25",  # Use a different date for fresh output
                "--verbose",
                "--debug",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            print("📊 Check outputs/forecast_images/25Oct.png")
            print("🔍 Look for straight dashed forecast lines")
        else:
            print("❌ Pipeline failed:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("⏱️ Pipeline timed out (taking too long)")
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")


def compare_plots():
    """Compare old vs new plots."""

    print("\n📊 COMPARING PLOTS")
    print("-" * 20)

    from pathlib import Path

    old_plot = Path("outputs/forecast_images/24Oct.png")  # Curved lines
    new_plot = Path("outputs/forecast_images/25Oct.png")  # Should be straight

    if old_plot.exists():
        print(f"✅ Old plot (curved): {old_plot}")
    else:
        print(f"❌ Old plot not found: {old_plot}")

    if new_plot.exists():
        print(f"✅ New plot (should be straight): {new_plot}")
    else:
        print(f"❌ New plot not found: {new_plot}")

    print("\n💡 VISUAL COMPARISON:")
    print("- Open both plots side by side")
    print("- 24Oct.png should have curved dashed lines")
    print("- 25Oct.png should have straight dashed lines")
    print("- If both are straight, the fix worked!")


if __name__ == "__main__":
    print("BEFORE running this script:")
    print("1. Replace _plot_future_forecasts_connected() method in plotting.py")
    print("2. Save the file")
    print("3. Then run this test")
    print()

    response = input("Have you updated the plotting.py file? (y/n): ")

    if response.lower() == "y":
        test_plotting_fix()
        compare_plots()
    else:
        print("📝 Please update plotting.py first, then run this test")
