# test_existing_date.py
"""Test the plotting fix with a date that definitely works."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def test_with_working_date():
    """Test plotting fix by overwriting an existing file."""

    print("🔧 TESTING PLOTTING FIX WITH KNOWN WORKING DATE")
    print("=" * 60)

    # Use Oct 26 since we know it exists and works
    test_date = "10-26"
    expected_file = Path("outputs/forecast_images/26Oct.png")

    print(f"📅 Testing with date: {test_date}")
    print(f"📁 Expected file: {expected_file}")

    # Check original file timestamp
    if expected_file.exists():
        original_time = expected_file.stat().st_mtime
        original_datetime = datetime.fromtimestamp(original_time)
        print(f"📊 Original file time: {original_datetime}")
        print(f"📏 Original file size: {expected_file.stat().st_size} bytes")
    else:
        print(f"❌ Original file doesn't exist: {expected_file}")
        return

    print(f"\n🚀 Running pipeline with fixed plotting code...")

    try:
        # Run pipeline to regenerate 26Oct.png with fixed plotting
        result = subprocess.run(
            [sys.executable, "main.py", "--date", test_date, "--verbose", "--debug"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")

            # Check if file was updated
            if expected_file.exists():
                new_time = expected_file.stat().st_mtime
                new_datetime = datetime.fromtimestamp(new_time)
                new_size = expected_file.stat().st_size

                print(f"📊 New file time: {new_datetime}")
                print(f"📏 New file size: {new_size} bytes")

                if new_time > original_time:
                    print("✅ File was updated with new plotting code!")
                    print(f"🎯 CHECK: {expected_file}")
                    print("🔍 Look for STRAIGHT dashed forecast lines")

                    # Show the diagnostic output from the logs
                    if "forecast linear: True" in result.stdout:
                        print("✅ Diagnostics confirm forecasts are linear")

                    return str(expected_file)
                else:
                    print(
                        "❌ File timestamp didn't change - may not have been regenerated"
                    )
            else:
                print("❌ File missing after pipeline run")

        else:
            print("❌ Pipeline failed:")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])

    except subprocess.TimeoutExpired:
        print("⏱️ Pipeline timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

    return None


def compare_before_after():
    """Compare the visual results."""

    print(f"\n📊 VISUAL COMPARISON TEST")
    print("=" * 30)

    # Files to compare
    old_method_file = Path("outputs/forecast_images/24Oct.png")  # Generated before fix
    new_method_file = Path("outputs/forecast_images/26Oct.png")  # Generated after fix

    print(f"📈 OLD METHOD (should have curved lines): {old_method_file}")
    if old_method_file.exists():
        print(
            f"   ✅ Found - {datetime.fromtimestamp(old_method_file.stat().st_mtime)}"
        )
    else:
        print(f"   ❌ Missing")

    print(f"📈 NEW METHOD (should have straight lines): {new_method_file}")
    if new_method_file.exists():
        print(
            f"   ✅ Found - {datetime.fromtimestamp(new_method_file.stat().st_mtime)}"
        )
    else:
        print(f"   ❌ Missing")

    if old_method_file.exists() and new_method_file.exists():
        print(f"\n🎯 VISUAL TEST:")
        print(f"1. Open both files side by side")
        print(f"2. Look at the RED and BLUE dashed lines (forecasts)")
        print(f"3. In 24Oct.png they should be CURVED")
        print(f"4. In 26Oct.png they should be STRAIGHT")
        print(f"5. If 26Oct.png has straight lines, THE FIX WORKED! 🎉")

        return True
    else:
        print(f"❌ Cannot compare - missing files")
        return False


if __name__ == "__main__":
    updated_file = test_with_working_date()

    if updated_file:
        print(f"\n" + "=" * 60)
        print(f"🎉 SUCCESS! Updated file: {updated_file}")
        print(f"=" * 60)

        if compare_before_after():
            print(f"\n✅ Ready for visual comparison!")
        else:
            print(f"\n❌ Comparison files missing")
    else:
        print(f"\n❌ Test failed - file not updated")

    print(f"\n⏸️ Press Enter to continue...")
    input()
