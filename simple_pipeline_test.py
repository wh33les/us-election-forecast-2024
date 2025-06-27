# simple_pipeline_test.py
"""Simple test to capture pipeline output and find plotting issues."""

import subprocess
import sys
from pathlib import Path
import time


def run_pipeline_with_output_capture():
    """Run pipeline and capture all output."""

    print("🔍 RUNNING PIPELINE WITH FULL OUTPUT CAPTURE")
    print("=" * 60)

    # Check if 26Oct.png exists before running
    target_file = Path("outputs/forecast_images/26Oct.png")
    before_exists = target_file.exists()
    before_time = target_file.stat().st_mtime if before_exists else 0

    print(f"📁 Target file: {target_file}")
    print(f"📊 Before run - exists: {before_exists}")
    if before_exists:
        print(f"📊 Before run - time: {time.ctime(before_time)}")

    print(f"\n🚀 Running pipeline...")

    try:
        # Run pipeline with real-time output
        process = subprocess.Popen(
            [sys.executable, "main.py", "--date", "10-26", "--verbose", "--debug"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Capture output in real-time
        stdout_lines = []
        stderr_lines = []

        # Read stdout
        for line in process.stdout:
            stdout_lines.append(line.rstrip())
            print(f"📤 {line.rstrip()}")

        # Wait for process to complete
        process.wait()

        # Read stderr
        stderr_output = process.stderr.read()
        if stderr_output.strip():
            stderr_lines = stderr_output.strip().split("\n")
            print(f"\n❌ STDERR OUTPUT:")
            for line in stderr_lines:
                print(f"🔴 {line}")

        print(f"\n📊 Pipeline return code: {process.returncode}")

        # Check file status after running
        after_exists = target_file.exists()
        after_time = target_file.stat().st_mtime if after_exists else 0

        print(f"\n📁 After run - exists: {after_exists}")
        if after_exists:
            print(f"📊 After run - time: {time.ctime(after_time)}")
            print(f"📏 After run - size: {target_file.stat().st_size} bytes")

            if after_time > before_time:
                print("✅ FILE WAS UPDATED!")
                return True
            else:
                print("❌ File timestamp unchanged")
        else:
            print("❌ FILE MISSING AFTER RUN")

        # Look for specific plotting messages
        print(f"\n🎨 PLOTTING-RELATED MESSAGES:")
        plotting_messages = []
        for line in stdout_lines:
            if any(
                keyword in line.lower()
                for keyword in ["plot", "save", "visual", "forecast_images", "png"]
            ):
                plotting_messages.append(line)

        if plotting_messages:
            for msg in plotting_messages:
                print(f"   📊 {msg}")
        else:
            print("   ❌ No plotting messages found")

        # Look for error messages
        print(f"\n❌ ERROR MESSAGES:")
        error_messages = []
        for line in stdout_lines + stderr_lines:
            if any(
                keyword in line.lower()
                for keyword in ["error", "exception", "failed", "traceback"]
            ):
                error_messages.append(line)

        if error_messages:
            for msg in error_messages[:10]:  # Show first 10 errors
                print(f"   🔴 {msg}")
        else:
            print("   ✅ No error messages found")

        return False

    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        return False


def restore_original_file():
    """Restore the original 26Oct.png if it was deleted."""

    print(f"\n🔄 CHECKING IF FILE NEEDS RESTORATION")
    print("-" * 40)

    target_file = Path("outputs/forecast_images/26Oct.png")

    if not target_file.exists():
        print(f"❌ File missing: {target_file}")
        print(f"💡 You may need to restore from backup or regenerate")

        # Look for other similar files
        forecast_dir = Path("outputs/forecast_images")
        similar_files = [f for f in forecast_dir.glob("*.png") if "Oct" in f.name]

        if similar_files:
            print(f"📁 Other October files available:")
            for file in sorted(similar_files):
                print(f"   {file.name}")

        return False
    else:
        print(f"✅ File exists: {target_file}")
        return True


if __name__ == "__main__":
    success = run_pipeline_with_output_capture()

    if success:
        print(f"\n🎉 SUCCESS! File was updated.")
        print(f"📊 Check outputs/forecast_images/26Oct.png for straight forecast lines")
    else:
        print(f"\n❌ FAILED! File was not updated or was deleted.")

        # Try to restore
        if not restore_original_file():
            print(f"💡 Consider reverting your plotting changes and trying again")

    print(f"\n" + "=" * 60)
    print(f"🎯 TEST COMPLETE")
    print(f"=" * 60)
