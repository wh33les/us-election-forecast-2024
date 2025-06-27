# find_plot.py
"""Find the newly created plot file."""

import os
from pathlib import Path
from datetime import datetime


def find_new_plot():
    """Find the 25Oct plot file."""

    print("📁 SEARCHING FOR NEW PLOT FILE")
    print("=" * 40)

    # Check expected locations
    possible_paths = [
        "outputs/forecast_images/25Oct.png",
        "outputs\\forecast_images\\25Oct.png",
        "outputs/forecast_images/",
        "outputs\\forecast_images\\",
    ]

    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                stat = path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"✅ Found: {path}")
                print(f"   Modified: {mtime}")
                print(f"   Size: {stat.st_size} bytes")
            elif path.is_dir():
                print(f"📂 Directory: {path}")
                files = list(path.glob("*.png"))
                for file in sorted(files):
                    stat = file.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    print(f"   {file.name} - {mtime} ({stat.st_size} bytes)")
        else:
            print(f"❌ Not found: {path}")

    # Search for any files with "25Oct" in the name
    print(f"\n🔍 SEARCHING FOR ANY 25Oct FILES")
    print("-" * 30)

    found_any = False
    try:
        for root, dirs, files in os.walk("."):
            for file in files:
                if "25Oct" in file:
                    full_path = Path(root) / file
                    stat = full_path.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    print(f"✅ Found: {full_path}")
                    print(f"   Modified: {mtime}")
                    print(f"   Size: {stat.st_size} bytes")
                    found_any = True
    except Exception as e:
        print(f"Search error: {e}")

    if not found_any:
        print("❌ No 25Oct files found anywhere")

    # Check recent files in forecast_images
    print(f"\n⏰ RECENT FILES IN FORECAST_IMAGES")
    print("-" * 40)

    forecast_dir = Path("outputs/forecast_images")
    if forecast_dir.exists():
        files = list(forecast_dir.glob("*.png"))
        # Sort by modification time, most recent first
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        print("Most recent files:")
        for i, file in enumerate(files[:5]):  # Show top 5 most recent
            stat = file.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"  {i+1}. {file.name} - {mtime}")
    else:
        print("❌ forecast_images directory not found")


def check_file_timestamps():
    """Check if any files were created in the last few minutes."""

    print(f"\n🕐 FILES CREATED IN LAST 5 MINUTES")
    print("-" * 40)

    import time

    current_time = time.time()
    five_minutes_ago = current_time - (5 * 60)  # 5 minutes in seconds

    forecast_dir = Path("outputs/forecast_images")
    if forecast_dir.exists():
        recent_files = []
        for file in forecast_dir.glob("*.png"):
            if file.stat().st_mtime > five_minutes_ago:
                recent_files.append(file)

        if recent_files:
            print("Recent files (last 5 minutes):")
            for file in recent_files:
                stat = file.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"  ✅ {file.name} - {mtime}")
        else:
            print("❌ No files created in last 5 minutes")


if __name__ == "__main__":
    find_new_plot()
    check_file_timestamps()

    print(f"\n💡 NEXT STEPS:")
    print("1. Look at the most recent file in the list above")
    print("2. That should be your 25Oct.png file")
    print("3. Open it and check if the forecast lines are straight")
    print("4. Compare with 24Oct.png to see the difference")
