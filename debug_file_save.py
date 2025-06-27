# debug_file_save.py
"""Debug why the plot file isn't being updated."""

import os
from pathlib import Path
from datetime import datetime


def debug_file_save():
    """Check file save paths and permissions."""

    print("🔍 DEBUGGING FILE SAVE ISSUE")
    print("=" * 50)

    # Check current working directory
    print(f"📁 Current directory: {os.getcwd()}")

    # Check expected output paths
    paths_to_check = [
        "outputs/forecast_images/23Oct.png",
        "outputs\\forecast_images\\23Oct.png",  # Windows path
        "outputs/forecast_images/",
        "outputs\\forecast_images\\",
    ]

    for path_str in paths_to_check:
        path = Path(path_str)
        print(f"\n📂 Checking: {path}")
        print(f"   Exists: {path.exists()}")
        if path.exists():
            if path.is_file():
                stat = path.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime)
                print(f"   Modified: {mtime}")
                print(f"   Size: {stat.st_size} bytes")
            elif path.is_dir():
                print(f"   Contents: {list(path.glob('*'))}")

    # Check if we can write to the directory
    output_dir = Path("outputs/forecast_images")
    print(f"\n🔧 Testing write permissions...")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test_write.tmp"

        with open(test_file, "w") as f:
            f.write(f"Test write at {datetime.now()}")

        if test_file.exists():
            print("✅ Can write to directory")
            test_file.unlink()  # Clean up
        else:
            print("❌ File creation failed")

    except Exception as e:
        print(f"❌ Cannot write to directory: {e}")

    # Look for any 23Oct files anywhere
    print(f"\n🔍 Searching for 23Oct files...")
    try:
        for root, dirs, files in os.walk("."):
            for file in files:
                if "23Oct" in file:
                    full_path = Path(root) / file
                    stat = full_path.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    print(f"   Found: {full_path} (modified: {mtime})")
    except Exception as e:
        print(f"Search failed: {e}")


def create_test_plot():
    """Create a simple test plot to verify saving works."""

    print(f"\n🎨 CREATING TEST PLOT")
    print("=" * 30)

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create simple test plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, "b-", label="Test Line")
        plt.title(f'Test Plot - {datetime.now().strftime("%H:%M:%S")}')
        plt.legend()
        plt.grid(True)

        # Save to multiple locations
        test_paths = [
            "test_plot_1.png",
            "outputs/test_plot_2.png",
            "outputs/forecast_images/test_plot_3.png",
        ]

        for path_str in test_paths:
            path = Path(path_str)
            path.parent.mkdir(parents=True, exist_ok=True)

            try:
                plt.savefig(path, dpi=150, bbox_inches="tight")
                if path.exists():
                    print(f"✅ Successfully saved: {path}")
                else:
                    print(f"❌ Failed to save: {path}")
            except Exception as e:
                print(f"❌ Error saving {path}: {e}")

        plt.close()

    except Exception as e:
        print(f"❌ Error creating test plot: {e}")


if __name__ == "__main__":
    debug_file_save()
    create_test_plot()

    print(f"\n💡 NEXT STEPS:")
    print("1. Check if test plots were created successfully")
    print("2. If test plots work, the issue is in your plotting code")
    print("3. If test plots fail, there's a system-level permission issue")
    print("4. Try running with admin privileges if needed")
