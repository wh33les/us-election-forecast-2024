# fix_locked_file.py
"""Utility to fix the locked CSV file issue."""

import os
import time
import shutil
from pathlib import Path


def fix_locked_csv():
    """Fix the locked CSV file issue."""

    csv_path = Path("data/election_forecast_2024_comprehensive.csv")

    print("🔧 FIXING LOCKED CSV FILE")
    print("=" * 40)

    if not csv_path.exists():
        print("✅ CSV file doesn't exist - no issue to fix")
        return True

    print(f"📁 Found CSV file: {csv_path}")
    print(f"📊 File size: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Try to open the file to check if it's locked
    try:
        with open(csv_path, "r+") as f:
            pass
        print("✅ File is not locked - should work fine")
        return True
    except (IOError, OSError) as e:
        print(f"❌ File is locked: {e}")
        print("\n🔄 Attempting to fix...")

        # Strategy 1: Wait a bit and try again
        print("Waiting 5 seconds for file to be released...")
        time.sleep(5)

        try:
            with open(csv_path, "r+") as f:
                pass
            print("✅ File is now unlocked!")
            return True
        except (IOError, OSError):
            print("❌ Still locked after waiting")

        # Strategy 2: Create backup and delete original
        backup_path = csv_path.with_suffix(".backup.csv")
        try:
            print(f"📋 Creating backup: {backup_path}")
            shutil.copy2(csv_path, backup_path)

            print(f"🗑️  Deleting locked file: {csv_path}")
            csv_path.unlink()

            print("✅ Successfully removed locked file")
            print(f"📂 Backup saved as: {backup_path}")
            print("\n💡 Your pipeline should now run without permission errors")
            return True

        except Exception as e:
            print(f"❌ Could not fix file: {e}")
            print("\n🔧 MANUAL FIX REQUIRED:")
            print("1. Close Excel if you have the CSV file open")
            print("2. Close any other programs that might be using the file")
            print("3. If needed, restart your computer")
            print("4. Delete the file manually if it still exists")
            return False


def check_file_permissions():
    """Check if we can write to the data directory."""

    print("\n🔍 CHECKING FILE PERMISSIONS")
    print("=" * 40)

    data_dir = Path("data")

    # Check if data directory exists and is writable
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Created data directory")
        except Exception as e:
            print(f"❌ Cannot create data directory: {e}")
            return False

    # Test write permissions
    test_file = data_dir / "test_write.tmp"
    try:
        with open(test_file, "w") as f:
            f.write("test")
        test_file.unlink()
        print("✅ Write permissions OK")
        return True
    except Exception as e:
        print(f"❌ Cannot write to data directory: {e}")
        return False


if __name__ == "__main__":
    print("🚀 ELECTION FORECAST FILE FIX UTILITY")
    print("=" * 50)

    # Check permissions first
    if not check_file_permissions():
        print("\n💥 Cannot write to data directory - check permissions")
        exit(1)

    # Fix the locked file
    if fix_locked_csv():
        print("\n🎉 File issue resolved!")
        print("You can now run your pipeline with:")
        print("python main.py --verbose --debug")
    else:
        print("\n💥 Could not automatically fix the issue")
        print("Please follow the manual fix instructions above")
