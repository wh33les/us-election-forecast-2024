# enhanced_cleanup_polling_data.py
"""Enhanced pre-filtering: Remove unused candidates, dates AND columns."""

import pandas as pd
from datetime import datetime
import shutil
from pathlib import Path
from src.config import DataConfig


config = DataConfig()


def enhanced_prefilter_polling_data():
    """Pre-filter president_polls.csv for maximum efficiency."""

    # File paths
    original_file = "data/raw_polling_data.csv"
    backup_file = "data/raw_polling_data_backup.csv"

    print("🚀 Enhanced pre-filtering of polling data...")
    print("=" * 50)

    # 1. Create backup
    print("📋 Creating backup...")
    shutil.copy2(original_file, backup_file)
    print(f"✅ Backup saved as: {backup_file}")

    # 2. Analyze original file
    print("📁 Analyzing original data...")
    original_size_mb = Path(original_file).stat().st_size / (1024 * 1024)

    # Read just first row to get column info
    sample = pd.read_csv(original_file, nrows=1)
    total_columns = len(sample.columns)
    print(f"📊 Original: {total_columns} columns, {original_size_mb:.1f} MB")

    # 3. Define columns we actually need
    essential_columns = [
        "pollscore",  # Used for poll quality filtering
        "state",  # Used for geographic filtering
        "candidate_name",  # Used for candidate filtering
        "end_date",  # Used for date filtering
        "pct",  # The actual poll percentage (main data)
        "population",  # Used for population filtering (lv, rv, a)
    ]

    print(f"🎯 Keeping {len(essential_columns)}/{total_columns} columns:")
    for col in essential_columns:
        print(f"   • {col}")

    # 4. Read data with only essential columns (first optimization!)
    print(f"\n📥 Loading data with column selection...")
    df = pd.read_csv(original_file, usecols=essential_columns)
    print(f"   Column reduction: {total_columns} → {len(essential_columns)} columns")

    original_rows = len(df)
    print(f"   Loaded: {original_rows:,} rows")

    # 5. Parse dates
    df["end_date"] = pd.to_datetime(df["end_date"], format="mixed").dt.date
    biden_dropout = config.earliest_available_data_parsed

    # 6. Apply row filters
    print(f"\n🎯 Applying row filters...")

    # Filter by date (Biden dropout and after)
    df_filtered = df[df["end_date"] >= biden_dropout].copy()
    print(f"   After date filter: {len(df_filtered):,} records")

    # Filter by candidates (Trump and Harris only)
    relevant_candidates = ["Donald Trump", "Kamala Harris"]
    df_filtered = df_filtered[df_filtered["candidate_name"].isin(relevant_candidates)]
    print(f"   After candidate filter: {len(df_filtered):,} records")

    # Keep all population types and geographies (let pipeline filter these)
    print(f"   Keeping all population types for pipeline filtering")
    print(f"   Keeping all geographies for pipeline filtering")

    # 7. Calculate total reduction
    final_rows = len(df_filtered)
    row_reduction = (1 - final_rows / original_rows) * 100
    col_reduction = (1 - len(essential_columns) / total_columns) * 100

    print(f"\n📉 Optimization summary:")
    print(
        f"   Row reduction: {original_rows:,} -> {final_rows:,} ({row_reduction:.1f}% reduction)"
    )
    print(
        f"   Column reduction: {total_columns} -> {len(essential_columns)} ({col_reduction:.1f}% reduction)"
    )

    # 8. Save optimized file
    print(f"\n💾 Saving optimized file...")
    df_filtered.to_csv(original_file, index=False)

    new_size_mb = Path(original_file).stat().st_size / (1024 * 1024)
    total_reduction = (1 - new_size_mb / original_size_mb) * 100

    print(
        f"✅ File size: {original_size_mb:.1f} MB → {new_size_mb:.1f} MB ({total_reduction:.1f}% reduction)"
    )

    # 9. Show data quality preserved for pipeline
    print(f"\n🔧 Data preserved for pipeline filtering:")
    if len(df_filtered) > 0:
        population_types = df_filtered["population"].value_counts()
        print(f"   Population types: {list(population_types.index)}")

        state_counts = df_filtered["state"].value_counts()
        national_polls = df_filtered["state"].isnull().sum()
        print(
            f"   Geographic coverage: {len(state_counts)} states + {national_polls} national polls"
        )

        pollscore_range = f"{df_filtered['pollscore'].min():.1f} to {df_filtered['pollscore'].max():.1f}"
        print(f"   Pollscore range: {pollscore_range}")

        date_range = (
            f"{df_filtered['end_date'].min()} to {df_filtered['end_date'].max()}"
        )
        print(f"   Date range: {date_range}")

    # 10. Show sample of optimized data
    print(f"\n📋 Sample of optimized data:")
    if len(df_filtered) > 0:
        print(df_filtered.head(3).to_string())

    # 11. Create summary file
    summary_content = f"""Enhanced Pre-filtering Results
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OPTIMIZATIONS APPLIED:
• Column selection: {total_columns} -> {len(essential_columns)} columns ({col_reduction:.1f}% reduction)
• Date filtering: Only post-Biden dropout ({biden_dropout})
• Candidate filtering: Only Trump and Harris
• Row reduction: {original_rows:,} -> {final_rows:,} rows ({row_reduction:.1f}% reduction)

TOTAL FILE SIZE REDUCTION: {total_reduction:.1f}%
Original: {original_size_mb:.1f} MB -> Optimized: {new_size_mb:.1f} MB

PRESERVED FOR PIPELINE FILTERING:
• All population types: {list(population_types.index) if len(df_filtered) > 0 else 'N/A'}
• All pollscore values: {pollscore_range if len(df_filtered) > 0 else 'N/A'}
• All geographies: States + national polls

COLUMNS REMOVED:
{chr(10).join([f'• {col}' for col in sample.columns if col not in essential_columns])}

BACKUP LOCATION: {backup_file}
"""

    with open("data/PREFILTER_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(summary_content)

    print(f"\n🎉 Enhanced pre-filtering complete!")
    print(
        f"📊 Total reduction: {total_reduction:.1f}% ({original_size_mb:.1f} MB -> {new_size_mb:.1f} MB)"
    )
    print(f"🔧 Your pipeline will work exactly the same, just much faster!")
    print(f"📁 Backup: {backup_file}")
    print(f"📄 Summary: data/PREFILTER_SUMMARY.txt")


if __name__ == "__main__":
    enhanced_prefilter_polling_data()
