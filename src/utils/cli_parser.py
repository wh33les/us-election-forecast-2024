# src/utils/cli_parser.py
"""Date parsing utilities for command line arguments."""

import argparse
import pandas as pd
from datetime import datetime, date
from typing import Optional, List

from src.config import DataConfig

# Create single config instance for the module
_config = DataConfig()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Rolling Election Forecast Pipeline 2024",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Process all dates (Oct 23 - Nov 5)
  python main.py --date 2024-10-25        # Process single date
  python main.py --date 10-25             # Process single date (flexible format)
  python main.py --start 10-23 --end 10-27 # Process date range
  python main.py --verbose --debug        # Show detailed output
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed progress and explanations",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show technical debugging information",
    )
    parser.add_argument(
        "--date",
        type=str,
        help=f"Process single date between {_config.forecast_start_date} and {_config.election_day} (flexible formats)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help=f"Start date for date range (default: {_config.forecast_start_date})",
    )
    parser.add_argument(
        "--end",
        type=str,
        help=f"End date for date range (default: {_config.election_day})",
    )

    return parser.parse_args()


def parse_flexible_date(date_string: str) -> date:
    """Parse flexible date formats, defaulting to 2024."""
    if not date_string or not date_string.strip():
        raise ValueError("Date string cannot be empty")

    # Use module-level config
    min_date = _config.min_valid_date_parsed
    max_date = _config.max_valid_date_parsed

    # Define all format patterns and how to normalize them
    format_configs = [
        # Formats that already include years - use as-is
        (date_string, "%Y-%m-%d"),  # 2024-10-25
        (date_string, "%m-%d-%Y"),  # 10-25-2024
        # Formats without years - add 2024
        (f"{date_string}-2024", "%m-%d-%Y"),  # 10-25 -> 10-25-2024
        (f"{date_string.replace('/', '-')}-2024", "%m-%d-%Y"),  # 10/25 -> 10-25-2024
        (f"{date_string} 2024", "%b %d %Y"),  # Oct 25 -> Oct 25 2024
        (f"{date_string} 2024", "%B %d %Y"),  # October 25 -> October 25 2024
    ]

    # Single loop to try all formats
    for normalized_string, date_format in format_configs:
        try:
            parsed_date = datetime.strptime(normalized_string, date_format).date()
            if min_date <= parsed_date <= max_date:
                return parsed_date
        except ValueError:
            continue

    # If nothing worked, raise error
    raise ValueError(
        f"Could not parse date '{date_string}'. Try formats like: 2024-10-25, 10-25, Oct 25. "
        f"Valid range: {min_date} to {max_date}"
    )


def determine_forecast_dates(args) -> List[date]:
    """Determine which dates to process based on command line arguments."""
    # Use module-level config
    default_start = _config.forecast_start_date_parsed
    default_end = _config.election_day_parsed

    if args.date:
        return [parse_flexible_date(args.date)]

    if args.start or args.end:
        start_date = parse_flexible_date(args.start) if args.start else default_start
        end_date = parse_flexible_date(args.end) if args.end else default_end

        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        return pd.date_range(start=start_date, end=end_date).date.tolist()

    return pd.date_range(start=default_start, end=default_end).date.tolist()
