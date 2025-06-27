# src/utils/date_parser.py
"""Date parsing utilities for command line arguments."""

import argparse
import pandas as pd
from datetime import datetime, date


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
        help="Process single date (YYYY-MM-DD, MM-DD, or flexible formats)",
    )
    parser.add_argument(
        "--start", type=str, help="Start date for date range (YYYY-MM-DD or MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, help="End date for date range (YYYY-MM-DD or MM-DD)"
    )

    return parser.parse_args()


def parse_flexible_date(date_string):
    """Parse flexible date formats, defaulting to 2024."""
    if not date_string:
        return None

    formats_to_try = [
        "%Y-%m-%d",  # 2024-10-25
        "%m-%d-%Y",  # 10-25-2024
    ]

    year_agnostic_formats = [
        "%m-%d",  # 10-25 (will add 2024)
        "%m/%d",  # 10/25 (will add 2024)
        "%b %d",  # Oct 25 (will add 2024)
        "%B %d",  # October 25 (will add 2024)
    ]

    # Try full date formats first
    for date_format in formats_to_try:
        try:
            parsed_date = datetime.strptime(date_string, date_format).date()
            if date(2024, 10, 1) <= parsed_date <= date(2024, 11, 30):
                return parsed_date
        except ValueError:
            continue

    # Try year-agnostic formats and manually add 2024
    for date_format in year_agnostic_formats:
        try:
            if date_format == "%m-%d":
                temp_date_str = f"2023-{date_string}"
                parsed_date = datetime.strptime(temp_date_str, "2023-%m-%d").date()
                parsed_date = parsed_date.replace(year=2024)
            elif date_format == "%m/%d":
                temp_date_str = f"2023-{date_string.replace('/', '-')}"
                parsed_date = datetime.strptime(temp_date_str, "2023-%m-%d").date()
                parsed_date = parsed_date.replace(year=2024)
            else:
                parsed_date = datetime.strptime(
                    f"{date_string} 2024", f"{date_format} %Y"
                ).date()

            if date(2024, 10, 1) <= parsed_date <= date(2024, 11, 30):
                return parsed_date

        except ValueError:
            continue

    raise ValueError(
        f"Could not parse date '{date_string}'. Try formats like: 2024-10-25, 10-25, Oct 25"
    )


def determine_forecast_dates(args):
    """Determine which dates to process based on command line arguments."""
    default_start = date(2024, 10, 23)
    default_end = date(2024, 11, 5)

    if args.date:
        return [parse_flexible_date(args.date)]

    if args.start or args.end:
        start_date = parse_flexible_date(args.start) if args.start else default_start
        end_date = parse_flexible_date(args.end) if args.end else default_end

        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        return pd.date_range(start=start_date, end=end_date).date

    return pd.date_range(start=default_start, end=default_end).date
