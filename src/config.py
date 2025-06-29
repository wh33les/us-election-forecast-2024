# src/config.py
"""Configuration settings for the election forecasting pipeline."""

from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""

    # Cross-validation settings
    n_splits: int = 5
    test_size: int = 7

    # Grid search parameters for Holt smoothing
    grid_min: float = 0.0
    grid_max: float = 0.5
    grid_step: float = 0.1

    # Electoral college settings
    swing_state_electoral_votes: int = 93
    trump_safe_electoral_votes: int = 219
    harris_safe_electoral_votes: int = 226

    @property
    def grid_numbers(self):
        """Generate grid search parameter range."""
        return np.arange(self.grid_min, self.grid_max, self.grid_step)


@dataclass
class DataConfig:
    """Configuration for data sources and filtering."""

    # Data file paths
    raw_data_path: str = "data/raw_polling_data.csv"
    polling_cache_path: str = "data/polling_averages_cache.csv"
    forecast_history_path: str = "data/forecast_history.csv"

    # Output directories and files
    forecast_images_dir: str = "outputs/forecast_images"
    historical_plots_dir: str = "outputs/previous_forecasts"

    # Data filtering criteria
    candidates: List[str] = field(
        default_factory=lambda: ["Donald Trump", "Kamala Harris"]
    )
    population_filter: str = "lv"  # likely voters
    pollscore_threshold: float = 0.0  # negative pollscore only

    # Core date strings
    earliest_available_data: str = "2024-07-21"
    forecast_start_date: str = "2024-10-23"
    election_day: str = "2024-11-05"

    # Explicitly declare parsed date attributes for type checking
    # These are set in __post_init__ and should never be None
    earliest_available_data_parsed: date = field(init=False)
    forecast_start_date_parsed: date = field(init=False)
    election_day_parsed: date = field(init=False)
    min_valid_date_parsed: date = field(init=False)
    max_valid_date_parsed: date = field(init=False)

    # Single source of truth for swing states
    swing_states: Dict[str, Dict] = field(
        default_factory=lambda: {
            "Arizona": {"code": "AZ", "electoral_votes": 11},
            "Georgia": {"code": "GA", "electoral_votes": 16},
            "Michigan": {"code": "MI", "electoral_votes": 15},
            "Nevada": {"code": "NV", "electoral_votes": 6},
            "North Carolina": {"code": "NC", "electoral_votes": 16},
            "Pennsylvania": {"code": "PA", "electoral_votes": 19},
            "Wisconsin": {"code": "WI", "electoral_votes": 10},
        }
    )

    def __post_init__(self):
        """Initialize derived properties and parsed dates."""
        # Parse all dates and assign to typed attributes
        self.earliest_available_data_parsed = datetime.strptime(
            self.earliest_available_data, "%Y-%m-%d"
        ).date()
        self.forecast_start_date_parsed = datetime.strptime(
            self.forecast_start_date, "%Y-%m-%d"
        ).date()
        self.election_day_parsed = datetime.strptime(
            self.election_day, "%Y-%m-%d"
        ).date()

        # Set validation date range (same as core dates)
        self.min_valid_date_parsed = self.earliest_available_data_parsed
        self.max_valid_date_parsed = self.election_day_parsed

        # Legacy string attributes for backward compatibility
        self.min_valid_date = self.earliest_available_data
        self.max_valid_date = self.election_day

    def get_swing_state_names(self) -> List[str]:
        """Get list of swing state names."""
        return list(self.swing_states.keys())

    def get_swing_state_codes(self) -> Dict[str, int]:
        """Get mapping of state codes to electoral votes."""
        return {
            state_info["code"]: state_info["electoral_votes"]
            for state_info in self.swing_states.values()
        }
