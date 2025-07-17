# src/config.py
"""Configuration settings for the election forecasting pipeline."""

from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class PlotConfig:
    """Configuration for all plotting and visualization settings."""

    # Figure settings
    figure_size: Tuple[int, int] = (12, 6)
    dpi: int = 150

    # Plot limits and ranges
    date_range_start: str = "2024-07-21"
    date_range_end: str = "2024-11-05"
    y_axis_min: float = 42.0
    y_axis_max: float = 52.0

    # Historical plot settings
    historical_date_range_start: str = "2024-10-23"
    historical_date_range_end: str = "2024-11-05"

    # Font settings
    title_font_size: int = 16
    axis_label_font_size: int = 12
    annotation_font_size: int = 11

    # Line and marker settings
    polling_line_width: float = 2.0
    prediction_line_width: float = 2.0
    baseline_line_width: float = 1.5

    # Alpha (transparency) settings
    prediction_alpha: float = 0.7
    baseline_alpha: float = 0.7
    grid_alpha: float = 0.3

    # Marker settings
    model_marker: str = "o"
    baseline_marker: str = "s"
    model_marker_size: int = 4
    baseline_marker_size: int = 4

    # Line styles
    polling_line_style: str = "-"
    model_line_style: str = "--"
    baseline_line_style: str = ":"
    historical_model_line_style: str = "-"
    historical_baseline_line_style: str = "--"

    # Annotation positions (as tuples of (x, y) coordinates)
    hyperparameter_annotation_pos: Tuple[str, float] = ("2024-07-25", 42.3)
    mase_annotation_pos: Tuple[str, float] = ("2024-08-17", 42.73)

    # Legend settings
    legend_location: str = "lower right"

    # Tick formatting
    date_format_style: str = (
        "short_month"  # Options: "numeric" (10-7), "short_month" (Oct 7), "full_month" (October 7)
    )
    rotate_tick_labels: bool = False  # Rotate labels for better readability
    tick_label_rotation: int = 45  # Rotation angle
    tick_label_ha: str = "right"  # Horizontal alignment

    # Forecast plot tick intervals (longer time range, fewer ticks)
    forecast_major_tick_interval: int = 14  # Show major ticks every 14 days

    # Historical plot tick intervals (shorter time range, more ticks)
    historical_major_tick_interval: int = 2  # Show major ticks every 2 days

    # Historical plot tick dates
    historical_tick_dates: List[str] = field(
        default_factory=lambda: [
            "2024-10-23",
            "2024-10-25",
            "2024-10-27",
            "2024-10-29",
            "2024-10-31",
            "2024-11-02",
            "2024-11-04",
        ]
    )

    def __post_init__(self):
        """Parse date strings into datetime objects for plotting."""
        self.date_range_start_parsed = datetime.strptime(
            self.date_range_start, "%Y-%m-%d"
        )
        self.date_range_end_parsed = datetime.strptime(self.date_range_end, "%Y-%m-%d")
        self.historical_date_range_start_parsed = datetime.strptime(
            self.historical_date_range_start, "%Y-%m-%d"
        )
        self.historical_date_range_end_parsed = datetime.strptime(
            self.historical_date_range_end, "%Y-%m-%d"
        )

        # Parse annotation positions
        self.hyperparameter_annotation_pos_parsed = (
            datetime.strptime(self.hyperparameter_annotation_pos[0], "%Y-%m-%d"),
            self.hyperparameter_annotation_pos[1],
        )
        self.mase_annotation_pos_parsed = (
            datetime.strptime(self.mase_annotation_pos[0], "%Y-%m-%d"),
            self.mase_annotation_pos[1],
        )

        # Parse historical tick dates
        self.historical_tick_dates_parsed = [
            datetime.strptime(date_str, "%Y-%m-%d")
            for date_str in self.historical_tick_dates
        ]


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""

    # Cross-validation settings
    n_splits: int = 5
    test_size: int = 7

    # Grid search parameters for Holt smoothing
    alpha_grid_min: float = 0.05
    alpha_grid_max: float = 0.5
    alpha_grid_step: float = 0.05
    beta_grid_min: float = 0.05
    beta_grid_max: float = 0.3
    beta_grid_step: float = 0.05

    # Electoral college settings
    swing_state_electoral_votes: int = 93
    trump_safe_electoral_votes: int = 219
    harris_safe_electoral_votes: int = 226

    @property
    def alpha_grid_numbers(self):
        """Generate grid search parameter range."""
        return np.arange(self.alpha_grid_min, self.alpha_grid_max, self.alpha_grid_step)

    @property
    def beta_grid_numbers(self):
        """Generate grid search parameter range."""
        return np.arange(self.beta_grid_min, self.beta_grid_max, self.beta_grid_step)


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

    # Candidate visualization configuration
    candidate_config: Dict[str, Dict] = field(
        default_factory=lambda: {
            "trump": {
                "full_name": "Donald Trump",
                "display_name": "Trump",
                "color": "red",
            },
            "harris": {
                "full_name": "Kamala Harris",
                "display_name": "Harris",
                "color": "blue",
            },
        }
    )

    # Core date strings
    earliest_available_data: str = "2024-07-21"
    forecast_start_date: str = "2024-10-23"
    election_day: str = "2024-11-05"

    # Plot configuration
    plot_config: PlotConfig = field(default_factory=PlotConfig)

    # Explicitly declare parsed date attributes for type checking
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

    def get_candidate_keys(self) -> List[str]:
        """Get list of candidate keys for iteration."""
        return list(self.candidate_config.keys())

    def get_candidate_full_names(self) -> List[str]:
        """Get list of candidate full names."""
        return [config["full_name"] for config in self.candidate_config.values()]
