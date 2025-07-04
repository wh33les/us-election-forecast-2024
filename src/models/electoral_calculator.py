# src/models/electoral_calculator.py
"""Electoral college calculation for election forecasting."""

import pandas as pd
from datetime import date
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ElectoralCollegeCalculator:
    def __init__(self, model_config, data_config):
        self.model_config = model_config
        self.data_config = data_config
        self.election_day = data_config.election_day_parsed  # From data_config

        self.swing_states_map = {
            "AZ": 11,
            "GA": 16,
            "NC": 16,
            "NV": 6,
            "PA": 19,
            "WI": 10,
            "MI": 15,
        }

        # Electoral vote counts from model_config
        self.trump_safe_votes = model_config.trump_safe_electoral_votes
        self.harris_safe_votes = model_config.harris_safe_electoral_votes
        self.total_swing_votes = model_config.swing_state_electoral_votes

    def calculate_electoral_outcomes_if_election_day(
        self, forecasts: Dict, baselines: Dict, forecast_date: date
    ) -> Optional[Dict]:
        """Calculate electoral outcomes only on Election Day."""
        if forecast_date != self.election_day:
            return None

        # Only run electoral college math on Election Day
        if forecasts["trump"].size > 0 and forecasts["harris"].size > 0:
            # Create electoral data (data preparation)
            electoral_data = pd.DataFrame(
                [
                    {
                        "candidate_name": "Donald Trump",
                        "end_date": self.election_day,
                        "daily_average": None,
                        "model": forecasts["trump"][-1],
                        "drift_pred": baselines["trump"][-1],
                    },
                    {
                        "candidate_name": "Kamala Harris",
                        "end_date": self.election_day,
                        "daily_average": None,
                        "model": forecasts["harris"][-1],
                        "drift_pred": baselines["harris"][-1],
                    },
                ]
            )

            logger.info("Calculating electoral college outcomes...")

            # Electoral calculation logic (previously in calculate_all_outcomes)
            predictions = self._extract_final_predictions(electoral_data)

            # Calculate outcomes for both model and baseline
            model_outcome = self._calculate_swing_state_allocation(
                predictions["model"]["trump_normalized"],
                predictions["model"]["harris_normalized"],
            )
            model_outcome.update(
                {
                    "trump_vote_pct": predictions["model"]["trump_raw"],
                    "harris_vote_pct": predictions["model"]["harris_raw"],
                }
            )

            baseline_outcome = self._calculate_swing_state_allocation(
                predictions["baseline"]["trump_normalized"],
                predictions["baseline"]["harris_normalized"],
            )
            baseline_outcome.update(
                {
                    "trump_vote_pct": predictions["baseline"]["trump_raw"],
                    "harris_vote_pct": predictions["baseline"]["harris_raw"],
                }
            )

            return {
                "model": model_outcome,
                "baseline": baseline_outcome,
                "predictions": predictions,
            }

        return None

    def _extract_final_predictions(
        self, df_cleaned: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Extract final prediction percentages from cleaned dataframe."""
        logger.info("Extracting final predictions...")

        try:
            # Get final predictions for each candidate
            trump_idx = round(len(df_cleaned) / 2) - 1
            harris_idx = len(df_cleaned) - 1

            trump_model = df_cleaned["model"].iloc[trump_idx]
            harris_model = df_cleaned["model"].iloc[harris_idx]
            trump_baseline = df_cleaned["drift_pred"].iloc[trump_idx]
            harris_baseline = df_cleaned["drift_pred"].iloc[harris_idx]

            # Validate and normalize predictions
            if any(
                pd.isna([trump_model, harris_model, trump_baseline, harris_baseline])
            ):
                raise ValueError("Predictions contain NaN values")

            model_total = trump_model + harris_model
            baseline_total = trump_baseline + harris_baseline

            return {
                "model": {
                    "trump_normalized": trump_model / model_total,
                    "harris_normalized": harris_model / model_total,
                    "trump_raw": trump_model,
                    "harris_raw": harris_model,
                },
                "baseline": {
                    "trump_normalized": trump_baseline / baseline_total,
                    "harris_normalized": harris_baseline / baseline_total,
                    "trump_raw": trump_baseline,
                    "harris_raw": harris_baseline,
                },
            }

        except Exception as e:
            logger.error(f"Error extracting predictions: {e}")
            raise

    def _calculate_swing_state_allocation(
        self, trump_share: float, harris_share: float
    ) -> Dict:
        """Calculate swing state allocation based on vote shares."""
        logger.info(
            f"Allocating swing states: Trump {trump_share:.1%}, Harris {harris_share:.1%}"
        )

        trump_swing_votes = round(trump_share * self.total_swing_votes)

        trump_total = self.trump_safe_votes
        harris_total = self.harris_safe_votes
        trump_states = []
        harris_states = []

        remaining_votes = trump_swing_votes

        # Allocate swing states
        for state_code, electoral_votes in self.swing_states_map.items():
            if remaining_votes >= electoral_votes:
                trump_total += electoral_votes
                trump_states.append(state_code)
                remaining_votes -= electoral_votes
            else:
                harris_total += electoral_votes
                harris_states.append(state_code)

        winner = "Trump" if trump_total > harris_total else "Harris"

        logger.info(
            f"Electoral outcome: {winner} wins with {max(trump_total, harris_total)} votes"
        )

        return {
            "trump_states": trump_states,
            "harris_states": harris_states,
            "trump_electoral_votes": trump_total,
            "harris_electoral_votes": harris_total,
            "winner": winner,
        }
