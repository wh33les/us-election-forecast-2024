# src/models/electoral_calculator.py
"""Electoral college calculation for election forecasting."""

import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ElectoralCollegeCalculator:
    """Calculate electoral college outcomes from polling predictions."""

    def __init__(self, config):
        self.config = config
        self.swing_states_map = {
            "AZ": 11,
            "GA": 16,
            "NC": 16,
            "NV": 6,
            "PA": 19,
            "WI": 10,
            "MI": 15,
        }
        # UPDATED: Use config instead of hardcoded values
        self.trump_safe_votes = config.trump_safe_electoral_votes
        self.harris_safe_votes = config.harris_safe_electoral_votes
        self.total_swing_votes = config.swing_state_electoral_votes

    def extract_final_predictions(
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

    def calculate_swing_state_allocation(
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

    def calculate_all_outcomes(self, df_cleaned: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate electoral outcomes for both model and baseline predictions."""
        logger.info("Calculating electoral college outcomes...")

        predictions = self.extract_final_predictions(df_cleaned)

        # Calculate outcomes for both model and baseline
        model_outcome = self.calculate_swing_state_allocation(
            predictions["model"]["trump_normalized"],
            predictions["model"]["harris_normalized"],
        )
        model_outcome.update(
            {
                "trump_vote_pct": predictions["model"]["trump_raw"],
                "harris_vote_pct": predictions["model"]["harris_raw"],
            }
        )

        baseline_outcome = self.calculate_swing_state_allocation(
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
