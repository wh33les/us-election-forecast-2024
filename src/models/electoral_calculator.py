# src/models/electoral_calculator.py
"""Electoral college calculation for election forecasting."""

import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ElectoralCollegeCalculator:
    """Calculate electoral college outcomes from polling predictions."""

    def __init__(self, config):
        """Initialize with model configuration."""
        self.config = config

        # Electoral vote mapping
        self.swing_states_map = {
            "AZ": 11,
            "GA": 16,
            "NC": 16,
            "NV": 6,
            "PA": 19,
            "WI": 10,
            "MI": 15,
        }

        # Fixed electoral votes (safe states)
        self.trump_safe_votes = 219
        self.harris_safe_votes = 226
        self.total_swing_votes = 93

    def extract_final_predictions(
        self, df_cleaned: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Extract final prediction percentages from cleaned dataframe."""
        logger.info("Extracting final predictions from dataframe...")

        try:
            # Extract model predictions (final values for each candidate)
            trump_model_idx = round(len(df_cleaned) / 2) - 1
            harris_model_idx = len(df_cleaned) - 1

            trump_pred_pct = df_cleaned["model"].iloc[trump_model_idx]
            harris_pred_pct = df_cleaned["model"].iloc[harris_model_idx]

            # Extract baseline predictions
            trump_b_pct = df_cleaned["drift_pred"].iloc[trump_model_idx]
            harris_b_pct = df_cleaned["drift_pred"].iloc[harris_model_idx]

            # Validate predictions
            if pd.isna(trump_pred_pct) or pd.isna(harris_pred_pct):
                raise ValueError("Model predictions contain NaN values")

            if pd.isna(trump_b_pct) or pd.isna(harris_b_pct):
                raise ValueError("Baseline predictions contain NaN values")

            # Normalize to ensure they sum to 100%
            model_total = trump_pred_pct + harris_pred_pct
            baseline_total = trump_b_pct + harris_b_pct

            predictions = {
                "model": {
                    "trump_raw": trump_pred_pct,
                    "harris_raw": harris_pred_pct,
                    "trump_normalized": trump_pred_pct / model_total,
                    "harris_normalized": harris_pred_pct / model_total,
                },
                "baseline": {
                    "trump_raw": trump_b_pct,
                    "harris_raw": harris_b_pct,
                    "trump_normalized": trump_b_pct / baseline_total,
                    "harris_normalized": harris_b_pct / baseline_total,
                },
            }

            logger.info(
                f"Model predictions - Trump: {trump_pred_pct:.1f}%, Harris: {harris_pred_pct:.1f}%"
            )
            logger.info(
                f"Baseline predictions - Trump: {trump_b_pct:.1f}%, Harris: {harris_b_pct:.1f}%"
            )

            return predictions

        except Exception as e:
            logger.error(f"Error extracting predictions: {e}")
            raise

    def calculate_swing_state_allocation(
        self, trump_share: float, harris_share: float
    ) -> Dict[str, any]:
        """Calculate swing state allocation based on vote shares."""
        logger.info(
            f"Calculating swing state allocation: Trump {trump_share:.1%}, Harris {harris_share:.1%}"
        )

        trump_swing_votes = round(trump_share * self.total_swing_votes)

        trump_total = self.trump_safe_votes
        harris_total = self.harris_safe_votes
        trump_states = []
        harris_states = []

        remaining_trump_votes = trump_swing_votes

        # Allocate swing states based on vote share
        for state_code, electoral_votes in self.swing_states_map.items():
            if electoral_votes < remaining_trump_votes:
                # Trump gets this state
                remaining_trump_votes -= electoral_votes
                trump_total += electoral_votes
                trump_states.append(state_code)
            elif remaining_trump_votes > 0:
                # Check if Trump gets majority of remaining votes for this state
                if remaining_trump_votes / electoral_votes >= 0.5:
                    trump_total += electoral_votes
                    trump_states.append(state_code)
                else:
                    harris_total += electoral_votes
                    harris_states.append(state_code)
                remaining_trump_votes = 0
            else:
                # No more Trump votes, Harris gets remaining states
                harris_total += electoral_votes
                harris_states.append(state_code)

        # Determine winner
        winner = "Trump" if trump_total >= harris_total else "Harris"

        result = {
            "trump_states": trump_states,
            "harris_states": harris_states,
            "trump_electoral_votes": trump_total,
            "harris_electoral_votes": harris_total,
            "trump_swing_votes": trump_swing_votes,
            "harris_swing_votes": round(harris_share * self.total_swing_votes),
            "winner": winner,
        }

        logger.info(
            f"Electoral outcome: {winner} wins with {max(trump_total, harris_total)} electoral votes"
        )
        logger.info(f"Trump states: {trump_states} ({trump_total} votes)")
        logger.info(f"Harris states: {harris_states} ({harris_total} votes)")

        return result

    def calculate_all_outcomes(self, df_cleaned: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate electoral outcomes for both model and baseline predictions."""
        logger.info("Calculating all electoral college outcomes...")

        # Extract predictions
        predictions = self.extract_final_predictions(df_cleaned)

        # Calculate outcomes for model predictions
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

        # Calculate outcomes for baseline predictions
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

        results = {
            "model": model_outcome,
            "baseline": baseline_outcome,
            "predictions": predictions,
        }

        logger.info("Electoral college calculations completed")
        return results
