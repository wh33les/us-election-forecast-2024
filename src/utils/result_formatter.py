# src/utils/result_formatter.py
"""Result formatting and logging utilities."""

import logging
from datetime import date

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Handles formatting and logging of forecast results."""

    def __init__(self, verbose=False, debug=False):
        self.verbose = verbose
        self.debug = debug

    def log_forecast_start(self, day_num, total_days, forecast_date):
        """Log the start of a forecast run."""
        date_str = forecast_date.strftime("%a %b %d, %Y")
        logger.info(f"\n{'='*40}")
        logger.info(f"DAY {day_num}/{total_days}: {date_str}")
        logger.info(f"{'='*40}")

    def log_forecast_success(self, forecast_date):
        """Log successful completion of a forecast."""
        prefix = "   ✅" if self.verbose else "✅"
        logger.info(f"{prefix} Completed forecast for {forecast_date}")

    def log_forecast_skip(self, forecast_date):
        """Log when a forecast is skipped."""
        logger.warning(f"⏭️  Skipped forecast for {forecast_date}")

    def log_forecast_error(self, forecast_date, error):
        """Log when a forecast fails."""
        if self.debug:
            logger.exception(f"❌ Failed forecast for {forecast_date}")
        else:
            logger.error(f"❌ Failed forecast for {forecast_date}: {error}")

    def log_forecast_results(self, forecast_results, forecast_date, election_day):
        """Log the results of a forecast, handling both electoral and non-electoral cases."""
        electoral_results = forecast_results["electoral_results"]

        if electoral_results:
            # Election Day - show both popular vote and electoral results
            self._log_electoral_results_detailed(electoral_results, forecast_date)
        else:
            # Non-Election Day - show popular vote predictions only
            self._log_popular_vote_only(forecast_results, forecast_date)

    def _log_popular_vote_only(self, forecast_results, forecast_date):
        """Log popular vote predictions for non-Election Day forecasts."""
        # Extract vote percentages from forecasts
        forecasts = forecast_results["forecasts"]
        baselines = forecast_results["baselines"]

        # Get final predictions (last value in the forecast arrays)
        trump_model = forecasts["trump"][-1] if len(forecasts["trump"]) > 0 else 0
        harris_model = forecasts["harris"][-1] if len(forecasts["harris"]) > 0 else 0
        trump_baseline = baselines["trump"][-1] if len(baselines["trump"]) > 0 else 0
        harris_baseline = baselines["harris"][-1] if len(baselines["harris"]) > 0 else 0

        prefix = "   📊" if self.verbose else "  "
        logger.info(
            f"{prefix} Model prediction: Trump {trump_model:.1f}%, Harris {harris_model:.1f}%"
        )
        logger.info(
            f"{prefix} Baseline prediction: Trump {trump_baseline:.1f}%, Harris {harris_baseline:.1f}%"
        )
        logger.info(
            f"{prefix} Interim forecast (electoral calculation only on Election Day)"
        )

    def _log_electoral_results_detailed(self, electoral_results, forecast_date):
        """Log detailed electoral college results for Election Day."""
        # Log popular vote predictions for both model and baseline
        model_trump_pct = electoral_results["model"]["trump_vote_pct"]
        model_harris_pct = electoral_results["model"]["harris_vote_pct"]
        baseline_trump_pct = electoral_results["baseline"]["trump_vote_pct"]
        baseline_harris_pct = electoral_results["baseline"]["harris_vote_pct"]

        prefix = "   📊" if self.verbose else "  "
        logger.info(
            f"{prefix} Model prediction: Trump {model_trump_pct:.1f}%, Harris {model_harris_pct:.1f}%"
        )
        logger.info(
            f"{prefix} Baseline prediction: Trump {baseline_trump_pct:.1f}%, Harris {baseline_harris_pct:.1f}%"
        )

        # Log electoral college results
        self._log_electoral_outcomes(electoral_results)

    def _log_electoral_outcomes(self, electoral_results):
        """Log detailed electoral college results for both model and baseline."""
        prefix = "   🏆" if self.verbose else "  "

        # Log Model Results
        model = electoral_results["model"]
        model_winner = model["winner"]
        model_winner_evs = (
            model["trump_electoral_votes"]
            if model_winner == "Trump"
            else model["harris_electoral_votes"]
        )

        logger.info(
            f"{prefix} MODEL Electoral outcome: {model_winner} wins with {model_winner_evs} electoral votes"
        )
        self._log_state_allocations(model, "MODEL", prefix)

        # Log Baseline Results
        baseline = electoral_results["baseline"]
        baseline_winner = baseline["winner"]
        baseline_winner_evs = (
            baseline["trump_electoral_votes"]
            if baseline_winner == "Trump"
            else baseline["harris_electoral_votes"]
        )

        logger.info(
            f"{prefix} BASELINE Electoral outcome: {baseline_winner} wins with {baseline_winner_evs} electoral votes"
        )
        self._log_state_allocations(baseline, "BASELINE", prefix)

        # Log comparison if different
        if model_winner != baseline_winner:
            logger.info(f"{prefix} ⚠️  MODEL and BASELINE predict different winners!")
        elif model["trump_electoral_votes"] != baseline["trump_electoral_votes"]:
            logger.info(f"{prefix} ℹ️  Same winner, different electoral vote counts")
        else:
            logger.info(f"{prefix} ✅ MODEL and BASELINE predictions agree completely")

    def _log_state_allocations(self, results, prediction_type, prefix):
        """Log state allocations for a given prediction (model or baseline)."""
        state_names = {
            "AZ": "Arizona (11)",
            "GA": "Georgia (16)",
            "NC": "North Carolina (16)",
            "NV": "Nevada (6)",
            "PA": "Pennsylvania (19)",
            "WI": "Wisconsin (10)",
            "MI": "Michigan (15)",
        }

        trump_swing = max(0, results["trump_electoral_votes"] - 219)
        harris_swing = max(0, results["harris_electoral_votes"] - 226)

        for candidate, states_key, total_evs, swing_evs in [
            ("Trump", "trump_states", results["trump_electoral_votes"], trump_swing),
            (
                "Harris",
                "harris_states",
                results["harris_electoral_votes"],
                harris_swing,
            ),
        ]:
            states = results[states_key]

            # Fix: Ensure all elements are strings (filter out None values)
            if states:
                state_details = [
                    state_names.get(state, str(state))
                    for state in states
                    if state is not None
                ]
            else:
                state_details = ["None"]

            safe_votes = 219 if candidate == "Trump" else 226
            logger.info(
                f"      {prediction_type} {candidate}: {total_evs} total = {safe_votes} safe + {swing_evs} swing"
            )
            logger.info(
                f"      {prediction_type} {candidate} swing states: {', '.join(state_details)}"
            )
