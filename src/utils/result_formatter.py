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

        if self.verbose:
            logger.info(f"\n📅 DAY {day_num}/{total_days}: {date_str}")
            logger.info("─" * 50)
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"FORECAST FOR {forecast_date} ({day_num}/{total_days})")
            logger.info(f"{'='*60}")

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
        """Log the results of a forecast."""
        electoral_results = forecast_results["electoral_results"]
        trump_pct = electoral_results["model"]["trump_vote_pct"]
        harris_pct = electoral_results["model"]["harris_vote_pct"]

        prefix = "   📊" if self.verbose else "  "
        logger.info(
            f"{prefix} Model prediction: Trump {trump_pct:.1f}%, Harris {harris_pct:.1f}%"
        )

        if forecast_date == date(2024, 11, 5):
            self._log_electoral_results(electoral_results)
        else:
            logger.info(
                f"{prefix} Interim forecast (electoral calculation only on Election Day)"
            )

    def _log_electoral_results(self, electoral_results):
        """Log detailed electoral college results."""
        model = electoral_results["model"]
        winner = model["winner"]
        winner_evs = (
            model["trump_electoral_votes"]
            if winner == "Trump"
            else model["harris_electoral_votes"]
        )

        trump_swing = max(0, model["trump_electoral_votes"] - 219)
        harris_swing = max(0, model["harris_electoral_votes"] - 226)

        state_names = {
            "AZ": "Arizona (11)",
            "GA": "Georgia (16)",
            "NC": "North Carolina (16)",
            "NV": "Nevada (6)",
            "PA": "Pennsylvania (19)",
            "WI": "Wisconsin (10)",
            "MI": "Michigan (15)",
        }

        prefix = "   🏆" if self.verbose else "  "
        logger.info(
            f"{prefix} Electoral outcome: {winner} wins with {winner_evs} electoral votes"
        )

        # Log state allocations
        for candidate, states_key, total_evs, swing_evs in [
            ("Trump", "trump_states", model["trump_electoral_votes"], trump_swing),
            ("Harris", "harris_states", model["harris_electoral_votes"], harris_swing),
        ]:
            states = model[states_key]
            state_details = (
                [state_names.get(state, state) for state in states]
                if states
                else ["None"]
            )

            safe_votes = 219 if candidate == "Trump" else 226
            logger.info(
                f"      {candidate}: {total_evs} total = {safe_votes} safe + {swing_evs} swing"
            )
            logger.info(f"      {candidate} swing states: {', '.join(state_details)}")
