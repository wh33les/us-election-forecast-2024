# src/utils/result_formatter.py
"""Result formatting and logging utilities."""

import logging
from datetime import date

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Handles formatting and logging of forecast results."""

    def __init__(self, verbose=False, debug=False):
        """Initialize with verbosity settings."""
        self.verbose = verbose
        self.debug = debug

    def log_forecast_start(self, day_num, total_days, forecast_date):
        """Log the start of a forecast run."""
        if self.verbose:
            logger.info(
                f"\n📅 DAY {day_num}/{total_days}: {forecast_date.strftime('%a %b %d, %Y')}"
            )
            logger.info("─" * 50)
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"FORECAST FOR {forecast_date} ({day_num}/{total_days})")
            logger.info(f"{'='*60}")

    def log_forecast_success(self, forecast_date):
        """Log successful completion of a forecast."""
        if self.verbose:
            logger.info(f"   ✅ COMPLETED {forecast_date.strftime('%a %b %d')}")
        else:
            logger.info(f"✅ Completed forecast for {forecast_date}")

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
        trump_pred_pct = electoral_results["model"]["trump_vote_pct"]
        harris_pred_pct = electoral_results["model"]["harris_vote_pct"]

        if self.verbose:
            logger.info(
                f"   📊 Model prediction: Trump {trump_pred_pct:.1f}%, Harris {harris_pred_pct:.1f}%"
            )
        else:
            logger.info(
                f"   Model prediction: Trump {trump_pred_pct:.1f}%, Harris {harris_pred_pct:.1f}%"
            )

        election_day_date = date(2024, 11, 5)
        if forecast_date == election_day_date:
            self._log_electoral_results(electoral_results)
        else:
            if self.verbose:
                logger.info(
                    "   📈 Interim forecast (electoral calculation only on Election Day)"
                )
            else:
                logger.info(
                    "   Interim forecast (electoral calculation only on Election Day)"
                )

    def _log_electoral_results(self, electoral_results):
        """Log detailed electoral college results."""
        trump_swing_evs = max(
            0, electoral_results["model"]["trump_electoral_votes"] - 219
        )
        harris_swing_evs = max(
            0, electoral_results["model"]["harris_electoral_votes"] - 226
        )

        state_names = {
            "AZ": "Arizona (11)",
            "GA": "Georgia (16)",
            "NC": "North Carolina (16)",
            "NV": "Nevada (6)",
            "PA": "Pennsylvania (19)",
            "WI": "Wisconsin (10)",
            "MI": "Michigan (15)",
        }

        winner = electoral_results["model"]["winner"]
        winner_evs = (
            electoral_results["model"]["trump_electoral_votes"]
            if winner == "Trump"
            else electoral_results["model"]["harris_electoral_votes"]
        )

        if self.verbose:
            logger.info(
                f"   🏆 Electoral outcome: {winner} wins with {winner_evs} electoral votes"
            )
            logger.info("   📊 Electoral Vote Breakdown:")
            logger.info(
                f"      Trump: {electoral_results['model']['trump_electoral_votes']} total = 219 safe + {trump_swing_evs} swing"
            )

            if electoral_results["model"]["trump_states"]:
                trump_state_details = [
                    state_names.get(state, state)
                    for state in electoral_results["model"]["trump_states"]
                ]
                logger.info(
                    f"      Trump swing states: {', '.join(trump_state_details)}"
                )
            else:
                logger.info(f"      Trump swing states: None")

            logger.info(
                f"      Harris: {electoral_results['model']['harris_electoral_votes']} total = 226 safe + {harris_swing_evs} swing"
            )

            if electoral_results["model"]["harris_states"]:
                harris_state_details = [
                    state_names.get(state, state)
                    for state in electoral_results["model"]["harris_states"]
                ]
                logger.info(
                    f"      Harris swing states: {', '.join(harris_state_details)}"
                )
            else:
                logger.info(f"      Harris swing states: None")
        else:
            logger.info(
                f"   Electoral outcome: {winner} wins with {winner_evs} electoral votes"
            )
            logger.info("   Electoral Vote Breakdown:")
            logger.info(
                f"   Trump: {electoral_results['model']['trump_electoral_votes']} total = 219 safe + {trump_swing_evs} swing"
            )

            if electoral_results["model"]["trump_states"]:
                trump_state_details = [
                    state_names.get(state, state)
                    for state in electoral_results["model"]["trump_states"]
                ]
                logger.info(f"   Trump swing states: {', '.join(trump_state_details)}")
            else:
                logger.info(f"   Trump swing states: None")

            logger.info(
                f"   Harris: {electoral_results['model']['harris_electoral_votes']} total = 226 safe + {harris_swing_evs} swing"
            )

            if electoral_results["model"]["harris_states"]:
                harris_state_details = [
                    state_names.get(state, state)
                    for state in electoral_results["model"]["harris_states"]
                ]
                logger.info(
                    f"   Harris swing states: {', '.join(harris_state_details)}"
                )
            else:
                logger.info(f"   Harris swing states: None")
