# src/utils/logging_setup.py
"""Logging configuration utilities."""

import logging


def setup_logging(verbose=False, debug=False):
    """Setup logging with appropriate level for main.py and all src/ modules."""
    if debug:
        root_level = logging.INFO
        your_level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    elif verbose:
        root_level = logging.INFO
        your_level = logging.INFO
        format_str = "%(asctime)s - %(levelname)s - %(message)s"
    else:
        root_level = logging.WARNING
        your_level = logging.WARNING
        format_str = "%(message)s"

    logging.basicConfig(level=root_level, format=format_str, force=True)

    if debug:
        logging.getLogger(__name__).setLevel(your_level)
        logging.getLogger("src").setLevel(your_level)
        logging.getLogger("src.data").setLevel(your_level)
        logging.getLogger("src.models").setLevel(your_level)
        logging.getLogger("src.visualization").setLevel(your_level)
        logging.getLogger("src.pipeline").setLevel(your_level)
        logging.getLogger("src.utils").setLevel(your_level)

        # Silence noisy third-party libraries
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)

    return logging.getLogger(__name__)
