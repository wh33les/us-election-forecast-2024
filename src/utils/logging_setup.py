# src/utils/logging_setup.py
"""Logging configuration utilities."""

import logging


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
        # Set debug level for application-specific modules
        app_modules = [
            __name__,
            "src",
            "src.data",
            "src.models",
            "src.visualization",
            "src.pipeline",
            "src.utils",
        ]

        for module in app_modules:
            logging.getLogger(module).setLevel(your_level)

        # Silence noisy third-party libraries efficiently
        noisy_loggers = [
            "matplotlib",
            "matplotlib.font_manager",
            "matplotlib.pyplot",
            "PIL",
            "pandas",
            "statsmodels",
            "sklearn",
            "seaborn",
            # "urllib3",
            # "requests",
            "numpy",
        ]

        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    return logging.getLogger(__name__)
