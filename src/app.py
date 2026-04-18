"""
Application entry point and configuration.
"""

import logging
import sys


def configure_logging():
    """Set up application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run():
    """Launch the CS2 Texture Upscaler application."""
    configure_logging()
    logger = logging.getLogger("CS2Upscaler")
    logger.info("Starting CS2 Texture Upscaler...")

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("CS2 Texture Upscaler")
    app.setOrganizationName("CS2 Texture Upscaler")
    app.setApplicationVersion("2.0.0")

    # Apply dark theme
    from .ui.theme import get_stylesheet
    app.setStyleSheet(get_stylesheet())

    # Import MainWindow here so heavy deps (torch, basicsr) load *after*
    # QApplication exists and startup messages are already visible.
    from .ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    logger.info("Application ready.")
    sys.exit(app.exec())
