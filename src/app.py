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
    """Launch the CS2 Neural VMAT Upscaler application."""
    configure_logging()
    logger = logging.getLogger("CS2Upscaler")
    logger.info("Starting CS2 Neural VMAT Upscaler...")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    app = QApplication(sys.argv)
    app.setApplicationName("CS2 Neural VMAT Upscaler")
    app.setOrganizationName("CS2 Neural VMAT Upscaler")
    app.setApplicationVersion("2.0.0")

    # Set application icon
    import os
    icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

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
