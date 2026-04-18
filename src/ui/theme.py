"""
Professional Dark Theme for the CS2 Texture Upscaler.

Neutral dark tones with a sharp blue accent, consistent spacing,
and proper DPI-aware sizing.
"""

# ── Colour palette ───────────────────────────────────────────────────
ACCENT = "#2B7DE9"
ACCENT_HOVER = "#1A6DD8"
ACCENT_PRESSED = "#1060C2"
BG_DARK = "#1E1E1E"
BG_MID = "#272727"
BG_LIGHT = "#333333"
BG_INPUT = "#1A1A1A"
TEXT_PRIMARY = "#D4D4D4"
TEXT_SECONDARY = "#858585"
TEXT_DISABLED = "#555555"
BORDER = "#3C3C3C"
SUCCESS = "#4EC050"
WARNING = "#DDB100"
ERROR = "#E04040"
SURFACE = "#242424"


def get_stylesheet() -> str:
    """Return the complete application stylesheet."""
    return f"""
    /* ── Global ──────────────────────────────────────────── */
    QWidget {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
        font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
        font-size: 13px;
        outline: none;
    }}

    /* ── Main Window ─────────────────────────────────────── */
    QMainWindow {{
        background-color: {BG_DARK};
    }}
    QMainWindow::separator {{
        background: {TEXT_DISABLED};
        width: 4px;
        height: 4px;
        border-radius: 2px;
    }}
    QMainWindow::separator:hover {{
        background: {ACCENT};
    }}

    /* ── Menu Bar ────────────────────────────────────────── */
    QMenuBar {{
        background-color: {BG_DARK};
        border-bottom: 1px solid {BORDER};
        padding: 1px;
        font-size: 12px;
    }}
    QMenuBar::item {{
        padding: 5px 10px;
        border-radius: 3px;
    }}
    QMenuBar::item:selected {{
        background-color: {BG_LIGHT};
    }}
    QMenu {{
        background-color: {BG_MID};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 28px 6px 20px;
        border-radius: 3px;
    }}
    QMenu::item:selected {{
        background-color: {ACCENT};
        color: white;
    }}
    QMenu::separator {{
        height: 1px;
        background: {BORDER};
        margin: 4px 8px;
    }}

    /* ── Tool Bar ────────────────────────────────────────── */
    QToolBar {{
        background-color: {BG_DARK};
        border-bottom: 1px solid {BORDER};
        padding: 2px 4px;
        spacing: 2px;
    }}
    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        padding: 4px 8px;
        border-radius: 4px;
        color: {TEXT_PRIMARY};
        font-size: 12px;
    }}
    QToolButton:hover {{
        background-color: {BG_LIGHT};
        border: 1px solid {BORDER};
    }}
    QToolButton:pressed {{
        background-color: {SURFACE};
    }}
    QToolButton:checked {{
        background-color: {ACCENT};
        border: 1px solid {ACCENT};
        color: white;
    }}

    /* ── Tab Widgets ─────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        background-color: {BG_MID};
        border-radius: 4px;
    }}
    QTabBar::tab {{
        background-color: {BG_DARK};
        color: {TEXT_SECONDARY};
        padding: 7px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 1px;
        font-size: 12px;
    }}
    QTabBar::tab:hover {{
        background-color: {BG_LIGHT};
        color: {TEXT_PRIMARY};
    }}
    QTabBar::tab:selected {{
        background-color: {BG_MID};
        color: {TEXT_PRIMARY};
        border-bottom: 2px solid {ACCENT};
    }}

    /* ── Scroll Area ─────────────────────────────────────── */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: transparent;
    }}

    /* ── Tree / Table ────────────────────────────────────── */
    QTreeView, QTreeWidget, QTableWidget {{
        background-color: {BG_MID};
        border: 1px solid {BORDER};
        border-radius: 4px;
        alternate-background-color: {SURFACE};
        outline: none;
        font-size: 12px;
    }}
    QTreeView::item, QTreeWidget::item {{
        padding: 3px 4px;
        min-height: 22px;
    }}
    QTreeView::item:selected, QTreeWidget::item:selected {{
        background-color: {ACCENT};
        color: white;
    }}
    QTreeView::item:hover:!selected, QTreeWidget::item:hover:!selected {{
        background-color: {BG_LIGHT};
    }}
    QHeaderView::section {{
        background-color: {SURFACE};
        color: {TEXT_SECONDARY};
        padding: 4px 6px;
        border: none;
        border-bottom: 1px solid {BORDER};
        border-right: 1px solid {BORDER};
        font-size: 11px;
        font-weight: 600;
    }}

    /* ── Buttons ─────────────────────────────────────────── */
    QPushButton {{
        background-color: {BG_LIGHT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        padding: 5px 14px;
        border-radius: 4px;
        font-size: 12px;
    }}
    QPushButton:hover {{
        background-color: {SURFACE};
        border: 1px solid {TEXT_SECONDARY};
    }}
    QPushButton:pressed {{
        background-color: {BG_DARK};
    }}
    QPushButton:disabled {{
        color: {TEXT_DISABLED};
        background-color: {BG_DARK};
        border: 1px solid {BG_MID};
    }}
    QPushButton#primaryButton, QPushButton#Primary {{
        background-color: {ACCENT};
        color: white;
        border: none;
        font-weight: 600;
    }}
    QPushButton#primaryButton:hover, QPushButton#Primary:hover {{
        background-color: {ACCENT_HOVER};
    }}
    QPushButton#primaryButton:pressed, QPushButton#Primary:pressed {{
        background-color: {ACCENT_PRESSED};
    }}

    /* ── Inputs ──────────────────────────────────────────── */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {BG_INPUT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        padding: 4px 8px;
        border-radius: 3px;
        font-size: 12px;
        min-height: 22px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {ACCENT};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: none;
    }}
    QComboBox QAbstractItemView {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        selection-background-color: {ACCENT};
        selection-color: white;
    }}

    /* ── Group Boxes ─────────────────────────────────────── */
    QGroupBox {{
        background-color: {BG_MID};
        border: 1px solid {BORDER};
        border-radius: 4px;
        margin-top: 14px;
        padding: 14px 8px 8px 8px;
        font-size: 12px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0 4px;
        background-color: transparent;
        color: {ACCENT};
        font-weight: 600;
        font-size: 12px;
    }}

    /* ── Scroll Bars ─────────────────────────────────────── */
    QScrollBar:vertical {{
        background-color: transparent;
        width: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {TEXT_DISABLED};
        min-height: 24px;
        border-radius: 5px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {TEXT_SECONDARY};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}
    QScrollBar:horizontal {{
        background-color: transparent;
        height: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {TEXT_DISABLED};
        min-width: 24px;
        border-radius: 5px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background-color: {TEXT_SECONDARY};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
    }}

    /* ── Sliders ─────────────────────────────────────────── */
    QSlider::groove:horizontal {{
        height: 4px;
        background: {BORDER};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{
        background: {ACCENT};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {TEXT_PRIMARY};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: white;
    }}

    /* ── CheckBoxes ──────────────────────────────────────── */
    QCheckBox {{
        spacing: 6px;
        font-size: 12px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid {BORDER};
        background: {BG_INPUT};
    }}
    QCheckBox::indicator:unchecked:hover {{
        border: 1px solid {TEXT_SECONDARY};
    }}
    QCheckBox::indicator:checked {{
        background: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    /* ── Progress Bar ────────────────────────────────────── */
    QProgressBar {{
        border: 1px solid {BORDER};
        border-radius: 3px;
        background-color: {BG_DARK};
        text-align: center;
        color: {TEXT_PRIMARY};
        font-size: 11px;
        min-height: 18px;
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: 2px;
    }}

    /* ── Splitters ───────────────────────────────────────── */
    QSplitter::handle {{
        background-color: {BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 4px;
        margin: 2px 0;
        border-radius: 2px;
        background-color: {TEXT_DISABLED};
    }}
    QSplitter::handle:vertical {{
        height: 4px;
        margin: 0 2px;
        border-radius: 2px;
        background-color: {TEXT_DISABLED};
    }}
    QSplitter::handle:hover {{
        background-color: {ACCENT};
    }}

    /* ── Status Bar ──────────────────────────────────────── */
    QStatusBar {{
        background-color: {SURFACE};
        border-top: 1px solid {BORDER};
        font-size: 11px;
        color: {TEXT_SECONDARY};
        min-height: 28px;
        max-height: 32px;
        padding: 2px 4px;
    }}
    QStatusBar::item {{
        border: none;
    }}
    QStatusBar QLabel {{
        font-size: 11px;
        padding: 0 2px;
    }}

    /* ── Dock Widgets ────────────────────────────────────── */
    QDockWidget {{
        titlebar-close-icon: none;
        font-size: 12px;
    }}
    QDockWidget::title {{
        background-color: {SURFACE};
        text-align: left;
        padding: 6px 8px;
        border-bottom: 1px solid {BORDER};
    }}

    /* ── Labels ──────────────────────────────────────────── */
    QLabel {{
        font-size: 12px;
    }}
    QLabel#sectionHeader {{
        font-size: 13px;
        font-weight: 700;
        color: {TEXT_PRIMARY};
        padding: 2px 0;
    }}

    /* ── Form Layout ─────────────────────────────────────── */
    QFormLayout {{
        spacing: 4px;
    }}

    /* ── Tooltips ────────────────────────────────────────── */
    QToolTip {{
        background-color: {BG_MID};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        padding: 4px 8px;
        border-radius: 3px;
        font-size: 11px;
    }}
    """
