"""
Custom reusable widgets for CS2 Texture Upscaler UI.
"""

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QColor, QFont, QPixmap, QImage, QIcon
from PyQt6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame,
    QProgressBar, QPushButton, QSizePolicy, QGraphicsDropShadowEffect,
)

import numpy as np

from .theme import ACCENT, BG_INPUT, BG_LIGHT, BG_MID, TEXT_SECONDARY, SUCCESS, ERROR


class StatusIndicator(QWidget):
    """Small coloured dot indicating status."""

    def __init__(self, color: str = ACCENT, size: int = 10, parent=None):
        super().__init__(parent)
        self._color = color
        self._size = size
        self.setFixedSize(size, size)

    def set_color(self, color: str):
        self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(self._color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, self._size, self._size)


class InfoCard(QFrame):
    """Rounded card with title, value, and optional icon."""

    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setObjectName("infoCard")
        self.setStyleSheet(f"""
            #infoCard {{
                background-color: {BG_MID};
                border: 1px solid #2A2A4A;
                border-radius: 10px;
                padding: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self._title = QLabel(title)
        self._title.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")

        self._value = QLabel(value)
        self._value.setStyleSheet(f"color: white; font-size: 16px; font-weight: 700;")

        layout.addWidget(self._title)
        layout.addWidget(self._value, 1)

    def set_value(self, value: str):
        self._value.setText(value)


class GPUInfoBar(QFrame):
    """Horizontal bar showing GPU name, VRAM usage, and temperature."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            background-color: transparent;
            border: none;
            padding: 0 4px;
        """)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)

        self._gpu_label = QLabel("GPU: Detecting...")
        self._gpu_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")

        self._vram_bar = QProgressBar()
        self._vram_bar.setFixedWidth(120)
        self._vram_bar.setFixedHeight(16)
        self._vram_bar.setTextVisible(True)
        self._vram_bar.setFormat("%v / %m MB")

        self._status = StatusIndicator(SUCCESS, size=8)

        layout.addWidget(self._status)
        layout.addWidget(self._gpu_label)
        layout.addStretch()
        layout.addWidget(self._vram_bar)

    def update_info(self, info: dict):
        name = info.get("name", "CPU")
        vram_total = info.get("vram_total", 0)
        vram_used = info.get("vram_used", 0)
        gen = info.get("generation", "")
        cc = info.get("compute_capability", "")
        gen_label = {
            "turing": "Turing",
            "ampere": "Ampere",
            "ada": "Ada Lovelace",
            "blackwell": "Blackwell",
        }.get(gen, gen.title() if gen else "")
        suffix = f"  [{gen_label} · SM {cc}]" if gen_label and cc else ""
        self._gpu_label.setText(f"GPU: {name}{suffix}")
        if vram_total > 0:
            self._vram_bar.setMaximum(int(vram_total * 1024))
            self._vram_bar.setValue(int(vram_used * 1024))
            self._vram_bar.setFormat(f"{vram_used:.1f} / {vram_total:.1f} GB")
            self._status.set_color(SUCCESS)
        else:
            self._vram_bar.setMaximum(100)
            self._vram_bar.setValue(0)
            self._vram_bar.setFormat("CPU Mode")
            self._status.set_color(ERROR)


class DragDropArea(QFrame):
    """Large drag-and-drop area for importing textures."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self._is_hovering = False
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_INPUT};
                border: 2px dashed #3A3A5A;
                border-radius: 12px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("📁")
        icon.setStyleSheet("font-size: 48px; border: none; background: transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text = QLabel("Drag & Drop Textures or Materials Here")
        text.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 16px; font-weight: 600; border: none; background: transparent;")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub = QLabel("Or use 📂 Open Materials Folder to scan recursively\nPNG, JPEG, TGA, BMP, TIFF, DDS, WebP, EXR")
        sub.setStyleSheet(f"color: #606078; font-size: 11px; border: none; background: transparent;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn = QPushButton("  Browse Files  ")
        btn.setObjectName("primaryButton")
        btn.clicked.connect(self._browse)

        layout.addWidget(icon)
        layout.addWidget(text)
        layout.addWidget(sub)
        layout.addSpacing(12)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _browse(self):
        from PyQt6.QtWidgets import QFileDialog
        from ..core.texture_io import build_file_filter
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Textures", "", build_file_filter()
        )
        if files:
            self.files_dropped.emit(files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._is_hovering = True
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {BG_INPUT};
                    border: 2px dashed {ACCENT};
                    border-radius: 12px;
                }}
            """)

    def dragLeaveEvent(self, event):
        self._is_hovering = False
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_INPUT};
                border: 2px dashed #3A3A5A;
                border-radius: 12px;
            }}
        """)

    def dropEvent(self, event):
        self._is_hovering = False
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_INPUT};
                border: 2px dashed #3A3A5A;
                border-radius: 12px;
            }}
        """)
        urls = event.mimeData().urls()
        files = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if files:
            self.files_dropped.emit(files)


class TextureInfoPanel(QFrame):
    """Panel showing metadata about the currently selected texture."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_MID};
                border: 1px solid #2A2A4A;
                border-radius: 10px;
                padding: 8px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        header = QLabel("Texture Info")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        self._fields = {}
        for label in ["File", "Dimensions", "Channels", "Format",
                       "File Size", "Power of 2", "Color Space"]:
            row = QHBoxLayout()
            key = QLabel(f"{label}:")
            key.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
            key.setFixedWidth(80)
            val = QLabel("—")
            val.setStyleSheet("font-weight: 500;")
            row.addWidget(key)
            row.addWidget(val)
            row.addStretch()
            layout.addLayout(row)
            self._fields[label] = val

        layout.addStretch()

    def update_info(self, info):
        """Update with a TextureInfo dataclass."""
        if info is None:
            for val in self._fields.values():
                val.setText("—")
            return
        self._fields["File"].setText(info.filename)
        self._fields["Dimensions"].setText(f"{info.width} × {info.height}")
        self._fields["Channels"].setText(
            f"{'RGBA' if info.has_alpha else 'RGB'} ({info.channels}ch)")
        self._fields["Format"].setText(info.format.value.upper())
        size_kb = info.file_size_bytes / 1024
        if size_kb > 1024:
            self._fields["File Size"].setText(f"{size_kb/1024:.1f} MB")
        else:
            self._fields["File Size"].setText(f"{size_kb:.0f} KB")
        pot = "✓  Yes" if info.is_power_of_two else "✗  No"
        self._fields["Power of 2"].setText(pot)
        self._fields["Color Space"].setText(info.color_space)


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Convert a numpy array (H, W, C) uint8 to QImage."""
    array = np.ascontiguousarray(array)
    if array.ndim == 2:
        h, w = array.shape
        return QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    h, w, c = array.shape
    if c == 4:
        bytes_per_line = 4 * w
        return QImage(array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()
    else:
        bytes_per_line = 3 * w
        return QImage(array.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def numpy_to_qpixmap(array: np.ndarray) -> QPixmap:
    """Convert numpy array to QPixmap."""
    return QPixmap.fromImage(numpy_to_qimage(array))
