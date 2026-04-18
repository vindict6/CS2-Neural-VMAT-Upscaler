"""
Preview widget with interactive before/after comparison slider,
zoom, pan, auto-fit, and multiple view modes.
"""

from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import (
    QPainter, QPixmap, QImage, QColor, QPen, QFont, QBrush,
    QWheelEvent, QMouseEvent, QPaintEvent,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QToolButton, QButtonGroup, QComboBox,
)

import numpy as np

from .theme import ACCENT, BG_DARK, BG_MID, BG_LIGHT, TEXT_SECONDARY, TEXT_PRIMARY, BORDER, SURFACE
from .widgets import numpy_to_qpixmap


class ViewMode:
    SLIDER = "slider"
    SIDE_BY_SIDE = "side_by_side"
    ORIGINAL_ONLY = "original_only"
    UPSCALED_ONLY = "upscaled_only"
    DIFFERENCE = "difference"


class FitMode:
    FIT_BOTH = "fit_both"
    FIT_WIDTH = "fit_width"
    FIT_HEIGHT = "fit_height"
    ACTUAL_SIZE = "actual_size"


class PreviewCanvas(QWidget):
    """
    Canvas widget that renders the before/after texture comparison
    with a draggable slider, zoom, and pan. Supports auto-fitting.
    """

    zoom_changed = pyqtSignal(float)
    pixel_info = pyqtSignal(int, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._original: QPixmap = None
        self._upscaled: QPixmap = None
        self._original_scaled: QPixmap = None
        self._original_array: np.ndarray = None
        self._upscaled_array: np.ndarray = None

        # View state
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._last_mouse = QPoint()
        self._panning = False

        # Comparison mode
        self._view_mode = ViewMode.SLIDER
        self._fit_mode = FitMode.FIT_BOTH
        self._slider_pos = 0.5
        self._dragging_slider = False

    @staticmethod
    def _pix_ok(pix: QPixmap) -> bool:
        """Return True only if *pix* is a usable (non-null) QPixmap."""
        return pix is not None and not pix.isNull()

    def set_original(self, image: np.ndarray):
        self._original_array = image
        self._original = numpy_to_qpixmap(image)
        self._auto_fit()
        self.update()
        # Deferred fit — widget geometry may not be final yet
        QTimer.singleShot(0, self._deferred_fit)

    def set_upscaled(self, image: np.ndarray):
        self._upscaled_array = image
        self._upscaled = numpy_to_qpixmap(image)
        if self._pix_ok(self._original) and self._pix_ok(self._upscaled):
            self._original_scaled = self._original.scaled(
                self._upscaled.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            self._original_scaled = None
        self._auto_fit()
        self.update()
        QTimer.singleShot(0, self._deferred_fit)

    def set_view_mode(self, mode: str):
        self._view_mode = mode
        self._auto_fit()
        self.update()

    def center_slider(self):
        """Reset the comparison slider to the center."""
        self._slider_pos = 0.5
        self.update()

    # Keep backward compat
    def set_mode(self, mode: str):
        mode_map = {
            "slider": ViewMode.SLIDER,
            "side_by_side": ViewMode.SIDE_BY_SIDE,
            "original": ViewMode.ORIGINAL_ONLY,
            "upscaled": ViewMode.UPSCALED_ONLY,
            "difference": ViewMode.DIFFERENCE,
        }
        self.set_view_mode(mode_map.get(mode, mode))

    def set_fit_mode(self, mode: str):
        self._fit_mode = mode
        self._auto_fit()
        self.update()

    def clear(self):
        self._original = None
        self._upscaled = None
        self._original_scaled = None
        self._original_array = None
        self._upscaled_array = None
        self.update()

    # ------------------------------------------------------------------
    # Fit / zoom
    # ------------------------------------------------------------------

    def _auto_fit(self):
        if self._fit_mode == FitMode.ACTUAL_SIZE:
            self._zoom_actual()
        elif self._fit_mode == FitMode.FIT_WIDTH:
            self._fit_width()
        elif self._fit_mode == FitMode.FIT_HEIGHT:
            self._fit_height()
        else:
            self._fit_to_view()

    def _fit_to_view(self):
        pix = self._get_active_pixmap()
        if pix is None:
            return
        vw, vh = self.width(), self.height()
        iw, ih = pix.width(), pix.height()
        if iw == 0 or ih == 0 or vw == 0 or vh == 0:
            return
        if self._view_mode == ViewMode.SIDE_BY_SIDE:
            scale_x = (vw / 2 - 8) / iw
            scale_y = vh / ih
        else:
            scale_x = vw / iw
            scale_y = vh / ih
        self._zoom = min(scale_x, scale_y) * 0.95
        self._center_image(pix)
        self.zoom_changed.emit(self._zoom)

    def _fit_width(self):
        pix = self._get_active_pixmap()
        if pix is None:
            return
        vw = self.width()
        iw = pix.width()
        if iw == 0:
            return
        if self._view_mode == ViewMode.SIDE_BY_SIDE:
            self._zoom = (vw / 2 - 8) / iw * 0.95
        else:
            self._zoom = vw / iw * 0.95
        self._center_image(pix)
        self.zoom_changed.emit(self._zoom)

    def _fit_height(self):
        pix = self._get_active_pixmap()
        if pix is None:
            return
        vh = self.height()
        ih = pix.height()
        if ih == 0:
            return
        self._zoom = vh / ih * 0.95
        self._center_image(pix)
        self.zoom_changed.emit(self._zoom)

    def _zoom_actual(self):
        pix = self._get_active_pixmap()
        if pix is None:
            return
        self._zoom = 1.0
        self._center_image(pix)
        self.zoom_changed.emit(self._zoom)

    def _center_image(self, pix: QPixmap):
        vw, vh = self.width(), self.height()
        if self._view_mode == ViewMode.SIDE_BY_SIDE:
            iw = pix.width() * self._zoom
            ih = pix.height() * self._zoom
            self._pan = QPointF((vw / 2 - iw) / 2, (vh - ih) / 2)
        else:
            self._pan = QPointF(
                (vw - pix.width() * self._zoom) / 2,
                (vh - pix.height() * self._zoom) / 2,
            )

    def _deferred_fit(self):
        """Called after event loop — widget geometry is now valid."""
        if self._pix_ok(self._original) or self._pix_ok(self._upscaled):
            self._auto_fit()
            self.update()

    def _get_active_pixmap(self) -> QPixmap:
        if self._view_mode == ViewMode.ORIGINAL_ONLY:
            return self._original if self._pix_ok(self._original) else None
        elif self._view_mode == ViewMode.UPSCALED_ONLY:
            return self._upscaled if self._pix_ok(self._upscaled) else None
        if self._pix_ok(self._upscaled):
            return self._upscaled
        if self._pix_ok(self._original):
            return self._original
        return None

    def fit_to_view(self):
        self._fit_mode = FitMode.FIT_BOTH
        self._fit_to_view()
        self.update()

    def zoom_to(self, factor: float):
        self._zoom = max(0.01, min(50.0, factor))
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self):
        self.zoom_to(self._zoom * 1.25)

    def zoom_out(self):
        self.zoom_to(self._zoom / 1.25)

    def zoom_100(self):
        pix = self._get_active_pixmap()
        if pix:
            self._zoom = 1.0
            self._center_image(pix)
            self.zoom_changed.emit(self._zoom)
            self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)
        painter.fillRect(self.rect(), QColor(BG_DARK))

        has_orig = self._pix_ok(self._original)
        has_up = self._pix_ok(self._upscaled)

        if not has_orig and not has_up:
            self._draw_placeholder(painter)
            painter.end()
            return

        ref = self._get_active_pixmap()
        if ref is None:
            self._draw_placeholder(painter)
            painter.end()
            return

        dest = QRectF(
            self._pan.x(), self._pan.y(),
            ref.width() * self._zoom, ref.height() * self._zoom,
        )

        if self._view_mode == ViewMode.ORIGINAL_ONLY:
            if has_orig:
                painter.drawPixmap(dest.toRect(), self._original)
                self._draw_label(painter, dest, "Original", self._original_array)
        elif self._view_mode == ViewMode.UPSCALED_ONLY:
            if has_up:
                painter.drawPixmap(dest.toRect(), self._upscaled)
                self._draw_label(painter, dest, "Upscaled", self._upscaled_array)
        elif self._view_mode == ViewMode.SLIDER:
            if has_orig and has_up:
                self._draw_slider_comparison(painter, self._original, self._upscaled, dest)
            elif has_orig:
                painter.drawPixmap(dest.toRect(), self._original)
                self._draw_label(painter, dest, "Original", self._original_array)
            elif has_up:
                painter.drawPixmap(dest.toRect(), self._upscaled)
                self._draw_label(painter, dest, "Upscaled", self._upscaled_array)
        elif self._view_mode == ViewMode.SIDE_BY_SIDE:
            if has_orig and has_up:
                self._draw_side_by_side(painter, self._original, self._upscaled)
            elif has_orig:
                painter.drawPixmap(dest.toRect(), self._original)
                self._draw_label(painter, dest, "Original", self._original_array)
            elif has_up:
                painter.drawPixmap(dest.toRect(), self._upscaled)
                self._draw_label(painter, dest, "Upscaled", self._upscaled_array)
        elif self._view_mode == ViewMode.DIFFERENCE:
            if has_orig and has_up:
                self._draw_difference(painter, dest)
            elif has_orig:
                painter.drawPixmap(dest.toRect(), self._original)
                self._draw_label(painter, dest, "Original", self._original_array)
            elif has_up:
                painter.drawPixmap(dest.toRect(), self._upscaled)
                self._draw_label(painter, dest, "Upscaled", self._upscaled_array)

        painter.end()

    def _draw_placeholder(self, painter: QPainter):
        painter.setPen(QColor(TEXT_SECONDARY))
        font = QFont("Segoe UI", 13)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                         "Load a texture to preview")

    def _draw_floating_label(self, painter: QPainter, x: int, y: int,
                             text: str, dims_array=None):
        """Draw a floating pill label at the given position."""
        dims = ""
        if dims_array is not None:
            h, w = dims_array.shape[:2]
            dims = f"  {w}\u00d7{h}"
        full_text = text + dims
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(full_text) + 16
        th = fm.height() + 8
        pill = QRectF(x, y, tw, th)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRoundedRect(pill, 4, 4)
        painter.setPen(QColor("white"))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, full_text)

    def _draw_label(self, painter: QPainter, dest: QRectF, text: str,
                    array=None):
        self._draw_floating_label(
            painter, int(dest.x()) + 8, int(dest.y()) + 6, text, array)

    def _draw_slider_comparison(self, painter: QPainter, orig: QPixmap,
                                upscaled: QPixmap, dest: QRectF):
        split_x = int(dest.x() + dest.width() * self._slider_pos)

        display_orig = self._original_scaled if self._pix_ok(self._original_scaled) else orig

        # Right: upscaled
        if self._pix_ok(upscaled):
            painter.setClipRect(QRectF(split_x, 0, self.width() - split_x, self.height()))
            painter.drawPixmap(dest.toRect(), upscaled)
            painter.setClipping(False)

        # Left: original
        if self._pix_ok(display_orig):
            painter.setClipRect(QRectF(0, 0, split_x, self.height()))
            painter.drawPixmap(dest.toRect(), display_orig)
            painter.setClipping(False)

        # Slider line
        pen = QPen(QColor(ACCENT), 2)
        painter.setPen(pen)
        painter.drawLine(split_x, int(dest.y()), split_x, int(dest.y() + dest.height()))

        # Slider handle
        handle_y = int(dest.y() + dest.height() / 2)
        handle_rect = QRectF(split_x - 14, handle_y - 14, 28, 28)
        painter.setPen(QPen(QColor("white"), 2))
        painter.setBrush(QColor(ACCENT))
        painter.drawRoundedRect(handle_rect, 14, 14)

        # Arrows
        cx = split_x
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawLine(cx - 6, handle_y, cx - 2, handle_y - 4)
        painter.drawLine(cx - 6, handle_y, cx - 2, handle_y + 4)
        painter.drawLine(cx + 6, handle_y, cx + 2, handle_y - 4)
        painter.drawLine(cx + 6, handle_y, cx + 2, handle_y + 4)

        # Labels
        self._draw_floating_label(
            painter, int(dest.x()) + 8, int(dest.y()) + 6,
            "Original", self._original_array)
        if self._pix_ok(upscaled):
            self._draw_floating_label(
                painter, split_x + 8, int(dest.y()) + 6,
                "Upscaled", self._upscaled_array)

    def _draw_side_by_side(self, painter: QPainter, orig: QPixmap, upscaled: QPixmap):
        vw, vh = self.width(), self.height()
        mid = vw // 2
        gap = 4

        display_orig = self._original_scaled if self._pix_ok(self._original_scaled) else orig
        ref = upscaled if self._pix_ok(upscaled) else display_orig
        if not self._pix_ok(ref):
            return

        half_w = mid - gap - 8
        scale_x = half_w / ref.width()
        scale_y = (vh - 16) / ref.height()
        half_zoom = min(scale_x, scale_y)
        iw = ref.width() * half_zoom
        ih = ref.height() * half_zoom

        if self._pix_ok(display_orig):
            lx = (mid - gap - iw) / 2
            ly = (vh - ih) / 2
            dest_l = QRectF(lx, ly, iw, ih)
            painter.drawPixmap(dest_l.toRect(), display_orig)
            self._draw_floating_label(
                painter, int(lx) + 8, int(ly) + 6,
                "Original", self._original_array)

        if self._pix_ok(upscaled):
            rx = mid + gap + (mid - gap - iw) / 2
            ry = (vh - ih) / 2
            dest_r = QRectF(rx, ry, iw, ih)
            painter.drawPixmap(dest_r.toRect(), upscaled)
            self._draw_floating_label(
                painter, int(rx) + 8, int(ry) + 6,
                "Upscaled", self._upscaled_array)

        # Divider
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawLine(mid, 0, mid, vh)

    def _draw_difference(self, painter: QPainter, dest: QRectF):
        if self._original_array is None or self._upscaled_array is None:
            return
        from PIL import Image as PILImage
        orig_resized = np.array(PILImage.fromarray(self._original_array).resize(
            (self._upscaled_array.shape[1], self._upscaled_array.shape[0]),
            PILImage.LANCZOS,
        ))
        c = min(orig_resized.shape[2], self._upscaled_array.shape[2])
        diff = np.abs(
            orig_resized[:, :, :c].astype(np.int16) -
            self._upscaled_array[:, :, :c].astype(np.int16)
        ).astype(np.uint8)
        diff = np.clip(diff * 4, 0, 255).astype(np.uint8)
        pix = numpy_to_qpixmap(diff)
        painter.drawPixmap(dest.toRect(), pix)
        self._draw_floating_label(
            painter, int(dest.x()) + 8, int(dest.y()) + 6,
            "Difference (4\u00d7)")

    # ------------------------------------------------------------------
    # Input events
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15

        mouse = event.position()
        old_scene = (mouse - self._pan) / self._zoom
        self._zoom = max(0.01, min(50.0, self._zoom * factor))
        new_screen = old_scene * self._zoom + self._pan
        self._pan += (mouse - new_screen)

        self.zoom_changed.emit(self._zoom)
        self.update()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_mouse = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._view_mode == ViewMode.SLIDER:
                self._dragging_slider = True
                self._update_slider(event.pos())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.pos() - self._last_mouse
            self._pan += QPointF(delta.x(), delta.y())
            self._last_mouse = event.pos()
            self.update()
        elif self._dragging_slider:
            self._update_slider(event.pos())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            self._dragging_slider = False

    def _update_slider(self, pos):
        ref = self._get_active_pixmap()
        if ref is None:
            return
        dest_w = ref.width() * self._zoom
        rel = (pos.x() - self._pan.x()) / dest_w
        self._slider_pos = max(0.0, min(1.0, rel))
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pix_ok(self._original) or self._pix_ok(self._upscaled):
            self._auto_fit()


class PreviewWidget(QFrame):
    """
    Complete preview panel with the canvas, view mode selector,
    fit mode selector, zoom controls, and info bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {SURFACE};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 3, 8, 3)
        tb_layout.setSpacing(4)

        # View mode buttons
        view_label = QLabel("View:")
        view_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; border: none;")
        tb_layout.addWidget(view_label)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        modes = [
            ("\u21c6 Slider", ViewMode.SLIDER),
            ("\u25eb Side by Side", ViewMode.SIDE_BY_SIDE),
            ("\u25fb Original", ViewMode.ORIGINAL_ONLY),
            ("\u25fc Upscaled", ViewMode.UPSCALED_ONLY),
            ("\u25b3 Diff", ViewMode.DIFFERENCE),
        ]
        for i, (label, mode) in enumerate(modes):
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.setMinimumHeight(24)
            btn.setStyleSheet(f"""
                QToolButton {{
                    padding: 2px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                    border: none;
                    background: transparent;
                    color: {TEXT_SECONDARY};
                }}
                QToolButton:hover {{
                    background: {BG_LIGHT};
                    color: {TEXT_PRIMARY};
                }}
                QToolButton:checked {{
                    background: {ACCENT};
                    color: white;
                }}
            """)
            btn.clicked.connect(lambda checked, m=mode: self._on_view_mode(m))
            self._mode_group.addButton(btn, i)
            tb_layout.addWidget(btn)

        # Center divider button (visible in slider mode)
        self._center_btn = QToolButton()
        self._center_btn.setText("\u25c8 Center")
        self._center_btn.setMinimumHeight(24)
        self._center_btn.setToolTip("Center the comparison divider")
        self._center_btn.setStyleSheet(f"""
            QToolButton {{
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
                border: none;
                background: transparent;
                color: {TEXT_SECONDARY};
            }}
            QToolButton:hover {{
                background: {BG_LIGHT};
                color: {TEXT_PRIMARY};
            }}
        """)
        self._center_btn.clicked.connect(lambda: self._canvas.center_slider())
        tb_layout.addWidget(self._center_btn)

        tb_layout.addSpacing(12)

        # Fit mode
        fit_label = QLabel("Fit:")
        fit_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; border: none;")
        tb_layout.addWidget(fit_label)

        self._fit_combo = QComboBox()
        self._fit_combo.addItems(["Fit All", "Fit Width", "Fit Height", "1:1 Actual"])
        self._fit_combo.setFixedWidth(100)
        self._fit_combo.setStyleSheet("font-size: 11px;")
        fit_modes = [FitMode.FIT_BOTH, FitMode.FIT_WIDTH, FitMode.FIT_HEIGHT, FitMode.ACTUAL_SIZE]
        self._fit_combo.currentIndexChanged.connect(
            lambda idx: self._canvas.set_fit_mode(fit_modes[idx]))
        tb_layout.addWidget(self._fit_combo)

        tb_layout.addStretch()

        # Zoom controls
        self._zoom_label = QLabel("100%")
        self._zoom_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; border: none;")
        self._zoom_label.setMinimumWidth(44)

        for text, slot in [("\u2212", lambda: self._canvas.zoom_out()),
                           ("+", lambda: self._canvas.zoom_in()),
                           ("Fit", lambda: self._canvas.fit_to_view()),
                           ("1:1", lambda: self._canvas.zoom_100())]:
            btn = QToolButton()
            btn.setText(text)
            btn.setFixedSize(28, 24) if len(text) <= 1 else btn.setMinimumHeight(24)
            btn.setStyleSheet(f"padding: 2px 6px; border-radius: 3px; border: none; font-size: 11px;")
            btn.clicked.connect(slot)
            tb_layout.addWidget(btn)

        tb_layout.addWidget(self._zoom_label)

        layout.addWidget(toolbar)

        # Canvas
        self._canvas = PreviewCanvas()
        self._canvas.zoom_changed.connect(
            lambda z: self._zoom_label.setText(f"{z*100:.0f}%"))
        layout.addWidget(self._canvas, 1)

    def _on_view_mode(self, mode: str):
        """Switch view mode and update center button visibility."""
        self._canvas.set_view_mode(mode)
        self._center_btn.setVisible(mode == ViewMode.SLIDER)

    @property
    def canvas(self) -> PreviewCanvas:
        return self._canvas

    def set_original(self, image: np.ndarray):
        self._canvas.set_original(image)

    def set_upscaled(self, image: np.ndarray):
        self._canvas.set_upscaled(image)

    def clear(self):
        self._canvas.clear()
