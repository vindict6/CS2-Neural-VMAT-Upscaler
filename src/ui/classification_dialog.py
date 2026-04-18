"""
Pre-batch classification review dialog.

Shows every queued texture with its auto-detected material type and
confidence, lets the user override classifications, and previews
each texture before committing to the batch.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QImage
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox, QSplitter,
    QHeaderView, QProgressBar, QWidget, QSizePolicy, QAbstractItemView,
)

from ..core.material_enhancer import MaterialType, classify_material
from ..core.texture_io import load_texture
from .theme import (
    ACCENT, ACCENT_HOVER, ACCENT_PRESSED, BG_DARK, BG_MID, BG_LIGHT,
    BG_INPUT, TEXT_PRIMARY, TEXT_SECONDARY, BORDER, SURFACE, SUCCESS,
    WARNING,
)
from .widgets import numpy_to_qpixmap

logger = logging.getLogger("CS2Upscaler.ClassificationDialog")

# Human-friendly names for material types
_MATERIAL_LABELS = {
    "auto": "Auto-detect",
    MaterialType.METAL_BARE.value: "Metal (Bare)",
    MaterialType.METAL_PAINTED.value: "Metal (Painted)",
    MaterialType.METAL_RUSTED.value: "Metal (Rusted)",
    MaterialType.METAL_BRUSHED.value: "Metal (Brushed)",
    MaterialType.STONE_ROUGH.value: "Stone (Rough)",
    MaterialType.STONE_POLISHED.value: "Stone (Polished)",
    MaterialType.CONCRETE.value: "Concrete",
    MaterialType.BRICK.value: "Brick",
    MaterialType.TILE_CERAMIC.value: "Ceramic Tile",
    MaterialType.WOOD_RAW.value: "Wood (Raw)",
    MaterialType.WOOD_FINISHED.value: "Wood (Finished)",
    MaterialType.WOOD_WEATHERED.value: "Wood (Weathered)",
    MaterialType.FABRIC_WOVEN.value: "Fabric (Woven)",
    MaterialType.FABRIC_LEATHER.value: "Leather",
    MaterialType.CARPET.value: "Carpet",
    MaterialType.FOLIAGE.value: "Foliage",
    MaterialType.BARK.value: "Bark",
    MaterialType.SKIN.value: "Skin",
    MaterialType.PLASTIC.value: "Plastic",
    MaterialType.RUBBER.value: "Rubber",
    MaterialType.GLASS.value: "Glass",
    MaterialType.TERRAIN_DIRT.value: "Terrain (Dirt)",
    MaterialType.TERRAIN_SAND.value: "Terrain (Sand)",
    MaterialType.TERRAIN_GRAVEL.value: "Terrain (Gravel)",
    MaterialType.ASPHALT.value: "Asphalt",
    MaterialType.PLASTER.value: "Plaster",
    MaterialType.WALLPAPER.value: "Wallpaper",
    MaterialType.DECAL_TEXT.value: "Decal / Text",
    MaterialType.EMISSIVE_SCREEN.value: "Emissive / Screen",
    MaterialType.GENERIC.value: "Generic",
}

# Reverse: label -> value
_LABEL_TO_VALUE = {v: k for k, v in _MATERIAL_LABELS.items()}


class _ClassifyWorker(QThread):
    """Background worker that classifies all textures."""
    progress = pyqtSignal(int, str, str, float)  # index, filepath, material_value, confidence
    finished = pyqtSignal()

    def __init__(self, file_paths: List[str], vmat_overrides: Dict[str, str]):
        super().__init__()
        self._files = file_paths
        self._vmat_overrides = vmat_overrides

    def run(self):
        for i, fpath in enumerate(self._files):
            try:
                # Check for VMAT override first
                if fpath in self._vmat_overrides:
                    mat_value = self._vmat_overrides[fpath]
                    self.progress.emit(i, fpath, mat_value, 1.0)
                    continue

                img, _ = load_texture(fpath)
                mat_type, conf = classify_material(img[:, :, :3])
                self.progress.emit(i, fpath, mat_type.value, conf)
            except Exception as e:
                logger.warning(f"Failed to classify {fpath}: {e}")
                self.progress.emit(i, fpath, MaterialType.GENERIC.value, 0.0)
        self.finished.emit()


class ClassificationReviewDialog(QDialog):
    """
    Modal dialog shown before batch processing starts.

    Displays each texture with its auto-detected material type,
    allows the user to override, and shows a preview pane.
    """

    def __init__(self, file_paths: List[str],
                 vmat_overrides: Dict[str, str] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Material Classifications")
        self.setMinimumSize(1100, 650)
        self.resize(1200, 700)
        self.setModal(True)

        self._file_paths = file_paths
        self._vmat_overrides = vmat_overrides or {}
        self._classifications: Dict[str, Tuple[str, float]] = {}
        # {filepath: (material_value, confidence)}
        self._overrides: Dict[str, str] = {}
        # {filepath: material_value} — user overrides
        self._worker = None
        self._preview_cache: Dict[str, QPixmap] = {}

        self._build_ui()
        self._start_classification()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel(
            f"Review classifications for {len(self._file_paths)} texture(s) "
            f"before processing. Override any misdetections.")
        header.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px; padding-bottom: 4px;")
        header.setWordWrap(True)
        layout.addWidget(header)

        # Progress bar (shown during classification)
        self._progress = QProgressBar()
        self._progress.setMaximum(len(self._file_paths))
        self._progress.setTextVisible(True)
        self._progress.setFormat("Classifying... %v / %m")
        self._progress.setFixedHeight(22)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {BG_INPUT}; border: 1px solid {BORDER};
                border-radius: 4px; text-align: center;
                color: {TEXT_PRIMARY}; font-size: 11px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT}; border-radius: 3px;
            }}
        """)
        layout.addWidget(self._progress)

        # Splitter: table | preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── Left: classification table ───────────────────────────
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels([
            "Texture", "Detected", "Confidence", "Override",
        ])
        self._table.setRowCount(len(self._file_paths))
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        # Pre-populate rows with filenames
        for row, fpath in enumerate(self._file_paths):
            name_item = QTableWidgetItem(Path(fpath).name)
            name_item.setToolTip(fpath)
            self._table.setItem(row, 0, name_item)

            # Detected material (filled in by worker)
            det_item = QTableWidgetItem("Classifying...")
            det_item.setForeground(QColor(TEXT_SECONDARY))
            self._table.setItem(row, 1, det_item)

            # Confidence
            conf_item = QTableWidgetItem("")
            conf_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 2, conf_item)

            # Override combo
            combo = QComboBox()
            combo.setMinimumWidth(160)
            combo.addItem("Auto-detect")
            for mt in MaterialType:
                combo.addItem(_MATERIAL_LABELS.get(mt.value, mt.value))
            combo.setCurrentIndex(0)
            combo.currentIndexChanged.connect(
                lambda idx, r=row: self._on_override_changed(r, idx))
            self._table.setCellWidget(row, 3, combo)

        self._table.currentCellChanged.connect(self._on_row_selected)

        splitter.addWidget(self._table)

        # ── Right: preview pane ──────────────────────────────────
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        preview_layout.setSpacing(6)

        self._preview_label = QLabel("Select a texture to preview")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 12px;")
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview_label.setMinimumSize(300, 300)
        preview_layout.addWidget(self._preview_label, 1)

        self._preview_info = QLabel("")
        self._preview_info.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; padding: 4px;")
        self._preview_info.setWordWrap(True)
        preview_layout.addWidget(self._preview_info)

        splitter.addWidget(preview_container)
        splitter.setSizes([650, 450])
        layout.addWidget(splitter, 1)

        # ── Bottom buttons ───────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_auto_all = QPushButton("Reset All to Auto")
        self._btn_auto_all.setToolTip(
            "Reset all overrides back to auto-detection")
        self._btn_auto_all.setStyleSheet(f"""
            QPushButton {{
                background: {BG_LIGHT}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 4px;
                padding: 8px 16px; font-size: 12px;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.08); }}
        """)
        self._btn_auto_all.clicked.connect(self._reset_all_overrides)
        btn_row.addWidget(self._btn_auto_all)

        btn_row.addStretch()

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: {BG_LIGHT}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 4px;
                padding: 8px 20px; font-size: 12px;
            }}
            QPushButton:hover {{ background: rgba(255,255,255,0.08); }}
        """)
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)

        self._btn_start = QPushButton(
            f"Start Processing ({len(self._file_paths)} files)")
        self._btn_start.setEnabled(False)  # enabled after classification
        self._btn_start.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT}; color: white; border: none;
                border-radius: 4px; padding: 8px 24px;
                font-weight: 600; font-size: 12px;
            }}
            QPushButton:hover {{ background: {ACCENT_HOVER}; }}
            QPushButton:pressed {{ background: {ACCENT_PRESSED}; }}
            QPushButton:disabled {{ background: {BG_LIGHT}; color: {TEXT_SECONDARY}; }}
        """)
        self._btn_start.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_start)

        layout.addLayout(btn_row)

    def _start_classification(self):
        """Kick off background classification of all textures."""
        self._worker = _ClassifyWorker(self._file_paths, self._vmat_overrides)
        self._worker.progress.connect(self._on_classify_result)
        self._worker.finished.connect(self._on_classify_done)
        self._worker.start()

    def _on_classify_result(self, index: int, fpath: str,
                            mat_value: str, confidence: float):
        """Update table row with classification result."""
        self._classifications[fpath] = (mat_value, confidence)
        self._progress.setValue(index + 1)

        label = _MATERIAL_LABELS.get(mat_value, mat_value)
        det_item = self._table.item(index, 1)
        det_item.setText(label)
        det_item.setForeground(QColor(TEXT_PRIMARY))

        # Confidence with colour coding
        conf_item = self._table.item(index, 2)
        conf_pct = f"{confidence:.0%}"
        conf_item.setText(conf_pct)
        if confidence >= 0.7:
            conf_item.setForeground(QColor(SUCCESS))
        elif confidence >= 0.4:
            conf_item.setForeground(QColor(WARNING))
        else:
            conf_item.setForeground(QColor("#E04040"))

        # If VMAT override was used, show in the override combo
        if fpath in self._vmat_overrides:
            combo: QComboBox = self._table.cellWidget(index, 3)
            vmat_val = self._vmat_overrides[fpath]
            vmat_label = _MATERIAL_LABELS.get(vmat_val, vmat_val)
            for i in range(combo.count()):
                if combo.itemText(i) == vmat_label:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    self._overrides[fpath] = vmat_val
                    # Mark as VMAT override in detected column
                    det_item.setText(f"{label}  (VMAT)")
                    break

    def _on_classify_done(self):
        """Classification complete — enable Start button."""
        self._progress.hide()
        self._btn_start.setEnabled(True)
        self._btn_start.setFocus()

    def _on_row_selected(self, row: int, col: int, prev_row: int,
                         prev_col: int):
        """Show preview of the selected texture."""
        if row < 0 or row >= len(self._file_paths):
            return
        fpath = self._file_paths[row]

        # Check cache
        if fpath in self._preview_cache:
            pix = self._preview_cache[fpath]
        else:
            try:
                img, info = load_texture(fpath)
                pix = numpy_to_qpixmap(img[:, :, :3])
                self._preview_cache[fpath] = pix
            except Exception as e:
                self._preview_label.setText(f"Failed to load:\n{e}")
                self._preview_info.setText("")
                return

        # Scale to fit
        label_size = self._preview_label.size()
        scaled = pix.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)

        # Info text
        fname = Path(fpath).name
        cls_info = self._classifications.get(fpath)
        if cls_info:
            mat_val, conf = cls_info
            mat_label = _MATERIAL_LABELS.get(mat_val, mat_val)
            self._preview_info.setText(
                f"{fname}\n"
                f"{pix.width()}×{pix.height()}  ·  "
                f"Detected: {mat_label} ({conf:.0%})")
        else:
            self._preview_info.setText(fname)

    def _on_override_changed(self, row: int, combo_index: int):
        """User changed the override combo for a row."""
        fpath = self._file_paths[row]
        if combo_index == 0:
            # "Auto-detect" — remove override
            self._overrides.pop(fpath, None)
        else:
            # Map combo text to MaterialType value
            combo: QComboBox = self._table.cellWidget(row, 3)
            label = combo.currentText()
            value = _LABEL_TO_VALUE.get(label, "auto")
            if value != "auto":
                self._overrides[fpath] = value

    def _reset_all_overrides(self):
        """Reset all override combos to Auto-detect."""
        self._overrides.clear()
        for row in range(self._table.rowCount()):
            combo: QComboBox = self._table.cellWidget(row, 3)
            if combo:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)

    def get_overrides(self) -> Dict[str, str]:
        """
        Return the final material override map.

        Returns {filepath: material_value_str} for every file that
        has either a user override or a VMAT override.
        Files left as "Auto-detect" are not included.
        """
        return dict(self._overrides)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait(2000)
        super().closeEvent(event)
