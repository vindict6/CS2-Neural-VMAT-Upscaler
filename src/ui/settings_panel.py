"""
Settings panel – model selection, upscale parameters, texture options,
output configuration, and advanced settings.
"""

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QGroupBox, QFormLayout, QFrame, QScrollArea, QSizePolicy,
    QProgressBar, QMessageBox, QLineEdit,
)

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ..core.upscaler import (
    UpscaleSettings, ModelVariant, TextureType,
)
from ..core.models import ModelManager, MODEL_REGISTRY
from .theme import ACCENT, TEXT_SECONDARY

logger = logging.getLogger("CS2Upscaler.SettingsPanel")

_NUM_PRESET_SLOTS = 32
_PRESETS_FILE = Path.home() / ".textureforge" / "presets.json"


# Model download URLs (same as upscaler.py but accessible without loading torch)
_MODEL_URLS = {
    "RealESRGAN_x4plus":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRNet_x4plus":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
}


class _DownloadWorker(QThread):
    """Downloads a model file in a background thread with progress."""

    progress = pyqtSignal(int)  # 0-100
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, url: str, dest_path: str, parent=None):
        super().__init__(parent)
        self._url = url
        self._dest = dest_path

    def run(self):
        import urllib.request
        import os
        try:
            dest_dir = os.path.dirname(self._dest)
            os.makedirs(dest_dir, exist_ok=True)

            tmp_path = self._dest + ".tmp"
            req = urllib.request.urlopen(self._url)
            total = int(req.headers.get("Content-Length", 0))
            downloaded = 0
            block = 1024 * 256  # 256 KB chunks

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = req.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        self.progress.emit(int(downloaded * 100 / total))

            os.replace(tmp_path, self._dest)
            self.finished.emit(True, "Download complete")
        except Exception as e:
            self.finished.emit(False, str(e))


class SettingsPanel(QScrollArea):
    """
    Scrollable panel with all upscaling parameters organised
    into collapsible groups.
    """

    settings_changed = pyqtSignal()

    def __init__(self, model_manager: ModelManager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(300)
        self.setMaximumWidth(420)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(8, 8, 8, 8)

        self._preset_slots: dict = {}  # {slot_number: {"name": str, "settings": dict}}
        self._load_preset_slots()

        self._build_presets_group()
        self._build_model_group()
        self._build_upscale_group()
        self._build_enhance_group()
        self._build_pbr_group()
        self._build_output_group()
        self._build_advanced_group()

        self._layout.addStretch()
        self.setWidget(container)
        self._refresh_slot_combo()

    # ------------------------------------------------------------------
    # Preset slots (32)
    # ------------------------------------------------------------------

    def _build_presets_group(self):
        from .theme import (
            ACCENT, ACCENT_HOVER, ACCENT_PRESSED,
            BG_LIGHT, BG_INPUT, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
            SUCCESS, WARNING,
        )
        group = QGroupBox("Presets")
        vbox = QVBoxLayout(group)
        vbox.setSpacing(6)

        # Slot selector
        self._slot_combo = QComboBox()
        self._slot_combo.setMaxVisibleItems(16)
        self._slot_combo.currentIndexChanged.connect(self._on_slot_changed)
        vbox.addWidget(self._slot_combo)

        # Slot name editor
        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        name_lbl = QLabel("Name:")
        name_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        name_row.addWidget(name_lbl)
        self._slot_name_edit = QLineEdit()
        self._slot_name_edit.setPlaceholderText("Untitled preset")
        self._slot_name_edit.setMaxLength(40)
        name_row.addWidget(self._slot_name_edit)
        vbox.addLayout(name_row)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._slot_save_btn = QPushButton("Save")
        self._slot_save_btn.setToolTip("Save current settings to this slot")
        self._slot_save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT};
                color: white; border: none; border-radius: 3px;
                padding: 4px 12px; font-weight: 600; font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_HOVER}; }}
            QPushButton:pressed {{ background-color: {ACCENT_PRESSED}; }}
        """)
        self._slot_save_btn.clicked.connect(self._on_slot_save)
        btn_row.addWidget(self._slot_save_btn)

        self._slot_load_btn = QPushButton("Load")
        self._slot_load_btn.setToolTip("Load settings from this slot")
        self._slot_load_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_LIGHT};
                color: {TEXT_PRIMARY}; border: 1px solid {BORDER};
                border-radius: 3px; padding: 4px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: rgba(255,255,255,0.08); }}
        """)
        self._slot_load_btn.clicked.connect(self._on_slot_load)
        btn_row.addWidget(self._slot_load_btn)

        self._slot_clear_btn = QPushButton("Clear")
        self._slot_clear_btn.setToolTip("Clear this slot")
        self._slot_clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_LIGHT};
                color: {TEXT_SECONDARY}; border: 1px solid {BORDER};
                border-radius: 3px; padding: 4px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{ color: {WARNING}; }}
        """)
        self._slot_clear_btn.clicked.connect(self._on_slot_clear)
        btn_row.addWidget(self._slot_clear_btn)

        vbox.addLayout(btn_row)

        # Slot info label
        self._slot_info = QLabel("")
        self._slot_info.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px; padding-top: 2px;")
        self._slot_info.setWordWrap(True)
        vbox.addWidget(self._slot_info)

        self._layout.addWidget(group)

    def _refresh_slot_combo(self):
        """Rebuild the slot combo items with names."""
        self._slot_combo.blockSignals(True)
        idx = max(self._slot_combo.currentIndex(), 0)
        self._slot_combo.clear()
        for i in range(1, _NUM_PRESET_SLOTS + 1):
            slot = self._preset_slots.get(str(i))
            if slot:
                name = slot.get("name", "").strip() or "Untitled"
                self._slot_combo.addItem(f"Slot {i}: {name}")
            else:
                self._slot_combo.addItem(f"Slot {i}: (empty)")
        if idx < self._slot_combo.count():
            self._slot_combo.setCurrentIndex(idx)
        self._slot_combo.blockSignals(False)
        self._on_slot_changed(self._slot_combo.currentIndex())

    def _on_slot_changed(self, index: int):
        """Update the name editor and buttons when slot selection changes."""
        slot_num = str(index + 1)
        slot = self._preset_slots.get(slot_num)
        occupied = slot is not None
        self._slot_load_btn.setEnabled(occupied)
        self._slot_clear_btn.setEnabled(occupied)
        if occupied:
            self._slot_name_edit.setText(slot.get("name", ""))
            self._slot_info.setText(self._describe_slot(slot))
        else:
            self._slot_name_edit.setText("")
            self._slot_info.setText("Empty slot — save current settings here.")

    def _describe_slot(self, slot: dict) -> str:
        """Build a short description of a saved slot."""
        s = slot.get("settings", {})
        parts = []
        parts.append(f"{s.get('scale_factor', 4)}× {s.get('model_variant', '?')}")
        if s.get("generative_enhance"):
            parts.append("Enhance ON")
        if s.get("pbr_generate"):
            parts.append("PBR ON")
        fmt = s.get("output_format", "png").upper()
        parts.append(fmt)
        return "  ·  ".join(parts)

    def _on_slot_save(self):
        """Save current settings to the selected slot."""
        slot_num = str(self._slot_combo.currentIndex() + 1)
        name = self._slot_name_edit.text().strip() or "Untitled"
        settings = self.get_settings()
        data = asdict(settings)
        # Convert enums to their string values for JSON serialization
        data["model_variant"] = settings.model_variant.value
        data["texture_type"] = settings.texture_type.value
        self._preset_slots[slot_num] = {"name": name, "settings": data}
        self._save_preset_slots()
        self._refresh_slot_combo()
        self._slot_info.setText(f"Saved to slot {slot_num}.")
        logger.info(f"Preset saved: slot {slot_num} ({name})")

    def _on_slot_load(self):
        """Load settings from the selected slot into the panel."""
        slot_num = str(self._slot_combo.currentIndex() + 1)
        slot = self._preset_slots.get(slot_num)
        if not slot:
            return
        try:
            data = slot["settings"]
            # Reconstruct enum fields
            data["model_variant"] = ModelVariant(data.get("model_variant",
                                                          ModelVariant.REALESRGAN_X4PLUS.value))
            data["texture_type"] = TextureType(data.get("texture_type",
                                                        TextureType.AUTO.value))
            settings = UpscaleSettings(**{k: v for k, v in data.items()
                                          if k in UpscaleSettings.__dataclass_fields__})
            self.apply_settings(settings)
            self._slot_info.setText(f"Loaded slot {slot_num}.")
            logger.info(f"Preset loaded: slot {slot_num}")
        except Exception as e:
            logger.warning(f"Failed to load slot {slot_num}: {e}")
            self._slot_info.setText(f"Load failed: {e}")

    def _on_slot_clear(self):
        """Clear the selected slot."""
        slot_num = str(self._slot_combo.currentIndex() + 1)
        if slot_num in self._preset_slots:
            del self._preset_slots[slot_num]
            self._save_preset_slots()
            self._refresh_slot_combo()
            self._slot_info.setText(f"Slot {slot_num} cleared.")
            logger.info(f"Preset cleared: slot {slot_num}")

    def _load_preset_slots(self):
        """Load preset slots from disk."""
        self._preset_slots = {}
        if _PRESETS_FILE.exists():
            try:
                with open(_PRESETS_FILE, "r", encoding="utf-8") as f:
                    self._preset_slots = json.load(f)
                logger.info(f"Loaded {len(self._preset_slots)} preset slots")
            except Exception as e:
                logger.warning(f"Failed to load presets: {e}")

    def _save_preset_slots(self):
        """Persist preset slots to disk."""
        _PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._preset_slots, f, indent=2, default=str)
        logger.debug(f"Presets saved to {_PRESETS_FILE}")

    def apply_settings(self, s: UpscaleSettings):
        """Restore all panel widgets from an UpscaleSettings object."""
        # Block signals during bulk update to avoid repeated settings_changed
        self.blockSignals(True)
        try:
            # Model
            for i in range(self._model_combo.count()):
                if self._model_combo.itemData(i) == s.model_variant.value:
                    self._model_combo.setCurrentIndex(i)
                    break

            # Scale
            self._scale_combo.setCurrentIndex(0 if s.scale_factor == 2 else 1)

            # Texture type
            for i in range(self._textype_combo.count()):
                if self._textype_combo.itemData(i) == s.texture_type.value:
                    self._textype_combo.setCurrentIndex(i)
                    break

            # Max output size
            size_str = str(s.max_output_size) if s.max_output_size > 0 else "Unlimited"
            idx = self._max_size_combo.findText(size_str)
            if idx >= 0:
                self._max_size_combo.setCurrentIndex(idx)

            # Checkboxes
            self._seamless.setChecked(s.seamless_mode)
            self._preserve_alpha.setChecked(s.preserve_alpha)

            # Generative enhance
            self._enhance_check.setChecked(s.generative_enhance)
            self._enhance_strength.setValue(int(s.enhance_strength * 100))
            self._photorealistic_check.setChecked(s.photorealistic_mode)
            self._anti_halo_check.setChecked(s.anti_halo)

            # Material override
            for i in range(self._material_override.count()):
                if self._material_override.itemData(i) == s.material_override:
                    self._material_override.setCurrentIndex(i)
                    break

            # Stage toggles
            self._stage_clahe.setChecked(s.enhance_clahe)
            self._stage_freq_split.setChecked(s.enhance_frequency_split)
            self._stage_detail_synth.setChecked(s.enhance_detail_synth)
            self._stage_micro_overlay.setChecked(s.enhance_micro_overlay)
            self._stage_wavelet.setChecked(s.enhance_wavelet)
            self._stage_sharpen.setChecked(s.enhance_sharpen)
            self._stage_structure.setChecked(s.enhance_structure)
            self._stage_colour.setChecked(s.enhance_colour)

            # PBR
            self._pbr_enable.setChecked(s.pbr_generate)
            self._pbr_roughness.setChecked(s.pbr_roughness)
            self._pbr_metalness.setChecked(s.pbr_metalness)
            self._pbr_normals.setChecked(s.pbr_normals)
            self._pbr_strength.setValue(int(s.pbr_strength * 100))
            self._pbr_update_vmat.setChecked(s.pbr_update_vmat)
            self._pbr_auto_values.setChecked(s.pbr_auto_values)

            # Output
            fmt_rev = {"png": "PNG", "jpeg": "JPEG", "tga": "TGA", "webp": "WebP"}
            fmt_idx = self._format_combo.findText(
                fmt_rev.get(s.output_format, "PNG"))
            if fmt_idx >= 0:
                self._format_combo.setCurrentIndex(fmt_idx)
            self._compression_check.setChecked(s.compression_enabled)
            self._compress_quality.setValue(s.compression_quality)

            # Advanced
            tile_str = "Auto" if s.tile_size == 0 else str(s.tile_size)
            tidx = self._tile_size.findText(tile_str)
            if tidx >= 0:
                self._tile_size.setCurrentIndex(tidx)
            self._tile_overlap.setValue(s.tile_overlap)
            self._half_precision.setChecked(s.half_precision)
            self._gpu_id.setValue(s.gpu_id)
        finally:
            self.blockSignals(False)
        self.settings_changed.emit()

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _build_model_group(self):
        group = QGroupBox("AI Model")
        form = QFormLayout(group)

        self._model_combo = QComboBox()
        for info in MODEL_REGISTRY:
            status = " ✓" if info.downloaded else ""
            self._model_combo.addItem(
                f"{info.name}{status}", info.variant.value)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        form.addRow("Model:", self._model_combo)

        self._model_desc = QLabel("")
        self._model_desc.setWordWrap(True)
        self._model_desc.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        form.addRow(self._model_desc)

        self._download_btn = QPushButton("Download Model")
        self._download_btn.setObjectName("primaryButton")
        self._download_btn.setVisible(False)
        self._download_btn.clicked.connect(self._on_download_model)
        form.addRow(self._download_btn)

        self._download_progress = QProgressBar()
        self._download_progress.setRange(0, 100)
        self._download_progress.setVisible(False)
        form.addRow(self._download_progress)

        self._download_worker = None

        self._on_model_changed()
        self._layout.addWidget(group)

    def _on_model_changed(self):
        idx = self._model_combo.currentIndex()
        if 0 <= idx < len(MODEL_REGISTRY):
            info = MODEL_REGISTRY[idx]
            self._model_desc.setText(
                f"{info.description}\n"
                f"Recommended for: {info.recommended_for}\n"
                f"Size: {info.file_size_mb:.0f} MB"
            )
            self._download_btn.setVisible(not info.downloaded)
        self.settings_changed.emit()

    def _on_download_model(self):
        """Start downloading the selected model in a background thread."""
        idx = self._model_combo.currentIndex()
        if idx < 0 or idx >= len(MODEL_REGISTRY):
            return

        info = MODEL_REGISTRY[idx]
        url = _MODEL_URLS.get(info.variant.value)
        if url is None:
            QMessageBox.warning(self, "Download Error",
                                f"No download URL for {info.name}")
            return

        dest = str(self.model_manager.models_dir / info.filename)

        self._download_btn.setEnabled(False)
        self._download_btn.setText("Downloading...")
        self._download_progress.setValue(0)
        self._download_progress.setVisible(True)

        self._download_worker = _DownloadWorker(url, dest, self)
        self._download_worker.progress.connect(self._download_progress.setValue)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.start()

    def _on_download_finished(self, success: bool, message: str):
        """Handle download completion."""
        self._download_progress.setVisible(False)
        self._download_btn.setEnabled(True)
        self._download_btn.setText("Download Model")

        if success:
            # Refresh model status and update combo box text
            self.model_manager._refresh_download_status()
            idx = self._model_combo.currentIndex()
            if 0 <= idx < len(MODEL_REGISTRY):
                info = MODEL_REGISTRY[idx]
                self._model_combo.setItemText(idx, f"{info.name} ✓")
                self._download_btn.setVisible(False)
        else:
            QMessageBox.warning(self, "Download Failed", message)

    # ------------------------------------------------------------------
    # Upscale settings (combined: scale + texture + processing)
    # ------------------------------------------------------------------

    def _build_upscale_group(self):
        group = QGroupBox("Upscale")
        form = QFormLayout(group)
        form.setSpacing(6)

        # Scale factor
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(["2×", "4×"])
        self._scale_combo.setCurrentIndex(1)
        self._scale_combo.currentIndexChanged.connect(self.settings_changed.emit)
        form.addRow("Scale:", self._scale_combo)

        # Texture type
        self._textype_combo = QComboBox()
        for tt in TextureType:
            self._textype_combo.addItem(tt.value.capitalize(), tt.value)
        self._textype_combo.currentIndexChanged.connect(self.settings_changed.emit)
        form.addRow("Type:", self._textype_combo)

        # Max output resolution
        self._max_size_combo = QComboBox()
        self._max_size_combo.addItems(["1024", "2048", "4096", "8192", "Unlimited"])
        self._max_size_combo.setCurrentIndex(2)  # default 4096
        self._max_size_combo.setToolTip("Clamp longest edge of output. Prevents huge files.")
        self._max_size_combo.currentIndexChanged.connect(self.settings_changed.emit)
        form.addRow("Max Size:", self._max_size_combo)

        # Seamless tiling
        self._seamless = QCheckBox("Seamless / Tileable")
        self._seamless.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._seamless)

        self._preserve_alpha = QCheckBox("Preserve Alpha")
        self._preserve_alpha.setChecked(True)
        self._preserve_alpha.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._preserve_alpha)

        self._layout.addWidget(group)

    # ------------------------------------------------------------------
    # Generative Enhance
    # ------------------------------------------------------------------

    def _build_enhance_group(self):
        from .theme import ACCENT
        group = QGroupBox("Generative Enhance")
        form = QFormLayout(group)
        form.setSpacing(6)

        self._enhance_check = QCheckBox("Enable Generative Enhance")
        self._enhance_check.setToolTip(
            "Analyses the material type (metal, wood, stone, fabric, etc.)\n"
            "and applies material-aware enhancement that dramatically\n"
            "improves surface detail, contrast, and colour beyond a\n"
            "simple AI upscale.")
        self._enhance_check.stateChanged.connect(self._on_enhance_toggled)
        self._enhance_check.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._enhance_check)

        self._enhance_desc = QLabel(
            "Detects material type (25+ categories) and applies\n"
            "multi-stage enhancement with text protection and\n"
            "anti-halo correction.")
        self._enhance_desc.setWordWrap(True)
        self._enhance_desc.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; padding: 2px 0;")
        form.addRow(self._enhance_desc)

        # Strength slider
        self._enhance_strength = QSlider(Qt.Orientation.Horizontal)
        self._enhance_strength.setRange(0, 200)
        self._enhance_strength.setValue(100)
        self._enhance_strength.setEnabled(False)
        self._enhance_strength_label = QLabel("1.0")
        self._enhance_strength.valueChanged.connect(
            lambda v: self._enhance_strength_label.setText(f"{v / 100:.1f}"))
        self._enhance_strength.valueChanged.connect(self.settings_changed.emit)
        row = QHBoxLayout()
        row.addWidget(self._enhance_strength)
        row.addWidget(self._enhance_strength_label)
        form.addRow("Strength:", row)

        strength_note = QLabel(
            "0.5 = subtle  ·  1.0 = balanced  ·  1.5+ = aggressive")
        strength_note.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px;")
        form.addRow(strength_note)

        # Anti-halo checkbox
        self._anti_halo_check = QCheckBox("Anti-Halo Correction")
        self._anti_halo_check.setChecked(True)
        self._anti_halo_check.setEnabled(False)
        self._anti_halo_check.setToolTip(
            "Compare upscaled output to the original texture to detect\n"
            "and suppress halo artifacts (bright/dark rings around edges)\n"
            "commonly introduced by AI upscalers.")
        self._anti_halo_check.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._anti_halo_check)

        # Photorealistic mode checkbox
        self._photorealistic_check = QCheckBox("Photorealistic Conversion")
        self._photorealistic_check.setChecked(False)
        self._photorealistic_check.setEnabled(False)
        self._photorealistic_check.setToolTip(
            "Convert game-looking textures toward photorealistic appearance.\n"
            "De-bands hand-painted colours, applies photographic tone curves,\n"
            "adds PBR micro-imperfections, ambient occlusion variation,\n"
            "and subtle film grain for natural realism.")
        self._photorealistic_check.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._photorealistic_check)

        photo_note = QLabel(
            "Removes flat game look: adds natural gradients,\n"
            "surface wear, realistic tone, and micro-detail.")
        photo_note.setWordWrap(True)
        photo_note.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px;")
        form.addRow(photo_note)

        # Material override combo
        self._material_override = QComboBox()
        self._material_override.setEnabled(False)
        self._material_override.addItem("Auto-Detect", "auto")
        for mat in [
            "brick", "concrete", "stone", "marble", "granite", "slate",
            "wood", "plywood", "bark", "bamboo",
            "metal", "rust", "copper", "bronze", "steel", "iron",
            "fabric", "leather", "carpet", "silk",
            "plastic", "rubber", "glass",
            "grass", "soil", "sand", "gravel",
            "tile", "paint", "paper",
        ]:
            self._material_override.addItem(mat.capitalize(), mat)
        self._material_override.setToolTip(
            "Force a specific material type instead of auto-detection.")
        self._material_override.currentIndexChanged.connect(self.settings_changed.emit)
        form.addRow("Material:", self._material_override)

        # ── Pipeline stage toggles ──
        stage_sep = QLabel("Pipeline Stages")
        stage_sep.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px; "
            f"font-weight: bold; padding-top: 6px;")
        form.addRow(stage_sep)

        self._stage_clahe = QCheckBox("Adaptive Contrast (CLAHE)")
        self._stage_clahe.setChecked(True)
        self._stage_clahe.setEnabled(False)
        self._stage_clahe.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_clahe)

        self._stage_freq_split = QCheckBox("Frequency-Split Detail")
        self._stage_freq_split.setChecked(True)
        self._stage_freq_split.setEnabled(False)
        self._stage_freq_split.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_freq_split)

        self._stage_detail_synth = QCheckBox("HF Detail Synthesis")
        self._stage_detail_synth.setChecked(True)
        self._stage_detail_synth.setEnabled(False)
        self._stage_detail_synth.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_detail_synth)

        self._stage_micro_overlay = QCheckBox("Micro-Detail Overlay")
        self._stage_micro_overlay.setChecked(True)
        self._stage_micro_overlay.setEnabled(False)
        self._stage_micro_overlay.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_micro_overlay)

        self._stage_wavelet = QCheckBox("Wavelet Sharpening")
        self._stage_wavelet.setChecked(True)
        self._stage_wavelet.setEnabled(False)
        self._stage_wavelet.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_wavelet)

        self._stage_sharpen = QCheckBox("Material Sharpening")
        self._stage_sharpen.setChecked(True)
        self._stage_sharpen.setEnabled(False)
        self._stage_sharpen.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_sharpen)

        self._stage_structure = QCheckBox("Structure Enhancement")
        self._stage_structure.setChecked(True)
        self._stage_structure.setEnabled(False)
        self._stage_structure.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_structure)

        self._stage_colour = QCheckBox("Colour Enhancement")
        self._stage_colour.setChecked(True)
        self._stage_colour.setEnabled(False)
        self._stage_colour.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._stage_colour)

        # Material detection result (shown after processing)
        self._material_label = QLabel("")
        self._material_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; font-weight: bold;")
        self._material_label.setWordWrap(True)
        self._material_label.setVisible(False)
        form.addRow("Detected:", self._material_label)

        self._layout.addWidget(group)

    def _on_enhance_toggled(self, state):
        enabled = bool(state)
        self._enhance_strength.setEnabled(enabled)
        self._anti_halo_check.setEnabled(enabled)
        self._photorealistic_check.setEnabled(enabled)
        self._material_override.setEnabled(enabled)
        self._stage_clahe.setEnabled(enabled)
        self._stage_freq_split.setEnabled(enabled)
        self._stage_detail_synth.setEnabled(enabled)
        self._stage_micro_overlay.setEnabled(enabled)
        self._stage_wavelet.setEnabled(enabled)
        self._stage_sharpen.setEnabled(enabled)
        self._stage_structure.setEnabled(enabled)
        self._stage_colour.setEnabled(enabled)

    def show_detected_material(self, material_name: str, confidence: float):
        """Show the detected material type after processing."""
        if material_name:
            self._material_label.setText(
                f"{material_name.capitalize()} ({confidence:.0%} confidence)")
            self._material_label.setVisible(True)
        else:
            self._material_label.setVisible(False)

    # ------------------------------------------------------------------
    # PBR Generation
    # ------------------------------------------------------------------

    def _build_pbr_group(self):
        from .theme import ACCENT, TEXT_SECONDARY
        group = QGroupBox("PBR Map Generation")
        form = QFormLayout(group)
        form.setSpacing(6)

        # Master enable
        self._pbr_enable = QCheckBox("Enable PBR Map Generation")
        self._pbr_enable.setToolTip(
            "Generate roughness, metalness, and normal maps from\n"
            "colour textures using AI material analysis.\n"
            "Updated maps are written alongside the upscaled textures\n"
            "and optionally patched into VMAT files.")
        self._pbr_enable.stateChanged.connect(self._on_pbr_toggled)
        self._pbr_enable.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._pbr_enable)

        pbr_desc = QLabel(
            "Analyses colour texture to generate PBR maps\n"
            "tuned for the detected material type.")
        pbr_desc.setWordWrap(True)
        pbr_desc.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; padding: 2px 0;")
        form.addRow(pbr_desc)

        # Individual map toggles
        self._pbr_roughness = QCheckBox("Generate Roughness Map")
        self._pbr_roughness.setChecked(True)
        self._pbr_roughness.setEnabled(False)
        self._pbr_roughness.setToolTip(
            "Create a roughness map based on surface detail analysis.\n"
            "Rough surfaces get high values, smooth get low.")
        self._pbr_roughness.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._pbr_roughness)

        self._pbr_metalness = QCheckBox("Generate Metalness Map")
        self._pbr_metalness.setChecked(True)
        self._pbr_metalness.setEnabled(False)
        self._pbr_metalness.setToolTip(
            "Detect metallic regions and generate a metalness mask.\n"
            "Also sets g_flMetalness in the VMAT.")
        self._pbr_metalness.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._pbr_metalness)

        self._pbr_normals = QCheckBox("Generate Normal Map")
        self._pbr_normals.setChecked(True)
        self._pbr_normals.setEnabled(False)
        self._pbr_normals.setToolTip(
            "Generate a tangent-space normal map from the colour texture\n"
            "using multi-scale Sobel gradient analysis.")
        self._pbr_normals.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._pbr_normals)

        # Strength slider
        self._pbr_strength = QSlider(Qt.Orientation.Horizontal)
        self._pbr_strength.setRange(0, 200)
        self._pbr_strength.setValue(100)
        self._pbr_strength.setEnabled(False)
        self._pbr_strength_label = QLabel("1.0")
        self._pbr_strength.valueChanged.connect(
            lambda v: self._pbr_strength_label.setText(f"{v / 100:.1f}"))
        self._pbr_strength.valueChanged.connect(self.settings_changed.emit)
        row = QHBoxLayout()
        row.addWidget(self._pbr_strength)
        row.addWidget(self._pbr_strength_label)
        form.addRow("PBR Strength:", row)

        # VMAT modification
        self._pbr_update_vmat = QCheckBox("Update VMAT Files")
        self._pbr_update_vmat.setChecked(True)
        self._pbr_update_vmat.setEnabled(False)
        self._pbr_update_vmat.setToolTip(
            "Automatically update VMAT files to reference the new\n"
            "roughness/normal/metalness maps and set PBR parameters.\n"
            "A backup (.vmat.bak) is created before modification.")
        self._pbr_update_vmat.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._pbr_update_vmat)

        # Auto roughness/metalness adjustment
        self._pbr_auto_values = QCheckBox("Auto-Adjust Roughness/Metalness Values")
        self._pbr_auto_values.setChecked(True)
        self._pbr_auto_values.setEnabled(False)
        self._pbr_auto_values.setToolTip(
            "Use AI to determine proper roughness and metalness scalar\n"
            "values for the VMAT based on material type analysis.\n"
            "Sets g_flMetalness and inline roughness values.")
        self._pbr_auto_values.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._pbr_auto_values)

        self._layout.addWidget(group)

    def _on_pbr_toggled(self, state):
        enabled = bool(state)
        self._pbr_roughness.setEnabled(enabled)
        self._pbr_metalness.setEnabled(enabled)
        self._pbr_normals.setEnabled(enabled)
        self._pbr_strength.setEnabled(enabled)
        self._pbr_update_vmat.setEnabled(enabled)
        self._pbr_auto_values.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _build_output_group(self):
        group = QGroupBox("Output & Compression")
        form = QFormLayout(group)
        form.setSpacing(6)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["PNG", "JPEG", "TGA", "WebP"])
        self._format_combo.currentIndexChanged.connect(self.settings_changed.emit)
        form.addRow("Format:", self._format_combo)

        # Compression toggle + quality
        self._compression_check = QCheckBox("Enable Compression")
        self._compression_check.setChecked(True)
        self._compression_check.setToolTip(
            "Apply lossy compression to reduce output file size dramatically.\n"
            "Disable for lossless output (larger files).")
        self._compression_check.stateChanged.connect(self._on_compression_toggled)
        self._compression_check.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._compression_check)

        self._compress_quality = QSlider(Qt.Orientation.Horizontal)
        self._compress_quality.setRange(50, 100)
        self._compress_quality.setValue(90)
        self._compress_quality_label = QLabel("90")
        self._compress_quality.valueChanged.connect(
            lambda v: self._compress_quality_label.setText(str(v)))
        self._compress_quality.valueChanged.connect(self.settings_changed.emit)
        row = QHBoxLayout()
        row.addWidget(self._compress_quality)
        row.addWidget(self._compress_quality_label)
        form.addRow("Quality:", row)

        note = QLabel("90 = great quality, ~5× smaller files")
        note.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px;")
        note.setWordWrap(True)
        form.addRow(note)

        self._layout.addWidget(group)

    def _on_compression_toggled(self, state):
        enabled = bool(state)
        self._compress_quality.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Advanced
    # ------------------------------------------------------------------

    def _build_advanced_group(self):
        group = QGroupBox("Advanced")
        form = QFormLayout(group)

        self._tile_size = QComboBox()
        self._tile_size.addItems(["Auto", "128", "256", "512", "768", "1024"])
        self._tile_size.setCurrentText("Auto")
        self._tile_size.setToolTip(
            "Auto picks the largest tile your GPU can handle.\n"
            "Smaller tiles use less VRAM but are slower.")
        self._tile_size.currentIndexChanged.connect(self.settings_changed.emit)
        form.addRow("Tile Size:", self._tile_size)

        self._tile_overlap = QSpinBox()
        self._tile_overlap.setRange(8, 128)
        self._tile_overlap.setValue(32)
        self._tile_overlap.setSuffix(" px")
        self._tile_overlap.valueChanged.connect(self.settings_changed.emit)
        form.addRow("Tile Overlap:", self._tile_overlap)

        self._half_precision = QCheckBox("Half Precision (FP16)")
        self._half_precision.setChecked(True)
        self._half_precision.setToolTip(
            "Uses less VRAM and is faster on most GPUs. Disable if you see artefacts.")
        self._half_precision.stateChanged.connect(self.settings_changed.emit)
        form.addRow(self._half_precision)

        self._gpu_id = QSpinBox()
        self._gpu_id.setRange(0, 7)
        self._gpu_id.setValue(0)
        form.addRow("GPU ID:", self._gpu_id)

        self._layout.addWidget(group)

    # ------------------------------------------------------------------
    # Collect settings
    # ------------------------------------------------------------------

    def get_settings(self) -> UpscaleSettings:
        """Build an UpscaleSettings from current panel state."""
        scales = [2, 4]
        sidx = self._scale_combo.currentIndex()

        fmt_map = {
            "PNG": "png", "JPEG": "jpeg", "TGA": "tga", "WebP": "webp",
        }

        variant_val = self._model_combo.currentData()
        variant = ModelVariant(variant_val)

        textype_val = self._textype_combo.currentData()
        textype = TextureType(textype_val)

        # Max output size
        max_size_text = self._max_size_combo.currentText()
        max_output_size = int(max_size_text) if max_size_text.isdigit() else 0

        compression_on = self._compression_check.isChecked()
        compress_q = self._compress_quality.value()

        return UpscaleSettings(
            scale_factor=scales[sidx] if 0 <= sidx < len(scales) else 4,
            model_variant=variant,
            texture_type=textype,
            tile_size=0 if self._tile_size.currentText() == "Auto" else int(self._tile_size.currentText()),
            tile_overlap=self._tile_overlap.value(),
            half_precision=self._half_precision.isChecked(),
            preserve_alpha=self._preserve_alpha.isChecked(),
            seamless_mode=self._seamless.isChecked(),
            denoise_strength=0.5,
            sharpen_amount=0.0,
            color_correction=True,
            gpu_id=self._gpu_id.value(),
            output_format=fmt_map.get(self._format_combo.currentText(), "png"),
            output_quality=compress_q if compression_on else 100,
            generate_mipmaps=False,
            max_mipmap_levels=8,
            max_output_size=max_output_size,
            compression_quality=compress_q,
            compression_enabled=compression_on,
            generative_enhance=self._enhance_check.isChecked(),
            enhance_strength=self._enhance_strength.value() / 100.0,
            photorealistic_mode=self._photorealistic_check.isChecked(),
            anti_halo=self._anti_halo_check.isChecked(),
            material_override=self._material_override.currentData() or "auto",
            enhance_clahe=self._stage_clahe.isChecked(),
            enhance_frequency_split=self._stage_freq_split.isChecked(),
            enhance_detail_synth=self._stage_detail_synth.isChecked(),
            enhance_micro_overlay=self._stage_micro_overlay.isChecked(),
            enhance_wavelet=self._stage_wavelet.isChecked(),
            enhance_sharpen=self._stage_sharpen.isChecked(),
            enhance_structure=self._stage_structure.isChecked(),
            enhance_colour=self._stage_colour.isChecked(),
            pbr_generate=self._pbr_enable.isChecked(),
            pbr_roughness=self._pbr_roughness.isChecked(),
            pbr_metalness=self._pbr_metalness.isChecked(),
            pbr_normals=self._pbr_normals.isChecked(),
            pbr_strength=self._pbr_strength.value() / 100.0,
            pbr_update_vmat=self._pbr_update_vmat.isChecked(),
            pbr_auto_values=self._pbr_auto_values.isChecked(),
        )
