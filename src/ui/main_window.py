"""
Main application window for CS2 Texture Upscaler.

Layout:
┌──────────────────────────────────────────────────────────┐
│  Menu Bar                                                │
├──────────────────────────────────────────────────────────┤
│  Toolbar                                                 │
├──────────────┬─────────────────────┬─────────────────────┤
│  Left Tabs   │   Preview Canvas    │  VMAT / Texture     │
│  - Materials │   (before/after     │  Info Panel         │
│    Browser   │    comparison)      │  (right dock)       │
│  - Settings  │                     │                     │
├──────────────┴─────────────────────┴─────────────────────┤
│  Batch Progress Panel (bottom)                           │
├──────────────────────────────────────────────────────────┤
│  Status Bar  │  GPU Info                                 │
└──────────────────────────────────────────────────────────┘
"""

import logging
import os
import traceback
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QMessageBox, QStatusBar, QLabel, QMenuBar,
    QToolBar, QTabWidget, QDockWidget, QApplication, QProgressBar,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QScrollArea,
    QFrame, QGroupBox, QPushButton, QLineEdit, QComboBox,
    QStackedLayout,
)

import numpy as np

logger = logging.getLogger("CS2Upscaler.MainWindow")


class MainWindow(QMainWindow):
    """CS2 Texture Upscaler – main window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS2 Neural VMAT Upscaler")
        self.setMinimumSize(1280, 800)
        self.resize(1600, 1000)

        # Lazy-imported references (set in _init_core)
        self._upscaler = None
        self._model_manager = None
        self._pipeline = None

        # State
        self._current_image: Optional[np.ndarray] = None
        self._current_info = None
        self._current_result = None
        self._loaded_files: list = []
        self._materials_root: str = ""  # root folder for recursive scanning
        self._vmats: list = []  # parsed VmatMaterial list
        self._preview_cache: dict = {}  # {job_id: (original_array, upscaled_array)}
        self._models_dir = str(Path(__file__).resolve().parent.parent.parent / "models")

        # Build lightweight UI shell immediately (no heavy imports)
        self._build_loading_ui()
        self.show()
        QApplication.processEvents()

        # Defer heavy init to after the window is visible
        QTimer.singleShot(0, self._init_core)

    # ------------------------------------------------------------------
    # Phased loading
    # ------------------------------------------------------------------

    def _build_loading_ui(self):
        """Build a minimal placeholder UI shown while heavy deps load."""
        from .theme import BG_DARK, TEXT_SECONDARY, ACCENT
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("CS2 Texture Upscaler")
        title.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {ACCENT}; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._loading_label = QLabel("Loading AI engine...")
        self._loading_label.setStyleSheet(
            f"font-size: 14px; color: {TEXT_SECONDARY}; padding: 20px;")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._loading_label)

        self._loading_progress = QProgressBar()
        self._loading_progress.setRange(0, 0)  # indeterminate
        self._loading_progress.setMaximumWidth(400)
        layout.addWidget(self._loading_progress, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setCentralWidget(central)

        # Minimal status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("Initialising...")
        self._status_bar.addWidget(self._status_label)

    def _init_core(self):
        """Load heavy dependencies and build the full UI (runs after window shown)."""
        import time
        t0 = time.perf_counter()

        self._loading_label.setText("Loading PyTorch and CUDA...")
        QApplication.processEvents()

        # --- Heavy imports (PyTorch, basicsr, etc.) ---
        from ..core.upscaler import TextureUpscaler
        from ..core.models import ModelManager
        from ..core.pipeline import ProcessingPipeline

        self._loading_label.setText("Initialising AI models...")
        QApplication.processEvents()

        self._upscaler = TextureUpscaler(self._models_dir)
        self._model_manager = ModelManager(self._models_dir)
        self._pipeline = ProcessingPipeline(self._upscaler)

        self._loading_label.setText("Building interface...")
        QApplication.processEvents()

        # Replace the loading placeholder with the real UI
        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()

        # GPU info timer
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_info)
        self._gpu_timer.start(3000)
        self._update_gpu_info()

        elapsed = time.perf_counter() - t0
        logger.info(f"Core initialised in {elapsed:.1f}s")
        self._status_label.setText("Ready")

        # Auto-load Slot 1 (Default) settings on startup
        self._settings_panel.load_slot(1)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        from .batch_panel import BatchPanel
        from .preview_widget import PreviewWidget
        from .settings_panel import SettingsPanel
        from .theme import (ACCENT, TEXT_SECONDARY, SUCCESS, BG_DARK,
                            BG_LIGHT, BG_MID, BORDER, TEXT_PRIMARY)
        from .widgets import DragDropArea, GPUInfoBar, TextureInfoPanel

        # ── Central area ─────────────────────────────────────────
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Vertical splitter: top area | batch panel
        vsplitter = QSplitter(Qt.Orientation.Vertical)

        # Horizontal splitter: left tabs | preview
        hsplitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: tabbed (Materials Browser / Settings) ────
        self._left_tabs = QTabWidget()
        self._left_tabs.setMinimumWidth(380)

        # --- Materials Browser tab ---
        browser_widget = QWidget()
        browser_layout = QVBoxLayout(browser_widget)
        browser_layout.setContentsMargins(6, 6, 6, 6)
        browser_layout.setSpacing(4)

        # Open Folder button
        open_folder_btn = QPushButton("📂  Open Materials Folder")
        open_folder_btn.setMinimumHeight(36)
        open_folder_btn.setStyleSheet(
            f"QPushButton {{ background: {ACCENT}; color: #fff; "
            f"font-weight: bold; border-radius: 4px; padding: 6px 12px; }}"
            f"QPushButton:hover {{ background: {ACCENT}cc; }}")
        open_folder_btn.clicked.connect(self._on_open_folder)
        browser_layout.addWidget(open_folder_btn)

        # Folder path label
        self._folder_label = QLabel("No folder selected")
        self._folder_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; padding: 2px;")
        self._folder_label.setWordWrap(True)
        browser_layout.addWidget(self._folder_label)

        # File count summary
        self._file_count_label = QLabel("")
        self._file_count_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        browser_layout.addWidget(self._file_count_label)

        # Material tree
        self._mat_tree = QTreeWidget()
        self._mat_tree.setHeaderLabels(["Name", "Type", "Dimensions", "Status", "Text"])
        # All columns are Interactive (user-draggable / resizable)
        header = self._mat_tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)
        # Reasonable initial widths (column 0 is Name – give it the most space)
        self._mat_tree.setColumnWidth(0, 250)
        self._mat_tree.setColumnWidth(1, 180)
        self._mat_tree.setColumnWidth(2, 90)
        self._mat_tree.setColumnWidth(3, 80)
        self._mat_tree.setColumnWidth(4, 40)
        self._mat_tree.setAlternatingRowColors(True)
        self._mat_tree.setRootIsDecorated(True)
        self._mat_tree.itemClicked.connect(self._on_tree_item_clicked)
        # Restore persisted column widths
        self._restore_tree_columns()
        header.sectionResized.connect(self._on_tree_column_resized)
        browser_layout.addWidget(self._mat_tree, 1)

        self._left_tabs.addTab(browser_widget, "Materials")

        # --- Settings tab ---
        self._settings_panel = SettingsPanel(self._model_manager)
        self._left_tabs.addTab(self._settings_panel, "Settings")

        # ── Preview area (centre) ────────────────────────────────
        self._preview = PreviewWidget()
        self._drag_drop = DragDropArea()
        self._preview_stack = QWidget()
        self._stack_layout = QStackedLayout(self._preview_stack)
        self._stack_layout.setContentsMargins(0, 0, 0, 0)
        self._stack_layout.addWidget(self._drag_drop)   # index 0
        self._stack_layout.addWidget(self._preview)      # index 1
        self._stack_layout.setCurrentIndex(0)

        self._hsplitter = hsplitter
        hsplitter.addWidget(self._left_tabs)
        hsplitter.addWidget(self._preview_stack)
        hsplitter.setStretchFactor(0, 0)
        hsplitter.setStretchFactor(1, 1)
        hsplitter.setSizes([420, 860])
        hsplitter.setChildrenCollapsible(False)

        # ── Batch panel (bottom) ─────────────────────────────────
        self._batch_panel = BatchPanel()

        self._vsplitter = vsplitter
        vsplitter.addWidget(hsplitter)
        vsplitter.addWidget(self._batch_panel)
        vsplitter.setStretchFactor(0, 3)
        vsplitter.setStretchFactor(1, 0)
        vsplitter.setSizes([600, 350])
        vsplitter.setChildrenCollapsible(False)

        main_layout.addWidget(vsplitter)
        self.setCentralWidget(central)

        # ── Right dock: VMAT / Texture info ──────────────────────
        self._info_panel = TextureInfoPanel()
        self._vmat_info = _VmatInfoWidget()

        info_tabs = QTabWidget()
        info_tabs.addTab(self._vmat_info, "VMAT")
        info_tabs.addTab(self._info_panel, "Texture")

        dock = QDockWidget("Material Info", self)
        dock.setWidget(info_tabs)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)
        dock.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ── Status bar ───────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._gpu_bar = GPUInfoBar()
        self._status_bar.addPermanentWidget(self._gpu_bar)

        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

        # ── Debug console dock ───────────────────────────────────
        from PyQt6.QtWidgets import QPlainTextEdit
        self._debug_console = QPlainTextEdit()
        self._debug_console.setReadOnly(True)
        self._debug_console.setMaximumBlockCount(5000)
        self._debug_console.setStyleSheet(
            f"font-family: 'Cascadia Code', 'Consolas', monospace; "
            f"font-size: 11px; background: #1a1a2e; color: #c0c0c0; "
            f"border: none; padding: 4px;"
        )
        self._debug_dock = QDockWidget("Debug Console", self)
        self._debug_dock.setWidget(self._debug_console)
        self._debug_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._debug_dock)
        self._debug_dock.hide()

        # Install log handler that writes to the debug console
        self._install_debug_log_handler()

    def _build_menu(self):
        menubar = self.menuBar()

        # File
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Texture...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)

        open_folder = QAction("Open Materials &Folder...", self)
        open_folder.setShortcut("Ctrl+Shift+O")
        open_folder.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_folder)

        file_menu.addSeparator()

        save_action = QAction("&Save Result...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit
        edit_menu = menubar.addMenu("&Edit")

        clear_action = QAction("&Clear Preview", self)
        clear_action.triggered.connect(self._on_clear)
        edit_menu.addAction(clear_action)

        # Process
        proc_menu = menubar.addMenu("&Process")

        upscale_action = QAction("&Upscale Current", self)
        upscale_action.setShortcut("Ctrl+U")
        upscale_action.triggered.connect(self._on_upscale_single)
        proc_menu.addAction(upscale_action)

        batch_action = QAction("Start &Batch", self)
        batch_action.setShortcut("Ctrl+B")
        batch_action.triggered.connect(self._on_start_batch)
        proc_menu.addAction(batch_action)

        proc_menu.addSeparator()

        cancel_action = QAction("Cance&l", self)
        cancel_action.setShortcut("Escape")
        cancel_action.triggered.connect(self._on_cancel)
        proc_menu.addAction(cancel_action)

        # View
        view_menu = menubar.addMenu("&View")

        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(lambda: self._preview.canvas.fit_to_view())
        view_menu.addAction(fit_action)

        zoom100 = QAction("Zoom &100%", self)
        zoom100.setShortcut("Ctrl+1")
        zoom100.triggered.connect(lambda: self._preview.canvas.zoom_100())
        view_menu.addAction(zoom100)

        # Help
        help_menu = menubar.addMenu("&Help")

        about = QAction("&About", self)
        about.triggered.connect(self._on_about)
        help_menu.addAction(about)

    def _build_toolbar(self):
        from .theme import ACCENT, ACCENT_HOVER, TEXT_PRIMARY, BG_LIGHT, BORDER, SUCCESS, ERROR

        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setStyleSheet(f"""
            QToolBar {{
                background-color: {BG_LIGHT};
                border-bottom: 1px solid {BORDER};
                padding: 2px 4px;
                spacing: 2px;
            }}
            QToolButton {{
                padding: 5px 12px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
                border: none;
                color: {TEXT_PRIMARY};
            }}
            QToolButton:hover {{
                background-color: rgba(255, 255, 255, 0.08);
            }}
        """)
        self.addToolBar(tb)

        open_act = tb.addAction("Open Folder")
        open_act.setToolTip("Open materials folder (Ctrl+Shift+O)")
        open_act.triggered.connect(self._on_open_folder)

        tb.addSeparator()

        upscale_act = tb.addAction("Upscale Current")
        upscale_act.setToolTip("Upscale the selected texture (Ctrl+U)")
        upscale_act.triggered.connect(self._on_upscale_single)

        batch_act = tb.addAction("Upscale All")
        batch_act.setToolTip("Start batch processing all textures (Ctrl+B)")
        batch_act.triggered.connect(self._on_start_batch)

        cancel_act = tb.addAction("Cancel")
        cancel_act.setToolTip("Cancel current operation (Escape)")
        cancel_act.triggered.connect(self._on_cancel)

        tb.addSeparator()

        save_act = tb.addAction("Save Result")
        save_act.setToolTip("Save upscaled result (Ctrl+S)")
        save_act.triggered.connect(self._on_save)

        tb.addSeparator()

        debug_act = tb.addAction("Debug Console")
        debug_act.setToolTip("Show/hide debug log output")
        debug_act.triggered.connect(self._toggle_debug_console)

    def _connect_signals(self):
        from ..core.pipeline import JobStatus

        # Drag & drop
        self._drag_drop.files_dropped.connect(self._on_files_dropped)

        # Batch panel
        self._batch_panel.request_start.connect(self._on_start_batch)
        self._batch_panel.request_cancel.connect(self._on_cancel)
        self._batch_panel.request_clear.connect(self._on_clear_queue)
        self._batch_panel.job_selected.connect(self._on_job_selected)

        # Pipeline signals
        self._pipeline.job_started.connect(
            lambda jid: self._batch_panel.update_job_status(jid, JobStatus.PROCESSING))
        self._pipeline.job_progress.connect(self._batch_panel.update_job_progress)
        self._pipeline.job_completed.connect(self._on_job_completed)
        self._pipeline.job_failed.connect(self._on_job_failed)
        self._pipeline.batch_progress.connect(self._batch_panel.update_overall_progress)
        self._pipeline.batch_completed.connect(self._on_batch_completed)
        self._pipeline.pipeline_status.connect(
            lambda msg: self._status_label.setText(msg))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_open(self):
        from ..core.texture_io import build_file_filter
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open Textures", "", build_file_filter())
        if files:
            self._on_files_dropped(files)

    def _on_open_folder(self):
        """Open a materials folder and recursively scan for textures and VMATs."""
        folder = QFileDialog.getExistingDirectory(
            self, "Open Materials Folder")
        if not folder:
            return
        self._load_materials_folder(folder)

    def _load_materials_folder(self, folder: str):
        """Recursively scan a folder for textures and VMAT files."""
        from ..core.vmat_parser import scan_vmats, scan_textures_recursive

        self._materials_root = folder
        self._folder_label.setText(folder)
        self._status_label.setText("Scanning folder...")
        QApplication.processEvents()

        # Scan textures recursively
        texture_files = scan_textures_recursive(folder)
        self._loaded_files = texture_files

        # Scan VMATfiles
        self._vmats = scan_vmats(folder, folder)

        # Update file count label
        vmat_count = len(self._vmats)
        tex_count = len(texture_files)
        self._file_count_label.setText(
            f"{tex_count} textures, {vmat_count} VMAT files found")

        # Build material tree
        self._build_material_tree(folder, texture_files)

        # Set output dir to upscaled/ sibling
        parent = str(Path(folder).parent)
        out_dir = str(Path(parent) / "upscaled" / Path(folder).name)
        self._batch_panel._out_path_label.setText(out_dir)

        self._status_label.setText(
            f"Loaded {tex_count} textures, {vmat_count} VMATs from "
            f"{Path(folder).name}")

    def _build_material_tree(self, root: str, texture_files: list):
        """Build a unified folder tree mirroring the actual directory layout.

        Shows VMATs and textures side-by-side in their real folders.
        VMATs are expandable to reveal their texture dependencies.
        """
        from .theme import SUCCESS, WARNING, TEXT_SECONDARY, ACCENT

        self._mat_tree.clear()
        root_path = Path(root)

        # Index VMATfiles by directory for fast lookup
        vmat_by_dir: dict = {}  # relative dir str -> [VmatMaterial, …]
        for vmat in (self._vmats or []):
            vmat_path = Path(vmat.path) if hasattr(vmat, 'path') else None
            if vmat_path:
                try:
                    rel_dir = str(vmat_path.parent.relative_to(root_path))
                except ValueError:
                    rel_dir = "."
            else:
                rel_dir = "."
            vmat_by_dir.setdefault(rel_dir, []).append(vmat)

        # Index texture files by directory
        tex_by_dir: dict = {}  # relative dir str -> [(full_path, rel_path), …]
        for fpath in texture_files:
            try:
                rel = Path(fpath).relative_to(root_path)
                rel_dir = str(rel.parent) if rel.parent != Path(".") else "."
            except ValueError:
                rel_dir = "."
                rel = Path(fpath).name
            tex_by_dir.setdefault(rel_dir, []).append((fpath, rel))

        # Collect all directories that have content
        all_dirs = sorted(set(list(vmat_by_dir.keys()) + list(tex_by_dir.keys())))

        # Build folder nodes
        folders: dict = {}  # relative path str -> QTreeWidgetItem

        def get_folder_node(rel_dir: str) -> QTreeWidgetItem:
            """Get or create a folder node, building parent chain as needed."""
            if rel_dir == "." or rel_dir == "":
                return self._mat_tree.invisibleRootItem()
            if rel_dir in folders:
                return folders[rel_dir]

            parts = Path(rel_dir).parts
            parent = self._mat_tree.invisibleRootItem()
            for i, part in enumerate(parts):
                key = "/".join(parts[:i + 1]) if i > 0 else parts[0]
                # Normalise separators
                key = key.replace("\\", "/")
                if key not in folders:
                    item = QTreeWidgetItem(parent, [part, "📁 Folder", "", "", ""])
                    item.setExpanded(len(parts) <= 2)  # auto-expand top 2 levels
                    folders[key] = item
                parent = folders[key]
            return parent

        # Populate
        for d in all_dirs:
            parent_node = get_folder_node(d)
            dir_vmats = vmat_by_dir.get(d, [])
            dir_texes = tex_by_dir.get(d, [])

            # Add VMAT files first
            for vmat in sorted(dir_vmats, key=lambda v: v.filename):
                surf = ""
                if hasattr(vmat, 'surface_property') and vmat.surface_property:
                    surf = f"[{vmat.surface_property}] "
                shader_desc = vmat.shader_description or "Shader"
                vmat_item = QTreeWidgetItem(parent_node, [
                    vmat.filename,
                    f"📄 VMAT — {surf}{shader_desc}",
                    "",
                    f"{vmat.texture_count} textures",
                    "",
                ])
                vmat_item.setData(0, Qt.ItemDataRole.UserRole, ("vmat", vmat))
                vmat_item.setExpanded(False)

                # Child texture entries from VMAT
                for tex in vmat.textures:
                    tex_dims = ""
                    tex_status = "✗ Missing"
                    if tex.resolved_path and Path(tex.resolved_path).exists():
                        tex_status = "✓ Found"
                        try:
                            from PIL import Image as PILImage
                            with PILImage.open(tex.resolved_path) as img:
                                tex_dims = f"{img.width}×{img.height}"
                        except Exception:
                            pass

                    tex_item = QTreeWidgetItem(vmat_item, [
                        Path(tex.path).name if tex.path else "?",
                        tex.role.value.upper(),
                        tex_dims,
                        tex_status,
                        "",
                    ])
                    if tex.resolved_path:
                        tex_item.setData(
                            0, Qt.ItemDataRole.UserRole, ("texture", tex.resolved_path))

            # Build a set of texture paths already shown under VMATs
            vmat_tex_paths = set()
            for vmat in dir_vmats:
                for tex in vmat.textures:
                    if tex.resolved_path:
                        vmat_tex_paths.add(str(Path(tex.resolved_path).resolve()))

            # Add standalone texture files (not already under a VMAT)
            for fpath, rel in sorted(dir_texes, key=lambda x: str(x[1])):
                resolved = str(Path(fpath).resolve())
                if resolved in vmat_tex_paths:
                    continue  # Already shown as a VMAT child

                fname = Path(fpath).name
                ext = Path(fname).suffix.upper().lstrip(".")

                dims = ""
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(fpath) as img:
                        dims = f"{img.width}×{img.height}"
                except Exception:
                    pass

                leaf = QTreeWidgetItem(parent_node, [
                    fname, ext, dims, "Pending", ""])
                leaf.setData(0, Qt.ItemDataRole.UserRole, ("texture", fpath))
                # Has Text checkbox
                leaf.setCheckState(4, Qt.CheckState.Unchecked)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle clicking an item in the material tree."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        kind, value = data
        if kind == "texture" and isinstance(value, str):
            self._load_single_texture(value)
        elif kind == "vmat":
            self._show_vmat_info(value)

    def _load_single_texture(self, filepath: str):
        """Load and preview a single texture file."""
        from ..core.texture_io import load_texture
        try:
            image, info = load_texture(filepath)
            self._current_image = image
            self._current_info = info
            self._current_result = None  # clear old result
            self._info_panel.update_info(info)

            # Clear any old upscaled result from the canvas so Original view
            # mode shows the newly loaded image instead of stale data.
            self._preview.clear()

            self._stack_layout.setCurrentIndex(1)  # show preview
            QApplication.processEvents()  # let canvas get a real size
            self._preview.set_original(image)

            self._status_label.setText(
                f"Loaded: {info.filename}  ({info.width}×{info.height})")
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            QMessageBox.warning(self, "Load Error", str(e))

    def _show_vmat_info(self, vmat):
        """Display parsed VMAT info in the info panel."""
        self._vmat_info.set_vmat(vmat)
        # Auto-load the first existing color texture into the preview
        for tex in vmat.textures:
            if tex.resolved_path and Path(tex.resolved_path).exists():
                role = tex.role.value.lower() if hasattr(tex.role, 'value') else str(tex.role).lower()
                if any(k in role for k in ('color', 'diffuse', 'albedo', 'base')):
                    self._load_single_texture(tex.resolved_path)
                    return
        # Fallback: load the first existing texture of any type
        for tex in vmat.textures:
            if tex.resolved_path and Path(tex.resolved_path).exists():
                self._load_single_texture(tex.resolved_path)
                return

    def _on_files_dropped(self, files: list):
        if not files:
            return

        self._loaded_files = files

        # Load first file for preview
        self._load_single_texture(files[0])

        # Add all files to batch queue
        if len(files) > 1:
            settings = self._settings_panel.get_settings()
            out_dir = self._batch_panel.output_dir
            if not out_dir:
                out_dir = str(Path(files[0]).parent / "upscaled")
            ids = self._pipeline.add_batch(
                files, out_dir, settings, input_root=self._materials_root)
            for jid in ids:
                job = self._pipeline.get_job(jid)
                if job:
                    self._batch_panel.add_job(job)
            self._update_batch_stats()

    def _on_save(self):
        if self._current_result is None:
            QMessageBox.information(self, "Nothing to Save",
                                    "Upscale a texture first.")
            return
        self._on_save_as()

    def _on_save_as(self):
        if self._current_result is None:
            return
        from ..core.texture_io import TextureFormat, detect_format, save_texture
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Upscaled Texture", "",
            "PNG (*.png);;JPEG (*.jpg);;TGA (*.tga);;BMP (*.bmp);;TIFF (*.tiff);;WebP (*.webp)")
        if path:
            fmt = detect_format(path) or TextureFormat.PNG
            quality = self._settings_panel.get_settings().output_quality
            save_texture(self._current_result.image, path, fmt=fmt, quality=quality)
            self._status_label.setText(f"Saved: {Path(path).name}")

    def _on_clear(self):
        self._current_image = None
        self._current_info = None
        self._current_result = None
        self._preview.clear()
        self._stack_layout.setCurrentIndex(0)  # show drag-drop
        self._info_panel.update_info(None)
        self._status_label.setText("Ready")

    def _on_upscale_single(self):
        """Upscale the currently loaded single texture."""
        if self._current_image is None:
            QMessageBox.information(self, "No Texture",
                                    "Load a texture first.")
            return

        settings = self._settings_panel.get_settings()

        # Check if current tree item has "Has Text" checked
        current_item = self._mat_tree.currentItem()
        if current_item is not None:
            try:
                text_checked = current_item.checkState(4) == Qt.CheckState.Checked
                settings.text_preserve = text_checked
            except Exception:
                pass

        # Apply VMAT surface property as material override if "auto"
        if settings.material_override == "auto" and current_item is not None:
            data = current_item.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "texture":
                vmat_map = self._build_vmat_material_map()
                if data[1] in vmat_map:
                    settings.material_override = vmat_map[data[1]]

        self._status_label.setText("Upscaling...")
        QApplication.processEvents()

        try:
            def progress(pct, msg):
                self._status_label.setText(f"{msg}  ({pct*100:.0f}%)")
                QApplication.processEvents()

            result = self._upscaler.upscale(
                self._current_image, settings, progress_callback=progress)
            self._current_result = result

            self._preview.set_upscaled(result.image)

            # Show detected material if generative enhance was used
            if result.detected_material:
                self._settings_panel.show_detected_material(
                    result.detected_material, result.material_confidence)

            self._status_label.setText(
                f"Done in {result.processing_time:.1f}s — "
                f"{result.original_size[0]}×{result.original_size[1]} → "
                f"{result.upscaled_size[0]}×{result.upscaled_size[1]}  "
                f"({result.model_used})"
                + (f"  [{result.detected_material}]" if result.detected_material else "")
            )

        except Exception as e:
            QMessageBox.critical(self, "Upscale Error", str(e))
            self._status_label.setText("Error during upscale")

    def _on_start_batch(self):
        if self._pipeline.get_queue_count() == 0:
            # If no explicit batch, add loaded files
            if self._loaded_files:
                settings = self._settings_panel.get_settings()
                out_dir = self._batch_panel.output_dir
                if not out_dir:
                    parent = str(Path(self._loaded_files[0]).parent)
                    if self._materials_root:
                        parent = str(Path(self._materials_root).parent)
                        out_dir = str(
                            Path(parent) / "upscaled" / Path(self._materials_root).name)
                    else:
                        out_dir = str(Path(parent) / "upscaled")
                    self._batch_panel._out_path_label.setText(out_dir)

                # Build text-preserve mapping from tree checkboxes
                text_map = self._collect_text_flags()

                # Build VMAT surface-property -> material type mapping
                vmat_material_map = self._build_vmat_material_map()

                ids = self._pipeline.add_batch(
                    self._loaded_files, out_dir, settings,
                    input_root=self._materials_root)

                # Apply per-file text_preserve flag and VMAT material override
                for jid in ids:
                    job = self._pipeline.get_job(jid)
                    if job:
                        overrides = {}
                        if job.input_path in text_map:
                            overrides['text_preserve'] = text_map[job.input_path]
                        # If material_override is "auto" and VMAT has a surface
                        # property, use that surface property as material type
                        if (job.settings.material_override == "auto" and
                                job.input_path in vmat_material_map):
                            overrides['material_override'] = vmat_material_map[job.input_path]
                        if overrides:
                            from ..core.upscaler import UpscaleSettings as _US
                            job.settings = _US(
                                **{**job.settings.__dict__, **overrides})
                        self._batch_panel.add_job(job)
            else:
                QMessageBox.information(self, "Empty Queue",
                                        "Open a materials folder first.")
                return

        self._batch_panel.set_running(True)
        self._pipeline.start()

    def _collect_text_flags(self) -> dict:
        """Walk the material tree and collect {filepath: bool} for the text checkbox."""
        result = {}
        iterator = self._mat_tree.invisibleRootItem()
        self._walk_tree_for_text(iterator, result)
        return result

    def _build_vmat_material_map(self) -> dict:
        """
        Build a mapping {texture_filepath: material_type_str} from VMAT
        surface properties.

        If a VMAT's PhysicsSurfaceProperties is set to anything other than
        'default' or empty, map all its textures to the corresponding
        MaterialType string.  This overrides the AI-based material detection.
        """
        from ..core.vmat_parser import surface_property_to_material
        result = {}
        for vmat in (self._vmats or []):
            mat_type = None
            if hasattr(vmat, 'surface_property') and vmat.surface_property:
                mat_type = surface_property_to_material(vmat.surface_property)
            if not mat_type:
                continue
            for tex in vmat.textures:
                if tex.resolved_path:
                    result[tex.resolved_path] = mat_type
        return result

    def _walk_tree_for_text(self, item, result: dict):
        for i in range(item.childCount()):
            child = item.child(i)
            data = child.data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "texture" and isinstance(data[1], str):
                try:
                    checked = child.checkState(4) == Qt.CheckState.Checked
                    result[data[1]] = checked
                except Exception:
                    pass
            self._walk_tree_for_text(child, result)

    def _on_cancel(self):
        self._pipeline.cancel()
        self._batch_panel.set_running(False)
        self._update_batch_stats()

    def _on_clear_queue(self):
        self._pipeline.reset()
        self._batch_panel._clear_table()
        self._batch_panel.set_running(False)
        self._batch_panel.update_stats(0, 0, 0, 0.0)
        self._preview_cache.clear()
        self._status_label.setText("Queue cleared")

    def _on_job_completed(self, job_id: int, result):
        try:
            from ..core.pipeline import JobStatus
            from ..core.texture_io import load_texture
            job = self._pipeline.get_job(job_id)
            self._batch_panel.update_job_status(
                job_id, JobStatus.COMPLETED, result.processing_time)
            self._update_batch_stats()

            # Show last completed result in preview
            self._current_result = result
            self._stack_layout.setCurrentIndex(1)  # show preview

            orig_image = None
            # Load original so preview can show before/after comparison
            if job and job.input_path:
                try:
                    orig_image, orig_info = load_texture(job.input_path)
                    self._current_image = orig_image
                    self._current_info = orig_info
                    self._preview.set_original(orig_image)
                    self._info_panel.update_info(orig_info)
                except Exception:
                    logger.debug(f"Could not reload original for preview: {job.input_path}")

            image = np.ascontiguousarray(result.image)
            self._preview.set_upscaled(image)

            # Cache preview images for later recall (limit to last 30 entries)
            self._preview_cache[job_id] = (orig_image, image)
            if len(self._preview_cache) > 30:
                oldest = next(iter(self._preview_cache))
                del self._preview_cache[oldest]
        except Exception:
            logger.exception(f"Error handling job #{job_id} completion")

    def _on_job_selected(self, job_id: int):
        """Restore cached preview when user clicks a job in the queue."""
        cached = self._preview_cache.get(job_id)
        if not cached:
            return
        orig, upscaled = cached
        self._stack_layout.setCurrentIndex(1)
        if orig is not None:
            self._preview.set_original(orig)
        if upscaled is not None:
            self._preview.set_upscaled(upscaled)

    def _on_job_failed(self, job_id: int, error: str):
        try:
            from ..core.pipeline import JobStatus
            self._batch_panel.update_job_status(job_id, JobStatus.FAILED)
            self._update_batch_stats()
            logger.error(f"Job #{job_id} failed: {error}")
        except Exception:
            logger.exception(f"Error handling job #{job_id} failure")

    def _on_batch_completed(self):
        try:
            from ..core.pipeline import JobStatus
            self._batch_panel.set_running(False)
            total_time = sum(
                j.processing_time for j in self._pipeline.get_all_jobs()
                if j.status == JobStatus.COMPLETED
            )
            self._status_label.setText(
                f"Batch complete — {self._pipeline.get_completed_count()} files "
                f"in {total_time:.1f}s")
            self._update_batch_stats()
        except Exception:
            logger.exception("Error in batch completion handler")

    def _update_batch_stats(self):
        from ..core.pipeline import JobStatus
        jobs = self._pipeline.get_all_jobs()
        queued = sum(1 for j in jobs if j.status == JobStatus.QUEUED)
        completed = sum(1 for j in jobs if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
        total_time = sum(j.processing_time for j in jobs
                         if j.status == JobStatus.COMPLETED)
        self._batch_panel.update_stats(queued, completed, failed, total_time)

    def _update_gpu_info(self):
        info = self._upscaler.gpu_info
        self._gpu_bar.update_info(info)

    # ------------------------------------------------------------------
    # Debug Console
    # ------------------------------------------------------------------

    def _toggle_debug_console(self):
        if self._debug_dock.isVisible():
            self._debug_dock.hide()
        else:
            self._debug_dock.show()

    def _install_debug_log_handler(self):
        import logging

        class _QtLogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self._widget = text_widget

            def emit(self, record):
                try:
                    msg = self.format(record)
                    self._widget.appendPlainText(msg)
                except RuntimeError:
                    pass  # widget destroyed

        handler = _QtLogHandler(self._debug_console)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)

    def _on_about(self):
        from .. import __version__, __app_name__
        from .theme import ACCENT, TEXT_SECONDARY
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle(f"About {__app_name__}")
        dlg.setFixedSize(420, 320)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)

        content = QLabel(
            f"<div style='text-align:center; padding: 20px;'>"
            f"<h1 style='color:{ACCENT}; margin-bottom:2px;'>{__app_name__}</h1>"
            f"<p style='font-size:14px; color:{TEXT_SECONDARY};'>v{__version__}</p>"
            f"<p style='font-size:13px;'>AI-Powered Material Upscaler for Counter-Strike 2</p>"
            f"<p style='font-size:12px; color:{TEXT_SECONDARY};'>"
            f"Real-ESRGAN &middot; PyTorch &middot; 30-Material Generative Enhancement</p>"
            f"<hr style='margin: 10px 30px;'>"
            f"<p style='font-size:12px;'>"
            f"Recursive VMAT scanning &middot; PBR map generation<br>"
            f"Alpha preservation &middot; Batch processing<br>"
            f"Before/after preview &middot; Smart tile sizing</p>"
            f"<hr style='margin: 10px 30px;'>"
            f"<p style='font-size:12px; color:{TEXT_SECONDARY};'>"
            f"&copy; 2026 BONE. All rights reserved.</p>"
            f"</div>"
        )
        content.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content.setWordWrap(True)
        layout.addWidget(content)

        btn = QPushButton("OK")
        btn.setFixedWidth(80)
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addSpacing(10)

        dlg.exec()

    # ------------------------------------------------------------------
    # Column width persistence
    # ------------------------------------------------------------------

    _COLUMN_WIDTHS_FILE = Path.home() / ".textureforge" / "column_widths.json"

    def _restore_tree_columns(self):
        """Load persisted column widths from disk."""
        import json
        try:
            if self._COLUMN_WIDTHS_FILE.exists():
                with open(self._COLUMN_WIDTHS_FILE, "r") as f:
                    widths = json.load(f)
                header = self._mat_tree.header()
                for i, w in enumerate(widths):
                    if i < header.count() and i != 0:  # skip stretch col 0
                        self._mat_tree.setColumnWidth(i, w)
        except Exception:
            pass

    def _on_tree_column_resized(self, logical_index, old_size, new_size):
        """Persist column widths whenever the user resizes one."""
        self._save_tree_columns()

    def _save_tree_columns(self):
        import json
        try:
            header = self._mat_tree.header()
            widths = [header.sectionSize(i) for i in range(header.count())]
            self._COLUMN_WIDTHS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self._COLUMN_WIDTHS_FILE, "w") as f:
                json.dump(widths, f)
        except Exception:
            pass

    def closeEvent(self, event):
        if self._pipeline and self._pipeline.is_running:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "A batch is still processing. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._pipeline.cancel()

        self._save_tree_columns()

        if self._upscaler:
            self._upscaler.unload_model()
        event.accept()


# ── VMAT Info Widget ─────────────────────────────────────────────────

class _VmatInfoWidget(QScrollArea):
    """Displays parsed VMAT material properties in a scrollable panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._content = QWidget()
        self._layout = QVBoxLayout(self._content)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)
        self.setWidget(self._content)

        self._placeholder = QLabel("Select a VMAT file to view its properties.")
        self._placeholder.setStyleSheet("color: #666; padding: 20px;")
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)

    def set_vmat(self, vmat):
        """Populate with parsed VmatMaterial data."""
        from .theme import ACCENT, TEXT_SECONDARY, SUCCESS, WARNING, ERROR, TEXT_PRIMARY

        # Clear existing
        while self._layout.count():
            child = self._layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Header
        title = QLabel(f"<b style='color:{ACCENT}'>{vmat.filename}</b>")
        title.setWordWrap(True)
        self._layout.addWidget(title)

        # Shader
        shader_label = QLabel(
            f"<b>Shader:</b> {vmat.shader_description}"
            f"<br><span style='color:{TEXT_SECONDARY}'>{vmat.shader}</span>")
        shader_label.setWordWrap(True)
        self._layout.addWidget(shader_label)

        # Textures
        if vmat.textures:
            tex_group = QGroupBox(f"Textures ({len(vmat.textures)})")
            tex_layout = QVBoxLayout(tex_group)
            tex_layout.setSpacing(2)
            for tex in vmat.textures:
                status_color = SUCCESS if tex.exists else ERROR
                status_icon = "✓" if tex.exists else "✗"
                lbl = QLabel(
                    f"<span style='color:{status_color}'>{status_icon}</span> "
                    f"<b>{tex.role.value}:</b> "
                    f"<span style='color:{TEXT_SECONDARY}'>{Path(tex.path).name if tex.path else '?'}</span>")
                lbl.setWordWrap(True)
                lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                tex_layout.addWidget(lbl)
            self._layout.addWidget(tex_group)

        # Feature flags
        if vmat.feature_flags:
            flags_group = QGroupBox(f"Feature Flags ({len(vmat.feature_flags)})")
            flags_layout = QVBoxLayout(flags_group)
            flags_layout.setSpacing(1)
            for k, v in sorted(vmat.feature_flags.items()):
                color = SUCCESS if v else TEXT_SECONDARY
                lbl = QLabel(f"<span style='color:{color}'>{k}</span> = {v}")
                lbl.setStyleSheet("font-size: 11px;")
                flags_layout.addWidget(lbl)
            self._layout.addWidget(flags_group)

        # Float params
        if vmat.float_params:
            fp_group = QGroupBox(f"Float Params ({len(vmat.float_params)})")
            fp_layout = QVBoxLayout(fp_group)
            fp_layout.setSpacing(1)
            for k, v in sorted(vmat.float_params.items()):
                lbl = QLabel(f"{k} = <b>{v:.3f}</b>")
                lbl.setStyleSheet("font-size: 11px;")
                fp_layout.addWidget(lbl)
            self._layout.addWidget(fp_group)

        # Vector params
        if vmat.vector_params:
            vp_group = QGroupBox(f"Vector Params ({len(vmat.vector_params)})")
            vp_layout = QVBoxLayout(vp_group)
            vp_layout.setSpacing(1)
            for k, v in sorted(vmat.vector_params.items()):
                vals = ", ".join(f"{c:.3f}" for c in v)
                lbl = QLabel(f"{k} = [{vals}]")
                lbl.setStyleSheet("font-size: 11px;")
                vp_layout.addWidget(lbl)
            self._layout.addWidget(vp_group)

        # Int params
        if vmat.int_params:
            ip_group = QGroupBox(f"Integer Params ({len(vmat.int_params)})")
            ip_layout = QVBoxLayout(ip_group)
            ip_layout.setSpacing(1)
            for k, v in sorted(vmat.int_params.items()):
                lbl = QLabel(f"{k} = {v}")
                lbl.setStyleSheet("font-size: 11px;")
                ip_layout.addWidget(lbl)
            self._layout.addWidget(ip_group)

        # String params
        if vmat.string_params:
            sp_group = QGroupBox(f"String Params ({len(vmat.string_params)})")
            sp_layout = QVBoxLayout(sp_group)
            sp_layout.setSpacing(1)
            for k, v in sorted(vmat.string_params.items()):
                lbl = QLabel(f"{k} = \"{v}\"")
                lbl.setWordWrap(True)
                lbl.setStyleSheet("font-size: 11px;")
                sp_layout.addWidget(lbl)
            self._layout.addWidget(sp_group)

        # Parse errors
        if vmat.parse_errors:
            err_group = QGroupBox("Parse Errors")
            err_layout = QVBoxLayout(err_group)
            for err in vmat.parse_errors:
                lbl = QLabel(f"<span style='color:{ERROR}'>{err}</span>")
                lbl.setWordWrap(True)
                err_layout.addWidget(lbl)
            self._layout.addWidget(err_group)

        self._layout.addStretch()


# ── Window closeEvent is on MainWindow, not here ─────────────────
