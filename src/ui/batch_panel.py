"""
Batch processing panel – queue table, progress bars, controls.
"""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QProgressBar, QHeaderView,
    QAbstractItemView, QFrame, QSizePolicy, QFileDialog,
)

from ..core.pipeline import JobItem, JobStatus, ProcessingPipeline
from ..core.upscaler import UpscaleSettings
from .theme import ACCENT, BG_MID, SUCCESS, ERROR, WARNING, TEXT_SECONDARY
from .widgets import InfoCard


class BatchPanel(QFrame):
    """
    Batch processing panel with a job queue table, progress tracking,
    start/cancel controls, and output directory picker.
    """

    request_start = pyqtSignal()
    request_cancel = pyqtSignal()
    request_clear = pyqtSignal()
    job_selected = pyqtSignal(int)  # job_id
    output_dir_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_dir = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Header
        header = QLabel("Batch Queue")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        # Stats row
        stats = QHBoxLayout()
        self._card_queued = InfoCard("Queued", "0")
        self._card_completed = InfoCard("Completed", "0")
        self._card_failed = InfoCard("Failed", "0")
        self._card_time = InfoCard("Total Time", "0s")
        for card in (self._card_queued, self._card_completed,
                     self._card_failed, self._card_time):
            card.setMinimumHeight(62)
            card.setMaximumHeight(80)
            stats.addWidget(card)
        layout.addLayout(stats)

        # Job table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["#", "File", "Status", "Progress", "Time"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 40)
        self._table.setColumnWidth(3, 160)
        self._table.setColumnWidth(4, 70)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setMinimumHeight(120)
        self._table.cellClicked.connect(self._on_row_clicked)
        layout.addWidget(self._table, 1)

        # Overall progress
        self._overall_progress = QProgressBar()
        self._overall_progress.setFixedHeight(26)
        self._overall_progress.setFormat("Batch Progress: %p%")
        layout.addWidget(self._overall_progress)

        # Output directory
        out_row = QHBoxLayout()
        out_label = QLabel("Output:")
        out_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._out_path_label = QLabel("Not set")
        self._out_path_label.setStyleSheet("font-weight: 500;")
        self._out_path_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        out_btn = QPushButton("Set Output Folder")
        out_btn.clicked.connect(self._pick_output_dir)

        out_row.addWidget(out_label)
        out_row.addWidget(self._out_path_label, 1)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # Action buttons
        btn_row = QHBoxLayout()

        self._btn_start = QPushButton("  ▶  Start Batch  ")
        self._btn_start.setObjectName("primaryButton")
        self._btn_start.clicked.connect(self.request_start.emit)

        self._btn_cancel = QPushButton("  ■  Cancel  ")
        self._btn_cancel.setObjectName("dangerButton")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self.request_cancel.emit)

        self._btn_clear = QPushButton("Clear Queue")
        self._btn_clear.clicked.connect(self._on_clear)

        btn_row.addWidget(self._btn_start)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

    def _pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_dir = d
            self._out_path_label.setText(d)
            self.output_dir_changed.emit(d)

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def _on_clear(self):
        self.request_clear.emit()

    def _on_row_clicked(self, row: int, column: int):
        """Emit job_selected when user clicks a row."""
        item = self._table.item(row, 0)
        if item:
            try:
                job_id = int(item.text())
                self.job_selected.emit(job_id)
            except ValueError:
                pass

    def _clear_table(self):
        self._table.setRowCount(0)
        self._overall_progress.setValue(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_job(self, job: JobItem):
        """Add a row to the queue table."""
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QTableWidgetItem(str(job.id)))
        self._table.setItem(row, 1, QTableWidgetItem(Path(job.input_path).name))

        status_item = QTableWidgetItem(job.status.value.capitalize())
        self._table.setItem(row, 2, status_item)

        progress = QProgressBar()
        progress.setFixedHeight(18)
        progress.setValue(0)
        self._table.setCellWidget(row, 3, progress)

        self._table.setItem(row, 4, QTableWidgetItem("—"))

    def update_job_progress(self, job_id: int, progress: float, message: str):
        """Update progress bar for a specific job."""
        row = self._find_row(job_id)
        if row < 0:
            return
        bar: QProgressBar = self._table.cellWidget(row, 3)
        if bar:
            bar.setValue(int(progress * 100))
            bar.setFormat(f"{message}  {int(progress*100)}%")

    def update_job_status(self, job_id: int, status: JobStatus,
                          time_s: float = 0):
        """Update status column for a job."""
        row = self._find_row(job_id)
        if row < 0:
            return

        status_item = self._table.item(row, 2)
        status_item.setText(status.value.capitalize())

        colors = {
            JobStatus.QUEUED: TEXT_SECONDARY,
            JobStatus.PROCESSING: ACCENT,
            JobStatus.COMPLETED: SUCCESS,
            JobStatus.FAILED: ERROR,
            JobStatus.CANCELLED: WARNING,
        }
        status_item.setForeground(QColor(colors.get(status, TEXT_SECONDARY)))

        if time_s > 0:
            self._table.item(row, 4).setText(f"{time_s:.1f}s")

    def update_overall_progress(self, progress: float):
        self._overall_progress.setValue(int(progress * 100))

    def update_stats(self, queued: int, completed: int, failed: int,
                     total_time: float):
        self._card_queued.set_value(str(queued))
        self._card_completed.set_value(str(completed))
        self._card_failed.set_value(str(failed))
        self._card_time.set_value(f"{total_time:.1f}s")

    def set_running(self, running: bool):
        self._btn_start.setEnabled(not running)
        self._btn_cancel.setEnabled(running)
        self._btn_clear.setEnabled(not running)

    def _find_row(self, job_id: int) -> int:
        for r in range(self._table.rowCount()):
            item = self._table.item(r, 0)
            if item and item.text() == str(job_id):
                return r
        return -1
