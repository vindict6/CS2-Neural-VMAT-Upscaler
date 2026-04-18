"""
Batch processing pipeline with queue management, sequential GPU execution,
progress tracking, and robust error handling.

Supports recursive folder upscaling with preserved directory structure.
"""

import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot, QMetaObject, Qt, Q_ARG

from .texture_io import (
    TextureFormat,
    load_texture,
    save_texture,
    detect_format,
    SAVE_EXTENSIONS,
)
from .upscaler import TextureUpscaler, UpscaleResult, UpscaleSettings

logger = logging.getLogger("CS2Upscaler.Pipeline")


class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobItem:
    """A single processing job in the batch queue."""
    id: int
    input_path: str
    output_path: str
    settings: UpscaleSettings
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    result: Optional[UpscaleResult] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class WorkerSignals(QObject):
    """Signals emitted by processing workers."""
    job_started = pyqtSignal(int)           # job_id
    job_progress = pyqtSignal(int, float, str)  # job_id, progress, message
    job_completed = pyqtSignal(int, object)  # job_id, UpscaleResult
    job_failed = pyqtSignal(int, str)       # job_id, error_message
    batch_completed = pyqtSignal()
    batch_progress = pyqtSignal(float)      # overall progress


class UpscaleWorker(QRunnable):
    """Worker that processes a single upscale job in the thread pool."""

    def __init__(self, job: JobItem, upscaler: TextureUpscaler):
        super().__init__()
        self.setAutoDelete(False)  # prevent C++ deletion while Python refs exist
        self.job = job
        self.upscaler = upscaler
        self.signals = WorkerSignals()
        self._cancelled = False

    @pyqtSlot()
    def run(self):
        if self._cancelled:
            return

        job = self.job
        self.signals.job_started.emit(job.id)

        try:
            job.status = JobStatus.PROCESSING

            # Load texture
            self.signals.job_progress.emit(job.id, 0.05, "Loading texture...")
            image, info = load_texture(job.input_path)

            if self._cancelled:
                job.status = JobStatus.CANCELLED
                return

            # Progress callback wrapper
            def on_progress(pct, msg):
                scaled = 0.05 + pct * 0.85
                self.signals.job_progress.emit(job.id, scaled, msg)

            # Upscale
            result = self.upscaler.upscale(image, job.settings,
                                           progress_callback=on_progress)

            if self._cancelled:
                job.status = JobStatus.CANCELLED
                return

            # Determine output format and ensure output directory exists
            self.signals.job_progress.emit(job.id, 0.92, "Saving texture...")
            Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
            out_fmt = detect_format(job.output_path) or TextureFormat.PNG

            # Use compression quality if enabled, otherwise raw quality
            quality = job.settings.output_quality
            if job.settings.compression_enabled:
                quality = job.settings.compression_quality

            save_texture(result.image, job.output_path, fmt=out_fmt,
                         quality=quality)

            # Save PBR maps if generated
            if job.settings.pbr_generate:
                self._save_pbr_maps(result, job.output_path, job.input_path,
                                    job.settings)

            job.result = result
            job.status = JobStatus.COMPLETED
            job.processing_time = result.processing_time

            # Save mipmaps if generated
            if result.mipmaps:
                self._save_mipmaps(result.mipmaps, job.output_path, out_fmt,
                                   job.settings.output_quality)

            self.signals.job_progress.emit(job.id, 1.0, "Complete")
            self.signals.job_completed.emit(job.id, result)

            # Release CUDA cache between jobs to prevent VRAM buildup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error(f"Job {job.id} failed: {e}\n{traceback.format_exc()}")
            self.signals.job_failed.emit(job.id, str(e))

    def _save_mipmaps(self, mipmaps: list, base_path: str,
                      fmt: TextureFormat, quality: int):
        """Save mipmap chain alongside the main output."""
        p = Path(base_path)
        for i, mip in enumerate(mipmaps):
            mip_path = p.parent / f"{p.stem}_mip{i+1}{p.suffix}"
            save_texture(mip, str(mip_path), fmt=fmt, quality=quality)

    def _save_pbr_maps(self, result: UpscaleResult, output_path: str,
                       input_path: str, settings: UpscaleSettings):
        """Save generated PBR maps as separate files alongside the upscaled output."""
        try:
            p = Path(output_path)
            pbr_fmt = detect_format(output_path) or TextureFormat.PNG
            quality = settings.output_quality
            if settings.compression_enabled:
                quality = settings.compression_quality

            if result.pbr_roughness_map is not None:
                rough_path = str(p.parent / f"{p.stem}_rough{p.suffix}")
                save_texture(result.pbr_roughness_map, rough_path,
                             fmt=pbr_fmt, quality=quality)
                logger.info(f"Saved roughness map: {rough_path}")

            if result.pbr_metalness_map is not None:
                metal_path = str(p.parent / f"{p.stem}_metal{p.suffix}")
                save_texture(result.pbr_metalness_map, metal_path,
                             fmt=pbr_fmt, quality=quality)
                logger.info(f"Saved metalness map: {metal_path}")

            if result.pbr_normal_map is not None:
                normal_path = str(p.parent / f"{p.stem}_normal{p.suffix}")
                save_texture(result.pbr_normal_map, normal_path,
                             fmt=pbr_fmt, quality=quality)
                logger.info(f"Saved normal map: {normal_path}")

            # Update VMAT if requested
            if settings.pbr_update_vmat:
                self._update_vmat(input_path, result, p.stem, p)

        except Exception as e:
            logger.warning(f"PBR map save error: {e}")

    def _update_vmat(self, input_path: str, result: UpscaleResult,
                     base_name: str, output_p: Path):
        """Find and update the associated VMAT file with new PBR values."""
        try:
            from .vmat_parser import parse_vmat, write_vmat
            # Look for a VMAT that references this texture
            input_p = Path(input_path)
            vmat_dir = input_p.parent
            vmat_files = list(vmat_dir.glob("*.vmat"))

            for vf in vmat_files:
                mat = parse_vmat(str(vf))
                if not mat:
                    continue
                # Check if any texture in this VMAT matches our input
                input_name = input_p.stem.lower()
                refs_this = False
                for tex in mat.textures:
                    if input_name in tex.path.lower():
                        refs_this = True
                        break
                if not refs_this:
                    continue

                # Build changes dict
                changes = {}
                # Convert output paths to materials-relative for VMAT
                rel_base = str(output_p.parent / base_name).replace("\\", "/")

                if result.pbr_roughness_map is not None:
                    changes["TextureRoughness"] = f"{rel_base}_rough{output_p.suffix}"
                if result.pbr_normal_map is not None:
                    changes["TextureNormal"] = f"{rel_base}_normal{output_p.suffix}"
                if result.pbr_global_metalness > 0.01:
                    changes["g_flMetalness"] = f"{result.pbr_global_metalness:.6f}"
                if result.pbr_global_roughness > 0.01:
                    changes["g_flRoughness"] = f"{result.pbr_global_roughness:.6f}"

                if changes:
                    write_vmat(mat, changes, str(vf))
                    logger.info(f"Updated VMAT: {vf.name} with {len(changes)} changes")
                break  # Only update the first matching VMAT

        except Exception as e:
            logger.warning(f"VMAT update error: {e}")

    def cancel(self):
        self._cancelled = True


class ProcessingPipeline(QObject):
    """
    Manages a queue of upscale jobs with sequential or parallel execution.

    Signals provide real-time progress updates for the UI.
    """

    # Re-export signals at pipeline level for convenience
    job_started = pyqtSignal(int)
    job_progress = pyqtSignal(int, float, str)
    job_completed = pyqtSignal(int, object)
    job_failed = pyqtSignal(int, str)
    batch_completed = pyqtSignal()
    batch_progress = pyqtSignal(float)
    pipeline_status = pyqtSignal(str)

    def __init__(self, upscaler: TextureUpscaler, parent=None):
        super().__init__(parent)
        self.upscaler = upscaler
        self._jobs: Dict[int, JobItem] = {}
        self._workers: Dict[int, UpscaleWorker] = {}
        self._next_id = 1
        self._is_running = False
        self._cancelled = False
        self._batch_total = 0
        self._batch_done = 0
        self._thread_pool = QThreadPool.globalInstance()
        self._thread_pool.setMaxThreadCount(1)  # Sequential by default for GPU

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def add_job(self, input_path: str, output_path: str,
                settings: UpscaleSettings) -> int:
        """Add a job to the processing queue. Returns job ID."""
        job_id = self._next_id
        self._next_id += 1
        job = JobItem(
            id=job_id,
            input_path=input_path,
            output_path=output_path,
            settings=settings,
        )
        self._jobs[job_id] = job
        logger.info(f"Queued job #{job_id}: {Path(input_path).name}")
        return job_id

    def add_batch(self, input_paths: List[str], output_dir: str,
                  settings: UpscaleSettings,
                  output_format: Optional[TextureFormat] = None,
                  input_root: str = "") -> List[int]:
        """
        Add multiple files to the queue.

        If input_root is set, output paths preserve the relative directory
        structure from input_root inside output_dir (recursive folder mode).
        Otherwise output filenames are derived from input names.
        """
        ids = []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for inp in input_paths:
            p = Path(inp)
            ext = p.suffix or ".png"
            if output_format:
                ext = SAVE_EXTENSIONS.get(output_format, ".png")

            if input_root:
                # Preserve relative directory structure
                try:
                    rel = p.relative_to(input_root)
                except ValueError:
                    rel = Path(p.name)
                # Keep exact same filename (no _upscaled suffix)
                out_path = str(out_dir / rel.parent / f"{rel.stem}{ext}")
            else:
                out_name = f"{p.stem}_upscaled{ext}"
                out_path = str(out_dir / out_name)

            ids.append(self.add_job(inp, out_path, settings))

        return ids

    def remove_job(self, job_id: int):
        """Remove a queued (not started) job."""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.QUEUED:
            del self._jobs[job_id]

    def clear_queue(self):
        """Remove all queued (not started) jobs."""
        to_remove = [jid for jid, j in self._jobs.items()
                     if j.status == JobStatus.QUEUED]
        for jid in to_remove:
            del self._jobs[jid]

    def get_job(self, job_id: int) -> Optional[JobItem]:
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[JobItem]:
        return list(self._jobs.values())

    def get_queue_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)

    def get_completed_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def start(self):
        """Start processing all queued jobs."""
        if self._is_running:
            return

        self._is_running = True
        self._cancelled = False

        # Disconnect old workers to prevent stale signal interference
        self._disconnect_old_workers()

        self.pipeline_status.emit("Processing...")

        queued = [j for j in self._jobs.values() if j.status == JobStatus.QUEUED]
        self._batch_total = len(queued)
        self._batch_done = 0

        for job in queued:
            if self._cancelled:
                break

            worker = UpscaleWorker(job, self.upscaler)
            self._workers[job.id] = worker

            # Wire signals with QueuedConnection for thread safety
            _Q = Qt.ConnectionType.QueuedConnection
            worker.signals.job_started.connect(self.job_started.emit, _Q)
            worker.signals.job_progress.connect(self._on_job_progress, _Q)
            worker.signals.job_progress.connect(self.job_progress.emit, _Q)
            worker.signals.job_completed.connect(self._on_worker_completed, _Q)
            worker.signals.job_failed.connect(self._on_worker_failed, _Q)

            self._thread_pool.start(worker)

    def _on_worker_completed(self, job_id: int, result):
        """Forward job completion and check for batch done."""
        try:
            self.job_completed.emit(job_id, result)
        except Exception:
            logger.exception(f"Error in job_completed handler for #{job_id}")
        self._check_batch_done()

    def _on_worker_failed(self, job_id: int, error: str):
        """Forward job failure and check for batch done."""
        try:
            self.job_failed.emit(job_id, error)
        except Exception:
            logger.exception(f"Error in job_failed handler for #{job_id}")
        self._check_batch_done()

    def _check_batch_done(self):
        """Increment counter and emit batch_completed when all jobs finish."""
        if self._cancelled or not self._is_running:
            return
        self._batch_done += 1
        if self._batch_done >= self._batch_total:
            self._is_running = False
            self.pipeline_status.emit("Batch complete")
            try:
                self.batch_completed.emit()
            except Exception:
                logger.exception("Error in batch_completed handler")

    def _on_job_progress(self, job_id: int, progress: float, message: str):
        """Update overall batch progress."""
        total = len(self._workers)
        if total == 0:
            return
        done = sum(1 for j in self._jobs.values()
                   if j.status in (JobStatus.COMPLETED, JobStatus.FAILED))
        overall = (done + progress) / total
        self.batch_progress.emit(overall)

    def cancel(self):
        """Cancel all running and queued jobs."""
        self._cancelled = True
        self._disconnect_old_workers()
        for job in self._jobs.values():
            if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                job.status = JobStatus.CANCELLED
        self._is_running = False
        self.pipeline_status.emit("Cancelled")

    def _disconnect_old_workers(self):
        """Disconnect signals from old workers so they can't interfere."""
        for worker in self._workers.values():
            worker.cancel()
            try:
                worker.signals.job_started.disconnect()
                worker.signals.job_progress.disconnect()
                worker.signals.job_completed.disconnect()
                worker.signals.job_failed.disconnect()
            except (TypeError, RuntimeError):
                pass
        self._workers.clear()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def reset(self):
        """Clear all jobs and reset state."""
        self.cancel()
        self._disconnect_old_workers()
        self._jobs.clear()
        self._workers.clear()
        self._batch_done = 0
        self._batch_total = 0
        self._next_id = 1
        self._cancelled = False
