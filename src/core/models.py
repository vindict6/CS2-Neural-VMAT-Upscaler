"""
Model management – downloading, caching, listing available models,
and managing model metadata.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from .upscaler import ModelVariant

logger = logging.getLogger("CS2Upscaler.Models")


@dataclass
class ModelInfo:
    """Metadata for an available model."""
    name: str
    variant: ModelVariant
    filename: str
    description: str
    scale: int
    file_size_mb: float
    downloaded: bool = False
    recommended_for: str = ""


# Built-in model registry
MODEL_REGISTRY: List[ModelInfo] = [
    ModelInfo(
        name="Real-ESRGAN x4+",
        variant=ModelVariant.REALESRGAN_X4PLUS,
        filename="RealESRGAN_x4plus.pth",
        description="General-purpose 4x upscaler. Best for diffuse/albedo textures.",
        scale=4,
        file_size_mb=63.0,
        recommended_for="Diffuse, Albedo, Emissive",
    ),
    ModelInfo(
        name="Real-ESRGAN x4+ Anime",
        variant=ModelVariant.REALESRGAN_X4PLUS_ANIME,
        filename="RealESRGAN_x4plus_anime_6B.pth",
        description="Optimised for anime-style and stylised textures. Lighter model.",
        scale=4,
        file_size_mb=17.0,
        recommended_for="Stylised, Cartoon, Hand-painted",
    ),
    ModelInfo(
        name="Real-ESRGAN x2+",
        variant=ModelVariant.REALESRGAN_X2PLUS,
        filename="RealESRGAN_x2plus.pth",
        description="Native 2x upscaler. Better quality vs running 4x at 2x.",
        scale=2,
        file_size_mb=63.0,
        recommended_for="Subtle upscaling, Detail textures",
    ),
    ModelInfo(
        name="Real-ESRNet x4+",
        variant=ModelVariant.REALESRNET_X4PLUS,
        filename="RealESRNet_x4plus.pth",
        description="PSNR-oriented model. Sharper but less perceptual quality.",
        scale=4,
        file_size_mb=63.0,
        recommended_for="Normal maps, Height maps, Data textures",
    ),
]


class ModelManager:
    """Manages model files and metadata."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self.models_dir / "model_cache.json"
        self._refresh_download_status()

    def _refresh_download_status(self):
        """Check which models exist on disk."""
        for info in MODEL_REGISTRY:
            path = self.models_dir / info.filename
            info.downloaded = path.exists()

    def list_models(self) -> List[ModelInfo]:
        """Return all available models with download status."""
        self._refresh_download_status()
        return MODEL_REGISTRY

    def get_model_info(self, variant: ModelVariant) -> Optional[ModelInfo]:
        """Get info for a specific model variant."""
        for info in MODEL_REGISTRY:
            if info.variant == variant:
                return info
        return None

    def is_downloaded(self, variant: ModelVariant) -> bool:
        """Check if a model is already downloaded."""
        info = self.get_model_info(variant)
        if info is None:
            return False
        return (self.models_dir / info.filename).exists()

    def get_model_path(self, variant: ModelVariant) -> Optional[Path]:
        """Get the path to a downloaded model."""
        info = self.get_model_info(variant)
        if info is None:
            return None
        path = self.models_dir / info.filename
        return path if path.exists() else None

    def get_models_disk_usage(self) -> float:
        """Total disk usage of all downloaded models in MB."""
        total = 0
        for f in self.models_dir.glob("*.pth"):
            total += f.stat().st_size
        return total / (1024 * 1024)

    def delete_model(self, variant: ModelVariant) -> bool:
        """Remove a downloaded model from disk."""
        info = self.get_model_info(variant)
        if info is None:
            return False
        path = self.models_dir / info.filename
        if path.exists():
            path.unlink()
            logger.info(f"Deleted model: {info.filename}")
            self._refresh_download_status()
            return True
        return False

    def get_recommended_model(self, texture_type: str) -> ModelVariant:
        """Suggest the best model for a given texture type."""
        texture_type = texture_type.lower()
        if texture_type in ("normal", "height", "ao"):
            return ModelVariant.REALESRNET_X4PLUS
        if texture_type in ("stylised", "cartoon", "anime"):
            return ModelVariant.REALESRGAN_X4PLUS_ANIME
        return ModelVariant.REALESRGAN_X4PLUS
