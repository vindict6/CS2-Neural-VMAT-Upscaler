"""
Texture I/O module – handles loading, saving, and format conversion
for all major game-engine texture formats.

Supported formats:
  PNG, JPEG, TGA, BMP, TIFF, DDS (via imageio/Pillow), WebP, EXR (HDR).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger("CS2Upscaler.TextureIO")

# Ensure Pillow loads TGA and other formats
PILImage.init()


class TextureFormat(Enum):
    """Supported texture file formats."""
    PNG = "png"
    JPEG = "jpeg"
    TGA = "tga"
    BMP = "bmp"
    TIFF = "tiff"
    WEBP = "webp"
    DDS = "dds"
    EXR = "exr"


EXTENSION_MAP = {
    ".png": TextureFormat.PNG,
    ".jpg": TextureFormat.JPEG,
    ".jpeg": TextureFormat.JPEG,
    ".tga": TextureFormat.TGA,
    ".bmp": TextureFormat.BMP,
    ".tif": TextureFormat.TIFF,
    ".tiff": TextureFormat.TIFF,
    ".webp": TextureFormat.WEBP,
    ".dds": TextureFormat.DDS,
    ".exr": TextureFormat.EXR,
}

SAVE_EXTENSIONS = {
    TextureFormat.PNG: ".png",
    TextureFormat.JPEG: ".jpg",
    TextureFormat.TGA: ".tga",
    TextureFormat.BMP: ".bmp",
    TextureFormat.TIFF: ".tiff",
    TextureFormat.WEBP: ".webp",
}


@dataclass
class TextureInfo:
    """Metadata about a loaded texture."""
    path: str
    filename: str
    width: int
    height: int
    channels: int
    has_alpha: bool
    bit_depth: int
    file_size_bytes: int
    format: TextureFormat
    is_power_of_two: bool
    color_space: str  # "sRGB" or "linear"


def detect_format(path: str) -> Optional[TextureFormat]:
    """Detect texture format from file extension."""
    ext = Path(path).suffix.lower()
    return EXTENSION_MAP.get(ext)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def load_texture(path: str) -> Tuple[np.ndarray, TextureInfo]:
    """
    Load a texture file and return (image_array, info).

    image_array is uint8 (H, W, C) with C = 3 (RGB) or 4 (RGBA).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Texture not found: {path}")

    fmt = detect_format(path)
    file_size = p.stat().st_size

    if fmt == TextureFormat.DDS:
        image = _load_dds(path)
    elif fmt == TextureFormat.EXR:
        image = _load_exr(path)
    else:
        image = _load_standard(path)

    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    has_alpha = channels == 4
    bit_depth = 8  # We normalise everything to uint8 for the pipeline

    info = TextureInfo(
        path=str(p.resolve()),
        filename=p.name,
        width=w,
        height=h,
        channels=channels,
        has_alpha=has_alpha,
        bit_depth=bit_depth,
        file_size_bytes=file_size,
        format=fmt or TextureFormat.PNG,
        is_power_of_two=is_power_of_two(w) and is_power_of_two(h),
        color_space="sRGB",
    )

    logger.info(f"Loaded: {p.name}  {w}x{h}  {'RGBA' if has_alpha else 'RGB'}")
    return image, info


def _load_standard(path: str) -> np.ndarray:
    """Load using Pillow (PNG, JPEG, TGA, BMP, TIFF, WebP)."""
    img = PILImage.open(path)
    if img.mode == "P":
        img = img.convert("RGBA")
    elif img.mode == "L":
        img = img.convert("RGB")
    elif img.mode == "LA":
        img = img.convert("RGBA")
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
    return np.array(img)


def _load_dds(path: str) -> np.ndarray:
    """Load DDS textures. Attempts imageio-based DDS loading, falls back to Pillow."""
    try:
        import imageio.v3 as iio
        data = iio.imread(path)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        return data.astype(np.uint8)
    except Exception:
        logger.warning("DDS imageio failed, trying Pillow DDS plugin...")
        return _load_standard(path)


def _load_exr(path: str) -> np.ndarray:
    """Load EXR (HDR) and tone-map to uint8."""
    try:
        import imageio.v3 as iio
        hdr = iio.imread(path).astype(np.float32)
        # Simple Reinhard tone mapping
        mapped = hdr / (hdr + 1.0)
        return (np.clip(mapped, 0.0, 1.0) * 255).astype(np.uint8)
    except Exception as e:
        raise IOError(f"Failed to load EXR: {e}")


def save_texture(image: np.ndarray, path: str, fmt: TextureFormat = TextureFormat.PNG,
                 quality: int = 100):
    """
    Save the texture array to disk in the requested format.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == TextureFormat.EXR:
        _save_exr(image, path)
        return
    if fmt == TextureFormat.DDS:
        logger.warning("DDS saving not supported – falling back to PNG.")
        fmt = TextureFormat.PNG
        p = p.with_suffix(".png")

    pil = PILImage.fromarray(image)

    save_kwargs = {}
    if fmt == TextureFormat.JPEG:
        if pil.mode == "RGBA":
            pil = pil.convert("RGB")
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = 0  # 4:4:4 for texture quality
    elif fmt == TextureFormat.PNG:
        save_kwargs["compress_level"] = max(0, min(9, (100 - quality) // 11))
    elif fmt == TextureFormat.WEBP:
        save_kwargs["quality"] = quality
        save_kwargs["lossless"] = quality >= 100
    elif fmt == TextureFormat.TGA:
        # Pillow TGA supports RLE compression
        save_kwargs["compression"] = "tga_rle"

    pil.save(str(p), **save_kwargs)
    logger.info(f"Saved: {p.name}  ({fmt.value})")


def _save_exr(image: np.ndarray, path: str):
    """Save as EXR (float32)."""
    try:
        import imageio.v3 as iio
        data = image.astype(np.float32) / 255.0
        iio.imwrite(path, data)
    except Exception as e:
        raise IOError(f"Failed to save EXR: {e}")


def get_supported_load_extensions() -> list:
    """Return list of loadable file extensions."""
    return list(EXTENSION_MAP.keys())


def get_supported_save_formats() -> list:
    """Return list of saveable format names."""
    return [f.value for f in SAVE_EXTENSIONS.keys()]


def build_file_filter() -> str:
    """Build a Qt-style file filter string for open dialogs."""
    all_exts = " ".join(f"*{ext}" for ext in EXTENSION_MAP.keys())
    parts = [f"All Textures ({all_exts})"]
    parts.append(f"PNG (*.png)")
    parts.append(f"JPEG (*.jpg *.jpeg)")
    parts.append(f"TGA (*.tga)")
    parts.append(f"BMP (*.bmp)")
    parts.append(f"TIFF (*.tif *.tiff)")
    parts.append(f"WebP (*.webp)")
    parts.append(f"DDS (*.dds)")
    parts.append(f"EXR (*.exr)")
    parts.append(f"All Files (*.*)")
    return ";;".join(parts)
