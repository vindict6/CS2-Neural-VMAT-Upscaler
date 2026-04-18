"""
PBR map generation for CS2 textures.

Generates roughness, metalness, and normal maps from colour textures
using material-aware heuristics and image-processing techniques.
Designed for Source 2 / CS2 VMAT material workflows.

Capabilities:
  1. AI roughness estimation — analyses texture detail to produce a
     roughness map appropriate for the detected material type.
  2. AI metalness estimation — classifies metallic regions and generates
     a metalness mask.
  3. Normal map generation — converts colour/height information into
     tangent-space normal maps.
  4. Material-type-aware defaults for CS2 PBR parameters.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("CS2Upscaler.PBRGenerator")


# ══════════════════════════════════════════════════════════════════════
# Material PBR profiles for CS2
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PBRProfile:
    """Material-specific PBR parameter ranges for CS2."""
    roughness_base: float       # Typical base roughness (0.0-1.0)
    roughness_range: float      # How much roughness varies across surface
    metalness_base: float       # Typical metalness (0.0 or 1.0 for PBR)
    metalness_range: float      # Variation in metalness
    normal_strength: float      # Normal map intensity multiplier
    detail_frequency: float     # Expected surface detail frequency
    # CS2-specific
    cubemap_scalar: float       # Typical g_flCubeMapScalar
    specular_indirect: bool     # Whether F_SPECULAR_INDIRECT should be on


# Profiles tuned for CS2 rendering
PBR_PROFILES = {
    "brick": PBRProfile(
        roughness_base=0.85, roughness_range=0.15,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=1.2, detail_frequency=0.6,
        cubemap_scalar=0.0, specular_indirect=False),
    "concrete": PBRProfile(
        roughness_base=0.90, roughness_range=0.10,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.8, detail_frequency=0.4,
        cubemap_scalar=0.0, specular_indirect=False),
    "stone_rough": PBRProfile(
        roughness_base=0.88, roughness_range=0.12,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=1.4, detail_frequency=0.7,
        cubemap_scalar=0.0, specular_indirect=False),
    "stone_polished": PBRProfile(
        roughness_base=0.25, roughness_range=0.15,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.3, detail_frequency=0.2,
        cubemap_scalar=0.3, specular_indirect=True),
    "metal_bare": PBRProfile(
        roughness_base=0.35, roughness_range=0.25,
        metalness_base=0.95, metalness_range=0.05,
        normal_strength=0.6, detail_frequency=0.3,
        cubemap_scalar=0.4, specular_indirect=True),
    "metal_painted": PBRProfile(
        roughness_base=0.55, roughness_range=0.20,
        metalness_base=0.0, metalness_range=0.3,
        normal_strength=0.5, detail_frequency=0.4,
        cubemap_scalar=0.15, specular_indirect=True),
    "metal_rusted": PBRProfile(
        roughness_base=0.75, roughness_range=0.20,
        metalness_base=0.6, metalness_range=0.35,
        normal_strength=1.0, detail_frequency=0.7,
        cubemap_scalar=0.1, specular_indirect=True),
    "metal_brushed": PBRProfile(
        roughness_base=0.30, roughness_range=0.15,
        metalness_base=0.95, metalness_range=0.05,
        normal_strength=0.4, detail_frequency=0.5,
        cubemap_scalar=0.5, specular_indirect=True),
    "wood_raw": PBRProfile(
        roughness_base=0.75, roughness_range=0.20,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.8, detail_frequency=0.5,
        cubemap_scalar=0.0, specular_indirect=False),
    "wood_finished": PBRProfile(
        roughness_base=0.35, roughness_range=0.15,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.5, detail_frequency=0.3,
        cubemap_scalar=0.2, specular_indirect=True),
    "wood_weathered": PBRProfile(
        roughness_base=0.85, roughness_range=0.15,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=1.0, detail_frequency=0.6,
        cubemap_scalar=0.0, specular_indirect=False),
    "tile_ceramic": PBRProfile(
        roughness_base=0.30, roughness_range=0.20,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.6, detail_frequency=0.4,
        cubemap_scalar=0.25, specular_indirect=True),
    "glass": PBRProfile(
        roughness_base=0.05, roughness_range=0.10,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.1, detail_frequency=0.1,
        cubemap_scalar=1.0, specular_indirect=True),
    "plastic": PBRProfile(
        roughness_base=0.45, roughness_range=0.15,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.3, detail_frequency=0.3,
        cubemap_scalar=0.15, specular_indirect=True),
    "rubber": PBRProfile(
        roughness_base=0.92, roughness_range=0.08,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.5, detail_frequency=0.4,
        cubemap_scalar=0.0, specular_indirect=False),
    "fabric_woven": PBRProfile(
        roughness_base=0.90, roughness_range=0.10,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.4, detail_frequency=0.6,
        cubemap_scalar=0.0, specular_indirect=False),
    "fabric_leather": PBRProfile(
        roughness_base=0.65, roughness_range=0.20,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.6, detail_frequency=0.5,
        cubemap_scalar=0.05, specular_indirect=True),
    "carpet": PBRProfile(
        roughness_base=0.95, roughness_range=0.05,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.3, detail_frequency=0.7,
        cubemap_scalar=0.0, specular_indirect=False),
    "foliage": PBRProfile(
        roughness_base=0.70, roughness_range=0.20,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.5, detail_frequency=0.5,
        cubemap_scalar=0.0, specular_indirect=False),
    "terrain_dirt": PBRProfile(
        roughness_base=0.92, roughness_range=0.08,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=1.0, detail_frequency=0.5,
        cubemap_scalar=0.0, specular_indirect=False),
    "terrain_sand": PBRProfile(
        roughness_base=0.88, roughness_range=0.12,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.7, detail_frequency=0.4,
        cubemap_scalar=0.0, specular_indirect=False),
    "terrain_gravel": PBRProfile(
        roughness_base=0.90, roughness_range=0.10,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=1.3, detail_frequency=0.6,
        cubemap_scalar=0.0, specular_indirect=False),
    "plaster": PBRProfile(
        roughness_base=0.85, roughness_range=0.10,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.5, detail_frequency=0.3,
        cubemap_scalar=0.0, specular_indirect=False),
    "asphalt": PBRProfile(
        roughness_base=0.88, roughness_range=0.12,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.8, detail_frequency=0.5,
        cubemap_scalar=0.0, specular_indirect=False),
    "generic": PBRProfile(
        roughness_base=0.70, roughness_range=0.20,
        metalness_base=0.0, metalness_range=0.0,
        normal_strength=0.8, detail_frequency=0.4,
        cubemap_scalar=0.0, specular_indirect=False),
}


def get_pbr_profile(material_type: str) -> PBRProfile:
    """Get PBR profile for a material type, with fallback to generic."""
    return PBR_PROFILES.get(material_type, PBR_PROFILES["generic"])


# ══════════════════════════════════════════════════════════════════════
# Roughness Map Generation
# ══════════════════════════════════════════════════════════════════════

def generate_roughness_map(
    color_image: np.ndarray,
    material_type: str = "generic",
    strength: float = 1.0,
) -> np.ndarray:
    """
    Generate a roughness map from a colour texture.

    Uses multi-scale detail analysis:
    - High-frequency detail → rough (porous, textured surfaces)
    - Smooth gradients → smooth (polished, glossy surfaces)
    - Material profile adjusts the baseline and range

    Parameters
    ----------
    color_image : np.ndarray
        RGB uint8 colour texture.
    material_type : str
        Material type key (matches PBR_PROFILES).
    strength : float
        How aggressively to deviate from the base roughness (0-2).

    Returns
    -------
    np.ndarray : Grayscale uint8 roughness map (0=smooth, 255=rough).
    """
    profile = get_pbr_profile(material_type)

    # Convert to grayscale float
    if color_image.ndim == 3:
        gray = cv2.cvtColor(color_image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = color_image.copy()
    gray_f = gray.astype(np.float32) / 255.0

    h, w = gray_f.shape

    # Multi-scale detail extraction
    # Fine detail → rougher
    blur_fine = cv2.GaussianBlur(gray_f, (3, 3), 0.8)
    detail_fine = np.abs(gray_f - blur_fine)

    # Medium detail
    blur_med = cv2.GaussianBlur(gray_f, (9, 9), 2.0)
    detail_med = np.abs(gray_f - blur_med)

    # Coarse detail
    blur_coarse = cv2.GaussianBlur(gray_f, (21, 21), 5.0)
    detail_coarse = np.abs(gray_f - blur_coarse)

    # Combine scales with material-tuned weights
    freq = profile.detail_frequency
    detail_map = (
        detail_fine * (0.5 + freq * 0.3) +
        detail_med * (0.3 + freq * 0.2) +
        detail_coarse * (0.2 - freq * 0.1)
    )

    # Normalise to 0-1
    d_min, d_max = detail_map.min(), detail_map.max()
    if d_max - d_min > 1e-6:
        detail_map = (detail_map - d_min) / (d_max - d_min)
    else:
        detail_map = np.full_like(detail_map, 0.5)

    # Edge detection for roughness variation (edges tend to be rougher)
    edges = cv2.Canny(gray, 30, 100).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (7, 7), 2.0)

    # Luminance-based component (darker areas tend to be rougher for many materials)
    # Inverted for metals (shinier darker areas)
    lum_component = 1.0 - gray_f  # dark = more rough
    if profile.metalness_base > 0.5:
        lum_component = gray_f  # metals: bright = more rough (oxidation in dark areas)

    # Combine into roughness
    roughness = (
        profile.roughness_base +
        (detail_map - 0.5) * profile.roughness_range * 2.0 * strength +
        edges * profile.roughness_range * 0.3 * strength +
        (lum_component - 0.5) * profile.roughness_range * 0.4 * strength
    )

    roughness = np.clip(roughness, 0.0, 1.0)

    # Convert to uint8
    roughness_map = (roughness * 255).astype(np.uint8)

    logger.info(f"Roughness map generated: {material_type}, "
                f"mean={roughness.mean():.2f}, range=[{roughness.min():.2f}, {roughness.max():.2f}]")
    return roughness_map


# ══════════════════════════════════════════════════════════════════════
# Metalness Map Generation
# ══════════════════════════════════════════════════════════════════════

def generate_metalness_map(
    color_image: np.ndarray,
    material_type: str = "generic",
    strength: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Generate a metalness map and global metalness value from a colour texture.

    Metal detection based on:
    - High saturation in warm tones (copper, gold, bronze)
    - Low saturation with high value (chrome, steel)
    - Material profile base metalness

    Parameters
    ----------
    color_image : np.ndarray
        RGB uint8 colour texture.
    material_type : str
        Material type key.
    strength : float
        Detection sensitivity (0-2).

    Returns
    -------
    (metalness_map, global_metalness)
        metalness_map: Grayscale uint8 (0=dielectric, 255=metal)
        global_metalness: Float 0.0-1.0 for g_flMetalness in VMAT
    """
    profile = get_pbr_profile(material_type)

    if profile.metalness_base < 0.01 and profile.metalness_range < 0.01:
        # Non-metallic material — skip expensive analysis
        h, w = color_image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), 0.0

    rgb = color_image[:, :, :3] if color_image.ndim == 3 else np.stack([color_image]*3, axis=-1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_chan = hsv[:, :, 0]  # 0-180
    s_chan = hsv[:, :, 1] / 255.0
    v_chan = hsv[:, :, 2] / 255.0

    # Metal colour heuristics
    # Warm metals (gold, copper, bronze): hue 5-35, saturation 0.3-0.8
    warm_metal = (
        ((h_chan >= 5) & (h_chan <= 35)).astype(np.float32) *
        np.clip((s_chan - 0.2) / 0.4, 0, 1) *
        np.clip(v_chan / 0.6, 0, 1)
    )

    # Cool metals (steel, chrome, aluminum): low saturation, medium-high value
    cool_metal = (
        np.clip(1.0 - s_chan * 3, 0, 1) *
        np.clip((v_chan - 0.3) / 0.4, 0, 1)
    )

    # Dark metals (iron, cast): very low sat, medium value
    dark_metal = (
        np.clip(1.0 - s_chan * 4, 0, 1) *
        np.clip(v_chan / 0.5, 0, 1) *
        np.clip(1.0 - (v_chan - 0.5).clip(0) * 3, 0, 1)
    )

    # Combine with profile weighting
    metal_score = np.maximum(warm_metal, np.maximum(cool_metal * 0.7, dark_metal * 0.5))
    metal_score *= strength

    # Blend with profile base
    metalness = profile.metalness_base + metal_score * profile.metalness_range
    metalness = np.clip(metalness, 0.0, 1.0)

    # Global value for VMAT g_flMetalness
    global_metalness = float(metalness.mean())

    metalness_map = (metalness * 255).astype(np.uint8)

    logger.info(f"Metalness map generated: {material_type}, "
                f"global={global_metalness:.3f}")
    return metalness_map, global_metalness


# ══════════════════════════════════════════════════════════════════════
# Normal Map Generation
# ══════════════════════════════════════════════════════════════════════

def generate_normal_map(
    color_image: np.ndarray,
    material_type: str = "generic",
    strength: float = 1.0,
) -> np.ndarray:
    """
    Generate a tangent-space normal map from a colour texture.

    Uses Sobel-based height gradient estimation with multi-scale blending,
    tuned per material type for CS2.

    Parameters
    ----------
    color_image : np.ndarray
        RGB uint8 colour texture.
    material_type : str
        Material type key.
    strength : float
        Normal map intensity multiplier (0-2).

    Returns
    -------
    np.ndarray : RGB uint8 normal map (tangent space, Blue = Z-up).
    """
    profile = get_pbr_profile(material_type)
    nstr = profile.normal_strength * strength

    # Convert to grayscale float for height
    if color_image.ndim == 3:
        gray = cv2.cvtColor(color_image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = color_image.copy()
    gray_f = gray.astype(np.float32) / 255.0

    h, w = gray_f.shape

    # Multi-scale approach: combine fine + coarse gradients
    normal_accum = np.zeros((h, w, 3), dtype=np.float32)

    scales = [
        (1, 0.5),   # Fine detail — pixel-level
        (3, 0.3),   # Medium detail
        (7, 0.2),   # Coarse structural detail
    ]

    for ksize, weight in scales:
        if ksize == 1:
            # Fine: direct Sobel
            dx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        else:
            # Blur then Sobel for larger scale
            blurred = cv2.GaussianBlur(gray_f, (ksize * 2 + 1, ksize * 2 + 1), ksize)
            dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

        normal_accum[:, :, 0] += -dx * weight * nstr
        normal_accum[:, :, 1] += -dy * weight * nstr

    # Z component
    normal_accum[:, :, 2] = 1.0

    # Normalise
    length = np.sqrt(np.sum(normal_accum ** 2, axis=-1, keepdims=True))
    length = np.maximum(length, 1e-6)
    normal_accum = normal_accum / length

    # Convert from [-1,1] to [0,255]
    normal_map = ((normal_accum + 1.0) * 0.5 * 255).astype(np.uint8)

    # Ensure RGB order: X=R, Y=G, Z=B (OpenGL tangent space standard)
    # Source 2 uses this convention
    logger.info(f"Normal map generated: {material_type}, strength={nstr:.2f}")
    return normal_map


# ══════════════════════════════════════════════════════════════════════
# Combined PBR Generation
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PBRResult:
    """Result of PBR map generation."""
    roughness_map: Optional[np.ndarray] = None
    metalness_map: Optional[np.ndarray] = None
    normal_map: Optional[np.ndarray] = None
    global_metalness: float = 0.0
    global_roughness: float = 0.7
    material_type: str = "generic"
    profile: Optional[PBRProfile] = None


def generate_pbr_maps(
    color_image: np.ndarray,
    material_type: str = "generic",
    strength: float = 1.0,
    generate_roughness: bool = True,
    generate_metalness: bool = True,
    generate_normals: bool = True,
) -> PBRResult:
    """
    Generate all PBR maps from a colour texture.

    Parameters
    ----------
    color_image : np.ndarray
        RGB uint8 colour texture.
    material_type : str
        Material type key (from VMAT surface property or AI detection).
    strength : float
        Generation intensity (0-2).
    generate_roughness, generate_metalness, generate_normals : bool
        Which maps to generate.

    Returns
    -------
    PBRResult with generated maps and global values.
    """
    profile = get_pbr_profile(material_type)
    result = PBRResult(material_type=material_type, profile=profile)

    if generate_roughness:
        result.roughness_map = generate_roughness_map(color_image, material_type, strength)
        result.global_roughness = float(result.roughness_map.mean()) / 255.0

    if generate_metalness:
        result.metalness_map, result.global_metalness = generate_metalness_map(
            color_image, material_type, strength)

    if generate_normals:
        result.normal_map = generate_normal_map(color_image, material_type, strength)

    logger.info(f"PBR maps generated for {material_type}: "
                f"rough={generate_roughness}, metal={generate_metalness}, "
                f"normal={generate_normals}")
    return result


def estimate_roughness_value(
    color_image: np.ndarray,
    material_type: str = "generic",
) -> float:
    """
    Estimate a single global roughness float suitable for VMAT inline value.

    Returns float 0.0-1.0.
    """
    profile = get_pbr_profile(material_type)

    if color_image.ndim == 3:
        gray = cv2.cvtColor(color_image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = color_image.copy()
    gray_f = gray.astype(np.float32) / 255.0

    # Measure surface detail amount
    blur = cv2.GaussianBlur(gray_f, (9, 9), 2.0)
    detail = np.abs(gray_f - blur).mean()

    # Map detail amount to roughness offset
    roughness = profile.roughness_base + (detail - 0.03) * profile.roughness_range * 5.0
    return float(np.clip(roughness, 0.0, 1.0))


def estimate_metalness_value(
    color_image: np.ndarray,
    material_type: str = "generic",
) -> float:
    """
    Estimate a single global metalness float for VMAT g_flMetalness.

    Returns float 0.0-1.0.
    """
    profile = get_pbr_profile(material_type)
    if profile.metalness_base < 0.01:
        return 0.0

    _, global_met = generate_metalness_map(color_image, material_type)
    return global_met
