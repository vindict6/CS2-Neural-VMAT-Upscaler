"""
Material-aware generative texture enhancement.

Visually classifies what kind of material a texture represents from pixel
statistics and frequency analysis, then applies material-specific
post-processing that dramatically improves detail beyond a simple AI upscale.

Capabilities:
  1. Broad material classification (25+ categories)
  2. Adaptive local contrast (CLAHE tuned per material)
  3. High-frequency detail synthesis (procedural micro-detail)
  4. Material-specific sharpening profiles
  5. Color vibrancy / surface response adjustment
  6. Anti-halo pass (compare upscaled vs original to suppress edge ringing)
  7. Text detection & preservation (protect text from AI scrambling)
  8. Photorealistic conversion (remove game-texture look, add PBR realism)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger("CS2Upscaler.MaterialEnhancer")


# ══════════════════════════════════════════════════════════════════════
# Material classification — 25 categories
# ══════════════════════════════════════════════════════════════════════

class MaterialType(Enum):
    """Detected surface material category."""
    # Metals
    METAL_BARE = "metal_bare"
    METAL_PAINTED = "metal_painted"
    METAL_RUSTED = "metal_rusted"
    METAL_BRUSHED = "metal_brushed"
    # Stone & masonry
    STONE_ROUGH = "stone_rough"
    STONE_POLISHED = "stone_polished"
    CONCRETE = "concrete"
    BRICK = "brick"
    TILE_CERAMIC = "tile_ceramic"
    # Wood
    WOOD_RAW = "wood_raw"
    WOOD_FINISHED = "wood_finished"
    WOOD_WEATHERED = "wood_weathered"
    # Fabric & soft
    FABRIC_WOVEN = "fabric_woven"
    FABRIC_LEATHER = "fabric_leather"
    CARPET = "carpet"
    # Organic
    FOLIAGE = "foliage"
    BARK = "bark"
    SKIN = "skin"
    # Synthetic
    PLASTIC = "plastic"
    RUBBER = "rubber"
    GLASS = "glass"
    # Environmental
    TERRAIN_DIRT = "terrain_dirt"
    TERRAIN_SAND = "terrain_sand"
    TERRAIN_GRAVEL = "terrain_gravel"
    ASPHALT = "asphalt"
    PLASTER = "plaster"
    WALLPAPER = "wallpaper"
    # Special
    DECAL_TEXT = "decal_text"
    EMISSIVE_SCREEN = "emissive_screen"
    GENERIC = "generic"


# Human-readable descriptions for the UI
MATERIAL_DESCRIPTIONS = {
    MaterialType.METAL_BARE:     "Bare metal – enhancing specular scratches & reflections",
    MaterialType.METAL_PAINTED:  "Painted metal – enhancing chip detail & colour depth",
    MaterialType.METAL_RUSTED:   "Rusted metal – enhancing corrosion texture & pitting",
    MaterialType.METAL_BRUSHED:  "Brushed metal – enhancing directional grain & anisotropy",
    MaterialType.STONE_ROUGH:    "Rough stone – enhancing surface pores & fracture detail",
    MaterialType.STONE_POLISHED: "Polished stone – enhancing reflections & veining",
    MaterialType.CONCRETE:       "Concrete – enhancing aggregate & surface roughness",
    MaterialType.BRICK:          "Brick – enhancing mortar joints & surface erosion",
    MaterialType.TILE_CERAMIC:   "Ceramic tile – enhancing glaze depth & grout detail",
    MaterialType.WOOD_RAW:       "Raw wood – enhancing grain & knot detail",
    MaterialType.WOOD_FINISHED:  "Finished wood – enhancing lacquer depth & grain clarity",
    MaterialType.WOOD_WEATHERED: "Weathered wood – enhancing cracks & surface erosion",
    MaterialType.FABRIC_WOVEN:   "Woven fabric – enhancing weave pattern & thread detail",
    MaterialType.FABRIC_LEATHER: "Leather – enhancing pore structure & patina",
    MaterialType.CARPET:         "Carpet – enhancing fibre texture & pile depth",
    MaterialType.FOLIAGE:        "Foliage – enhancing leaf veins & colour richness",
    MaterialType.BARK:           "Bark – enhancing fissure depth & lichen detail",
    MaterialType.SKIN:           "Skin – enhancing pore detail & subsurface tone",
    MaterialType.PLASTIC:        "Plastic – enhancing subtle moulding lines & sheen",
    MaterialType.RUBBER:         "Rubber – enhancing grip texture & surface compression",
    MaterialType.GLASS:          "Glass – enhancing clarity & refraction detail",
    MaterialType.TERRAIN_DIRT:   "Dirt terrain – enhancing particle variation & organic scatter",
    MaterialType.TERRAIN_SAND:   "Sand – enhancing grain individuation & colour scatter",
    MaterialType.TERRAIN_GRAVEL: "Gravel – enhancing stone separation & shadow depth",
    MaterialType.ASPHALT:        "Asphalt – enhancing aggregate chips & tar variation",
    MaterialType.PLASTER:        "Plaster / stucco – enhancing surface roughness & trowel marks",
    MaterialType.WALLPAPER:      "Wallpaper – enhancing pattern emboss & colour fidelity",
    MaterialType.DECAL_TEXT:     "Decal / text – preserving legibility & edge sharpness",
    MaterialType.EMISSIVE_SCREEN: "Emissive / screen – enhancing glow edges & pixel grid",
    MaterialType.GENERIC:        "Generic surface – applying balanced enhancement",
}


@dataclass
class MaterialProfile:
    """Per-material enhancement parameters."""
    # CLAHE local contrast
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    # Sharpening (unsharp mask)
    sharpen_radius: float = 1.5
    sharpen_amount: float = 0.5
    # High-frequency detail synthesis
    detail_strength: float = 0.15
    detail_scale: float = 1.0
    detail_octaves: int = 3
    # Color enhancement
    saturation_boost: float = 1.0
    vibrance: float = 0.0
    warmth_shift: float = 0.0
    # Micro-detail overlay
    micro_strength: float = 0.0
    micro_scale: float = 0.5
    # Photorealistic conversion strength (0 = disabled, 1 = full)
    pbr_response: float = 0.0


# ── Tuned profiles (25+ materials) ───────────────────────────────────

_PROFILES = {
    # ── Metals ───────────────────────────────────────────────────
    MaterialType.METAL_BARE: MaterialProfile(
        clahe_clip=2.5, clahe_grid=8,
        sharpen_radius=1.0, sharpen_amount=0.7,
        detail_strength=0.20, detail_scale=1.2, detail_octaves=4,
        saturation_boost=0.95, vibrance=0.0, warmth_shift=-0.02,
        micro_strength=0.18, micro_scale=0.4, pbr_response=0.8,
    ),
    MaterialType.METAL_PAINTED: MaterialProfile(
        clahe_clip=2.0, clahe_grid=8,
        sharpen_radius=1.2, sharpen_amount=0.5,
        detail_strength=0.14, detail_scale=1.0, detail_octaves=3,
        saturation_boost=1.08, vibrance=0.1, warmth_shift=0.0,
        micro_strength=0.10, micro_scale=0.5, pbr_response=0.5,
    ),
    MaterialType.METAL_RUSTED: MaterialProfile(
        clahe_clip=3.0, clahe_grid=6,
        sharpen_radius=1.5, sharpen_amount=0.65,
        detail_strength=0.28, detail_scale=1.1, detail_octaves=4,
        saturation_boost=1.1, vibrance=0.15, warmth_shift=0.04,
        micro_strength=0.25, micro_scale=0.45, pbr_response=0.9,
    ),
    MaterialType.METAL_BRUSHED: MaterialProfile(
        clahe_clip=2.0, clahe_grid=10,
        sharpen_radius=0.8, sharpen_amount=0.8,
        detail_strength=0.18, detail_scale=1.3, detail_octaves=3,
        saturation_boost=0.95, vibrance=0.0, warmth_shift=-0.01,
        micro_strength=0.15, micro_scale=0.3, pbr_response=0.7,
    ),
    # ── Stone & masonry ──────────────────────────────────────────
    MaterialType.STONE_ROUGH: MaterialProfile(
        clahe_clip=3.0, clahe_grid=6,
        sharpen_radius=1.5, sharpen_amount=0.6,
        detail_strength=0.25, detail_scale=1.0, detail_octaves=4,
        saturation_boost=0.95, vibrance=0.0, warmth_shift=0.0,
        micro_strength=0.22, micro_scale=0.5, pbr_response=0.85,
    ),
    MaterialType.STONE_POLISHED: MaterialProfile(
        clahe_clip=1.5, clahe_grid=12,
        sharpen_radius=0.8, sharpen_amount=0.4,
        detail_strength=0.08, detail_scale=0.6, detail_octaves=2,
        saturation_boost=1.05, vibrance=0.05, warmth_shift=0.0,
        micro_strength=0.05, micro_scale=0.3, pbr_response=0.4,
    ),
    MaterialType.CONCRETE: MaterialProfile(
        clahe_clip=2.8, clahe_grid=6,
        sharpen_radius=1.5, sharpen_amount=0.55,
        detail_strength=0.22, detail_scale=1.0, detail_octaves=4,
        saturation_boost=0.95, vibrance=0.0, warmth_shift=0.0,
        micro_strength=0.20, micro_scale=0.5, pbr_response=0.8,
    ),
    MaterialType.BRICK: MaterialProfile(
        clahe_clip=2.5, clahe_grid=6,
        sharpen_radius=1.5, sharpen_amount=0.55,
        detail_strength=0.20, detail_scale=0.9, detail_octaves=3,
        saturation_boost=1.05, vibrance=0.1, warmth_shift=0.02,
        micro_strength=0.18, micro_scale=0.5, pbr_response=0.75,
    ),
    MaterialType.TILE_CERAMIC: MaterialProfile(
        clahe_clip=1.5, clahe_grid=10,
        sharpen_radius=1.0, sharpen_amount=0.45,
        detail_strength=0.10, detail_scale=0.7, detail_octaves=2,
        saturation_boost=1.05, vibrance=0.08, warmth_shift=0.0,
        micro_strength=0.06, micro_scale=0.35, pbr_response=0.5,
    ),
    # ── Wood ─────────────────────────────────────────────────────
    MaterialType.WOOD_RAW: MaterialProfile(
        clahe_clip=2.0, clahe_grid=8,
        sharpen_radius=1.8, sharpen_amount=0.5,
        detail_strength=0.18, detail_scale=0.8, detail_octaves=3,
        saturation_boost=1.1, vibrance=0.15, warmth_shift=0.03,
        micro_strength=0.12, micro_scale=0.6, pbr_response=0.7,
    ),
    MaterialType.WOOD_FINISHED: MaterialProfile(
        clahe_clip=1.5, clahe_grid=10,
        sharpen_radius=1.2, sharpen_amount=0.4,
        detail_strength=0.10, detail_scale=0.7, detail_octaves=2,
        saturation_boost=1.1, vibrance=0.1, warmth_shift=0.02,
        micro_strength=0.06, micro_scale=0.5, pbr_response=0.5,
    ),
    MaterialType.WOOD_WEATHERED: MaterialProfile(
        clahe_clip=2.8, clahe_grid=6,
        sharpen_radius=1.8, sharpen_amount=0.6,
        detail_strength=0.24, detail_scale=1.0, detail_octaves=4,
        saturation_boost=0.95, vibrance=0.05, warmth_shift=0.01,
        micro_strength=0.20, micro_scale=0.55, pbr_response=0.85,
    ),
    # ── Fabric & soft ────────────────────────────────────────────
    MaterialType.FABRIC_WOVEN: MaterialProfile(
        clahe_clip=1.5, clahe_grid=10,
        sharpen_radius=2.0, sharpen_amount=0.35,
        detail_strength=0.12, detail_scale=0.6, detail_octaves=2,
        saturation_boost=1.05, vibrance=0.1, warmth_shift=0.0,
        micro_strength=0.10, micro_scale=0.7, pbr_response=0.5,
    ),
    MaterialType.FABRIC_LEATHER: MaterialProfile(
        clahe_clip=2.2, clahe_grid=8,
        sharpen_radius=1.5, sharpen_amount=0.5,
        detail_strength=0.18, detail_scale=0.8, detail_octaves=3,
        saturation_boost=1.05, vibrance=0.08, warmth_shift=0.02,
        micro_strength=0.15, micro_scale=0.5, pbr_response=0.7,
    ),
    MaterialType.CARPET: MaterialProfile(
        clahe_clip=1.5, clahe_grid=10,
        sharpen_radius=2.5, sharpen_amount=0.3,
        detail_strength=0.14, detail_scale=0.5, detail_octaves=2,
        saturation_boost=1.05, vibrance=0.1, warmth_shift=0.0,
        micro_strength=0.12, micro_scale=0.8, pbr_response=0.4,
    ),
    # ── Organic ──────────────────────────────────────────────────
    MaterialType.FOLIAGE: MaterialProfile(
        clahe_clip=2.0, clahe_grid=8,
        sharpen_radius=1.5, sharpen_amount=0.45,
        detail_strength=0.15, detail_scale=0.9, detail_octaves=3,
        saturation_boost=1.15, vibrance=0.2, warmth_shift=0.02,
        micro_strength=0.10, micro_scale=0.6, pbr_response=0.6,
    ),
    MaterialType.BARK: MaterialProfile(
        clahe_clip=2.8, clahe_grid=6,
        sharpen_radius=1.6, sharpen_amount=0.55,
        detail_strength=0.22, detail_scale=1.0, detail_octaves=4,
        saturation_boost=1.0, vibrance=0.08, warmth_shift=0.01,
        micro_strength=0.18, micro_scale=0.5, pbr_response=0.8,
    ),
    MaterialType.SKIN: MaterialProfile(
        clahe_clip=1.2, clahe_grid=12,
        sharpen_radius=1.0, sharpen_amount=0.3,
        detail_strength=0.08, detail_scale=0.5, detail_octaves=2,
        saturation_boost=1.05, vibrance=0.05, warmth_shift=0.01,
        micro_strength=0.06, micro_scale=0.4, pbr_response=0.4,
    ),
    # ── Synthetic ────────────────────────────────────────────────
    MaterialType.PLASTIC: MaterialProfile(
        clahe_clip=1.2, clahe_grid=12,
        sharpen_radius=1.0, sharpen_amount=0.3,
        detail_strength=0.06, detail_scale=0.5, detail_octaves=2,
        saturation_boost=1.05, vibrance=0.05, warmth_shift=0.0,
        micro_strength=0.04, micro_scale=0.3, pbr_response=0.35,
    ),
    MaterialType.RUBBER: MaterialProfile(
        clahe_clip=1.8, clahe_grid=8,
        sharpen_radius=1.2, sharpen_amount=0.4,
        detail_strength=0.12, detail_scale=0.7, detail_octaves=3,
        saturation_boost=0.98, vibrance=0.0, warmth_shift=0.0,
        micro_strength=0.10, micro_scale=0.5, pbr_response=0.5,
    ),
    MaterialType.GLASS: MaterialProfile(
        clahe_clip=1.0, clahe_grid=16,
        sharpen_radius=0.8, sharpen_amount=0.25,
        detail_strength=0.03, detail_scale=0.3, detail_octaves=1,
        saturation_boost=1.0, vibrance=0.0, warmth_shift=-0.01,
        micro_strength=0.02, micro_scale=0.2, pbr_response=0.2,
    ),
    # ── Environmental ────────────────────────────────────────────
    MaterialType.TERRAIN_DIRT: MaterialProfile(
        clahe_clip=2.8, clahe_grid=6,
        sharpen_radius=1.8, sharpen_amount=0.6,
        detail_strength=0.22, detail_scale=1.1, detail_octaves=4,
        saturation_boost=1.05, vibrance=0.1, warmth_shift=0.02,
        micro_strength=0.20, micro_scale=0.55, pbr_response=0.8,
    ),
    MaterialType.TERRAIN_SAND: MaterialProfile(
        clahe_clip=2.0, clahe_grid=8,
        sharpen_radius=1.5, sharpen_amount=0.45,
        detail_strength=0.18, detail_scale=0.7, detail_octaves=3,
        saturation_boost=1.05, vibrance=0.08, warmth_shift=0.03,
        micro_strength=0.16, micro_scale=0.45, pbr_response=0.7,
    ),
    MaterialType.TERRAIN_GRAVEL: MaterialProfile(
        clahe_clip=3.0, clahe_grid=6,
        sharpen_radius=1.6, sharpen_amount=0.6,
        detail_strength=0.25, detail_scale=1.0, detail_octaves=4,
        saturation_boost=0.98, vibrance=0.05, warmth_shift=0.0,
        micro_strength=0.22, micro_scale=0.5, pbr_response=0.85,
    ),
    MaterialType.ASPHALT: MaterialProfile(
        clahe_clip=2.5, clahe_grid=6,
        sharpen_radius=1.5, sharpen_amount=0.55,
        detail_strength=0.20, detail_scale=1.0, detail_octaves=4,
        saturation_boost=0.95, vibrance=0.0, warmth_shift=0.0,
        micro_strength=0.18, micro_scale=0.5, pbr_response=0.75,
    ),
    MaterialType.PLASTER: MaterialProfile(
        clahe_clip=2.0, clahe_grid=8,
        sharpen_radius=1.5, sharpen_amount=0.45,
        detail_strength=0.15, detail_scale=0.8, detail_octaves=3,
        saturation_boost=1.0, vibrance=0.05, warmth_shift=0.01,
        micro_strength=0.12, micro_scale=0.5, pbr_response=0.6,
    ),
    MaterialType.WALLPAPER: MaterialProfile(
        clahe_clip=1.5, clahe_grid=10,
        sharpen_radius=1.2, sharpen_amount=0.35,
        detail_strength=0.10, detail_scale=0.6, detail_octaves=2,
        saturation_boost=1.05, vibrance=0.08, warmth_shift=0.0,
        micro_strength=0.06, micro_scale=0.5, pbr_response=0.35,
    ),
    # ── Special ──────────────────────────────────────────────────
    MaterialType.DECAL_TEXT: MaterialProfile(
        clahe_clip=1.0, clahe_grid=16,
        sharpen_radius=0.5, sharpen_amount=0.2,
        detail_strength=0.0, detail_scale=0.0, detail_octaves=0,
        saturation_boost=1.0, vibrance=0.0, warmth_shift=0.0,
        micro_strength=0.0, micro_scale=0.0, pbr_response=0.0,
    ),
    MaterialType.EMISSIVE_SCREEN: MaterialProfile(
        clahe_clip=1.5, clahe_grid=12,
        sharpen_radius=0.8, sharpen_amount=0.3,
        detail_strength=0.04, detail_scale=0.4, detail_octaves=1,
        saturation_boost=1.1, vibrance=0.1, warmth_shift=0.0,
        micro_strength=0.02, micro_scale=0.3, pbr_response=0.2,
    ),
    MaterialType.GENERIC: MaterialProfile(
        clahe_clip=2.0, clahe_grid=8,
        sharpen_radius=1.5, sharpen_amount=0.4,
        detail_strength=0.12, detail_scale=1.0, detail_octaves=3,
        saturation_boost=1.0, vibrance=0.05, warmth_shift=0.0,
        micro_strength=0.08, micro_scale=0.5, pbr_response=0.5,
    ),
}


# ══════════════════════════════════════════════════════════════════════
# Classification engine — expanded 25-category classifier
# ══════════════════════════════════════════════════════════════════════

def classify_material(image: np.ndarray) -> Tuple[MaterialType, float]:
    """
    Classify the material type of an RGB uint8 image into 25+ categories.

    Uses colour statistics, frequency analysis, directional content,
    channel correlations, texture regularity, and spatial structure.

    Returns (MaterialType, confidence) where confidence is 0-1.
    """
    if image.ndim == 2:
        return MaterialType.GENERIC, 0.3

    rgb = image[:, :, :3].astype(np.float32)
    h, w = rgb.shape[:2]

    # ── Core statistics ──────────────────────────────────────────
    mean_rgb = rgb.mean(axis=(0, 1))
    std_rgb = rgb.std(axis=(0, 1))
    global_mean = float(mean_rgb.mean())
    global_std = float(std_rgb.mean())

    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    mean_sat = float(hsv[:, :, 1].mean())
    std_sat = float(hsv[:, :, 1].std())
    mean_val = float(hsv[:, :, 2].mean())
    mean_hue = float(hsv[:, :, 0].mean())
    std_hue = float(hsv[:, :, 0].std())

    grey = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    img_var = max(float(grey.var()), 1.0)

    # ── Frequency content ────────────────────────────────────────
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    lap_var = float(lap.var())
    hf_ratio = lap_var / img_var

    # ── Directional energy ───────────────────────────────────────
    sobel_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    energy_x = float(np.abs(sobel_x).mean())
    energy_y = float(np.abs(sobel_y).mean())
    total_edge = energy_x + energy_y
    directionality = abs(energy_x - energy_y) / max(total_edge, 1.0)

    # ── Channel correlation ──────────────────────────────────────
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg_corr = _safe_corr(r.flatten(), g.flatten())
    rb_corr = _safe_corr(r.flatten(), b.flatten())
    gb_corr = _safe_corr(g.flatten(), b.flatten())
    channel_corr = (rg_corr + rb_corr + gb_corr) / 3.0

    # ── Colour ratios ────────────────────────────────────────────
    mr, mg, mb = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    grey_tone = (abs(mr - mg) < 15 and abs(mg - mb) < 15)
    is_warm = 10 < mean_hue < 30
    is_brown = (100 < mr < 200 and 60 < mg < 160 and 30 < mb < 130)
    is_green_brown = (25 < mean_hue < 80) and mean_sat > 40
    is_green = (35 < mean_hue < 75) and mean_sat > 50
    is_bright = mean_val > 200
    is_dark = mean_val < 70
    is_saturated = mean_sat > 80
    is_desaturated = mean_sat < 30

    # ── Regularity / periodicity (autocorrelation peak) ──────────
    regularity = _estimate_regularity(grey)

    # ── Local variance map (for detecting mixed surfaces) ────────
    ksize = max(h // 16, 5) | 1  # ensure odd
    local_mean = cv2.blur(grey, (ksize, ksize))
    local_sq_mean = cv2.blur(grey ** 2, (ksize, ksize))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_var_std = float(local_var.std())
    local_var_mean = float(local_var.mean())

    # ── Text / edge structure detection ──────────────────────────
    text_score = _compute_text_score(grey, lap)

    # ── Emissive check (very bright, saturated, high std) ────────
    bright_fraction = float((hsv[:, :, 2] > 220).mean())
    saturated_bright = float(((hsv[:, :, 2] > 200) & (hsv[:, :, 1] > 80)).mean())

    # ══════════════════════════════════════════════════════════════
    # Score every material type
    # ══════════════════════════════════════════════════════════════
    scores = {}

    # ── Decal / text ─────────────────────────────────────────────
    scores[MaterialType.DECAL_TEXT] = (
        text_score * 0.50 +
        min(hf_ratio / 60.0, 1.0) * 0.20 +
        (0.15 if local_var_std > 500 else 0.0) +
        (0.15 if global_std > 40 else global_std / 40.0 * 0.15)
    )

    # ── Emissive / screen ────────────────────────────────────────
    scores[MaterialType.EMISSIVE_SCREEN] = (
        min(bright_fraction * 2.0, 0.3) +
        min(saturated_bright * 3.0, 0.3) +
        (0.2 if is_saturated and is_bright else 0.0) +
        (0.2 if global_std > 50 else global_std / 50.0 * 0.1)
    )

    # ── Metals ───────────────────────────────────────────────────
    _metal_base = (
        (1.0 - mean_sat / 255.0) * 0.20 +
        channel_corr * 0.15 +
        min(hf_ratio / 50.0, 1.0) * 0.15 +
        (0.10 if global_std > 30 else global_std / 30.0 * 0.10)
    )
    scores[MaterialType.METAL_BARE] = (
        _metal_base +
        (0.20 if grey_tone and not is_brown else 0.0) +
        (0.10 if 80 < global_mean < 200 else 0.0) +
        min(hf_ratio / 40.0, 1.0) * 0.10
    )
    scores[MaterialType.METAL_PAINTED] = (
        _metal_base * 0.6 +
        (0.20 if is_saturated and channel_corr > 0.7 else 0.0) +
        (0.15 if hf_ratio < 30 else 0.0) +
        (0.15 if global_std > 25 and global_std < 55 else 0.0) +
        (0.10 if mean_sat > 40 else 0.0)
    )
    scores[MaterialType.METAL_RUSTED] = (
        _metal_base * 0.5 +
        (0.25 if is_warm and is_brown else 0.0) +
        min(hf_ratio / 60.0, 1.0) * 0.15 +
        (0.15 if global_std > 35 else 0.0) +
        (0.10 if mean_sat > 30 and mean_sat < 90 else 0.0)
    )
    scores[MaterialType.METAL_BRUSHED] = (
        _metal_base +
        directionality * 0.25 +
        (0.10 if is_desaturated else 0.0) +
        (0.05 if regularity > 0.3 else 0.0)
    )

    # ── Stone / masonry ──────────────────────────────────────────
    _stone_base = (
        (1.0 - mean_sat / 255.0) * 0.15 +
        min(hf_ratio / 60.0, 1.0) * 0.20 +
        (0.10 if global_std > 25 else global_std / 25.0 * 0.10)
    )
    scores[MaterialType.STONE_ROUGH] = (
        _stone_base +
        (0.20 if grey_tone else 0.05) +
        min(hf_ratio / 55.0, 1.0) * 0.15 +
        (0.10 if 80 < global_mean < 180 else 0.0) +
        (0.10 if local_var_mean > 200 else 0.0)
    )
    scores[MaterialType.STONE_POLISHED] = (
        (0.25 if grey_tone and hf_ratio < 20 else 0.0) +
        (0.20 if global_std < 35 else 0.0) +
        (0.20 if is_desaturated else 0.0) +
        (0.15 if mean_val > 120 else 0.0) +
        (0.10 if channel_corr > 0.9 else 0.0) +
        (0.10 if regularity < 0.2 else 0.0)
    )
    scores[MaterialType.CONCRETE] = (
        _stone_base +
        (0.20 if grey_tone else 0.0) +
        (0.15 if 70 < global_mean < 180 else 0.0) +
        (0.10 if is_desaturated or mean_sat < 40 else 0.0) +
        (0.10 if hf_ratio > 10 else 0.05) +
        (0.05 if is_warm and grey_tone else 0.0)  # warm grey = concrete/plaster
    )
    scores[MaterialType.BRICK] = (
        (0.20 if is_warm or is_brown else 0.0) +
        (0.20 if regularity > 0.25 else 0.0) +
        (0.15 if 20 < mean_sat < 80 else 0.0) +
        min(hf_ratio / 40.0, 1.0) * 0.15 +
        (0.15 if global_std > 30 else 0.0) +
        (0.15 if local_var_std > 300 else local_var_std / 300.0 * 0.10)
    )
    scores[MaterialType.TILE_CERAMIC] = (
        (0.25 if regularity > 0.35 else 0.0) +
        (0.20 if hf_ratio < 25 else 0.0) +
        (0.20 if global_std < 40 else 0.0) +
        (0.15 if mean_val > 140 else 0.0) +
        (0.10 if local_var_std > 400 else 0.0) +
        (0.10 if mean_sat < 60 else 0.0)
    )

    # ── Wood ─────────────────────────────────────────────────────
    _wood_base = (
        (0.20 if is_warm else 0.0) +
        (0.15 if is_brown else 0.0) +
        directionality * 0.15 +
        min(mean_sat / 150.0, 1.0) * 0.10
    )
    scores[MaterialType.WOOD_RAW] = (
        _wood_base +
        min(hf_ratio / 40.0, 1.0) * 0.15 +
        (0.15 if global_std > 25 else 0.0) +
        (0.10 if mean_sat > 30 else 0.0)
    )
    scores[MaterialType.WOOD_FINISHED] = (
        _wood_base +
        (0.15 if hf_ratio < 30 else 0.0) +
        (0.10 if global_std < 35 else 0.0) +
        (0.10 if mean_val > 120 else 0.0) +
        (0.05 if mean_sat > 20 else 0.0)
    )
    scores[MaterialType.WOOD_WEATHERED] = (
        _wood_base * 0.8 +
        (0.15 if global_std > 35 else 0.0) +
        min(hf_ratio / 50.0, 1.0) * 0.15 +
        (0.10 if is_desaturated or mean_sat < 40 else 0.0) +
        (0.10 if is_dark else 0.0)
    )

    # ── Fabric / soft ────────────────────────────────────────────
    scores[MaterialType.FABRIC_WOVEN] = (
        (0.20 if regularity > 0.3 else regularity * 0.20) +
        (0.15 if 15 < hf_ratio < 40 else 0.05) +
        (0.15 if 20 < global_std < 50 else 0.05) +
        (0.15 if mean_sat > 30 else 0.0) +
        (0.15 if directionality > 0.1 else 0.0) +
        (0.10 if 80 < global_mean < 200 else 0.0) +
        (0.10 if local_var_std < 200 else 0.0)
    )
    scores[MaterialType.FABRIC_LEATHER] = (
        (0.20 if is_brown or is_warm else 0.0) +
        (0.15 if 15 < hf_ratio < 35 else 0.05) +
        (0.15 if 20 < mean_sat < 70 else 0.0) +
        (0.15 if global_std < 40 else 0.0) +
        (0.15 if local_var_mean > 100 and local_var_mean < 400 else 0.0) +
        (0.10 if regularity < 0.25 else 0.0) +
        (0.10 if directionality < 0.15 else 0.0)
    )
    scores[MaterialType.CARPET] = (
        (0.20 if regularity > 0.2 else 0.0) +
        (0.20 if hf_ratio > 20 and hf_ratio < 50 else 0.05) +
        (0.15 if mean_sat > 20 else 0.0) +
        (0.15 if global_std < 40 else 0.0) +
        (0.15 if directionality < 0.1 else 0.0) +
        (0.10 if 80 < global_mean < 180 else 0.0) +
        (0.05 if local_var_std < 150 else 0.0)
    )

    # ── Organic ──────────────────────────────────────────────────
    scores[MaterialType.FOLIAGE] = (
        (0.30 if is_green else 0.0) +
        min(mean_sat / 180.0, 1.0) * 0.15 +
        min(hf_ratio / 50.0, 1.0) * 0.15 +
        (1.0 - directionality) * 0.10 +
        (0.15 if global_std > 30 else global_std / 30.0 * 0.10) +
        (0.10 if is_green_brown else 0.0) +
        (0.05 if std_hue > 10 else 0.0)
    )
    scores[MaterialType.BARK] = (
        (0.20 if is_brown else 0.0) +
        (0.15 if is_warm else 0.0) +
        min(hf_ratio / 50.0, 1.0) * 0.20 +
        (0.15 if global_std > 30 else 0.0) +
        (0.15 if mean_sat < 60 else 0.0) +
        (0.10 if directionality > 0.15 else 0.0) +
        (0.05 if local_var_mean > 200 else 0.0)
    )
    scores[MaterialType.SKIN] = (
        (0.25 if 5 < mean_hue < 25 and 40 < mean_sat < 90 else 0.0) +
        (0.20 if hf_ratio < 12 else 0.0) +
        (0.20 if global_std < 22 else 0.0) +
        (0.15 if 140 < global_mean < 210 else 0.0) +
        (0.10 if channel_corr > 0.85 else 0.0) +
        (0.10 if directionality < 0.1 else 0.0) +
        (-0.15 if grey_tone else 0.0)  # grey surfaces are not skin
    )

    # ── Synthetic ────────────────────────────────────────────────
    scores[MaterialType.PLASTIC] = (
        (0.25 if hf_ratio < 15 else 0.0) +
        (0.20 if global_std < 30 else 0.0) +
        min(mean_sat / 200.0, 1.0) * 0.15 +
        (0.15 if mean_val > 140 else 0.0) +
        (0.10 if channel_corr > 0.8 else channel_corr * 0.10) +
        (0.10 if regularity < 0.15 else 0.0) +
        (0.05 if directionality < 0.08 else 0.0)
    )
    scores[MaterialType.RUBBER] = (
        (0.20 if is_dark or global_mean < 100 else 0.0) +
        (0.20 if is_desaturated else 0.0) +
        (0.15 if 10 < hf_ratio < 30 else 0.0) +
        (0.15 if global_std < 35 else 0.0) +
        (0.15 if channel_corr > 0.9 else 0.0) +
        (0.10 if regularity > 0.1 else 0.0) +
        (0.05 if local_var_mean < 200 else 0.0)
    )
    scores[MaterialType.GLASS] = (
        (0.30 if is_bright and is_desaturated else 0.0) +
        (0.20 if hf_ratio < 10 else 0.0) +
        (0.20 if global_std < 20 else 0.0) +
        (0.15 if mean_val > 200 else 0.0) +
        (0.15 if channel_corr > 0.95 else 0.0)
    )

    # ── Environmental ────────────────────────────────────────────
    scores[MaterialType.TERRAIN_DIRT] = (
        (0.20 if is_brown and not grey_tone else 0.0) +
        (0.10 if is_warm and not grey_tone else 0.0) +
        min(hf_ratio / 50.0, 1.0) * 0.15 +
        (0.15 if global_std > 25 else 0.0) +
        (0.15 if 30 < mean_sat < 70 else 0.0) +
        (0.15 if directionality < 0.12 else 0.0) +
        (0.10 if 70 < global_mean < 150 else 0.0) +
        (-0.10 if grey_tone else 0.0)  # grey surfaces are not dirt
    )
    scores[MaterialType.TERRAIN_SAND] = (
        (0.25 if 15 < mean_hue < 35 and mean_sat > 25 else 0.0) +
        (0.20 if 150 < global_mean < 220 else 0.0) +
        (0.15 if hf_ratio < 25 else 0.0) +
        (0.15 if global_std < 30 else 0.0) +
        (0.15 if is_warm else 0.0) +
        (0.10 if directionality < 0.1 else 0.0)
    )
    scores[MaterialType.TERRAIN_GRAVEL] = (
        (0.20 if grey_tone or is_desaturated else 0.0) +
        min(hf_ratio / 60.0, 1.0) * 0.20 +
        (0.15 if global_std > 35 else 0.0) +
        (0.15 if local_var_mean > 300 else 0.0) +
        (0.15 if directionality < 0.1 else 0.0) +
        (0.10 if 80 < global_mean < 160 else 0.0) +
        (0.05 if regularity < 0.2 else 0.0)
    )
    scores[MaterialType.ASPHALT] = (
        (0.25 if is_dark and grey_tone else 0.0) +
        (0.20 if is_desaturated else 0.0) +
        min(hf_ratio / 40.0, 1.0) * 0.15 +
        (0.15 if 40 < global_mean < 100 else 0.0) +
        (0.15 if global_std < 30 else 0.0) +
        (0.10 if local_var_mean > 50 and local_var_mean < 300 else 0.0)
    )
    scores[MaterialType.PLASTER] = (
        (0.20 if is_desaturated or mean_sat < 35 else 0.0) +
        (0.20 if 10 < hf_ratio < 30 else 0.0) +
        (0.15 if 140 < global_mean < 230 else 0.0) +
        (0.15 if global_std < 35 else 0.0) +
        (0.15 if channel_corr > 0.85 else 0.0) +
        (0.10 if local_var_std < 200 else 0.0) +
        (0.05 if directionality < 0.1 else 0.0)
    )
    scores[MaterialType.WALLPAPER] = (
        (0.25 if regularity > 0.4 else regularity * 0.25) +
        (0.20 if mean_sat > 30 else 0.0) +
        (0.15 if hf_ratio < 25 else 0.0) +
        (0.15 if 100 < global_mean < 210 else 0.0) +
        (0.15 if local_var_std > 200 else 0.0) +
        (0.10 if global_std > 20 and global_std < 50 else 0.0)
    )

    # ── Pick best ────────────────────────────────────────────────
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
    confidence = min(best_score * 0.55 + margin * 0.45, 1.0)

    if confidence < 0.20:
        best_type = MaterialType.GENERIC
        confidence = 0.25

    logger.info(
        f"Material classified as {best_type.value} "
        f"(confidence={confidence:.2f}, top3={_fmt_top3(scores)})")
    return best_type, confidence


def _estimate_regularity(grey: np.ndarray) -> float:
    """
    Estimate texture regularity / periodicity via autocorrelation.
    Returns 0 (random) to 1 (highly periodic).
    """
    h, w = grey.shape
    # Work on a small crop for speed
    crop_size = min(128, h, w)
    cy, cx = h // 2, w // 2
    crop = grey[cy - crop_size // 2:cy + crop_size // 2,
                cx - crop_size // 2:cx + crop_size // 2]
    if crop.size == 0:
        return 0.0

    crop = crop - crop.mean()
    norm = float(np.sum(crop ** 2))
    if norm < 1e-6:
        return 0.0

    # Compute normalised autocorrelation via FFT
    f = np.fft.fft2(crop)
    power = np.abs(f) ** 2
    acr = np.fft.ifft2(power).real / norm

    # Mask out the central DC peak (within 5% of centre)
    ch, cw = acr.shape
    mask_r = max(int(ch * 0.05), 2)
    acr[:mask_r, :mask_r] = 0
    acr[:mask_r, -mask_r:] = 0
    acr[-mask_r:, :mask_r] = 0
    acr[-mask_r:, -mask_r:] = 0

    peak = float(acr.max())
    return min(max(peak, 0.0), 1.0)


def _compute_text_score(grey: np.ndarray, laplacian: np.ndarray) -> float:
    """
    Estimate likelihood that the image contains text or decal content.
    Uses edge structure, bimodal intensity, and connected component analysis.
    """
    h, w = grey.shape

    # 1) Bimodal intensity distribution → text on background
    hist = cv2.calcHist([grey.astype(np.uint8)], [0], None, [32], [0, 256])
    hist = hist.flatten() / hist.sum()
    # Measure how concentrated the histogram is in few bins
    sorted_bins = np.sort(hist)[::-1]
    top2_mass = float(sorted_bins[:2].sum())  # mass in top 2 bins
    bimodal = min(top2_mass / 0.5, 1.0)  # normalise

    # 2) Strong edges with clean structure (text has sharp edges)
    edge_vals = np.abs(laplacian)
    strong_edge_frac = float((edge_vals > edge_vals.mean() * 3).mean())

    # 3) Connected component analysis on binary image
    binary = cv2.adaptiveThreshold(
        grey.astype(np.uint8), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=max(11, (min(h, w) // 20) | 1), C=5)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if n_labels < 2:
        return 0.0

    # Filter components by size (text glyphs are small-to-medium, many of them)
    areas = stats[1:, cv2.CC_STAT_AREA]
    total_px = h * w
    min_glyph = total_px * 0.0002
    max_glyph = total_px * 0.05
    glyph_count = int(np.sum((areas > min_glyph) & (areas < max_glyph)))

    # Text typically has many small components with consistent sizing
    if glyph_count < 3:
        comp_score = 0.0
    else:
        glyph_areas = areas[(areas > min_glyph) & (areas < max_glyph)]
        area_cv = float(glyph_areas.std() / max(glyph_areas.mean(), 1.0))
        # Low coefficient of variation = consistent glyph size
        consistency = max(1.0 - area_cv, 0.0)
        comp_score = min(glyph_count / 15.0, 1.0) * 0.6 + consistency * 0.4

    return bimodal * 0.25 + strong_edge_frac * 10.0 * 0.25 + comp_score * 0.50


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient, safe for constant arrays."""
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 1.0
    n = min(len(a), 10000)
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def _fmt_top3(scores: dict) -> str:
    top = sorted(scores.items(), key=lambda x: -x[1])[:3]
    return ", ".join(f"{k.value}={v:.2f}" for k, v in top)


# ══════════════════════════════════════════════════════════════════════
# Text detection & region masking
# ══════════════════════════════════════════════════════════════════════

def detect_text_regions(image: np.ndarray) -> np.ndarray:
    """
    Detect regions likely containing text, symbols, or decals.

    Returns a float32 mask (H, W) in [0, 1] where 1 = high text probability.
    Used to protect those regions from generative enhancement.
    """
    grey = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    grey = grey.astype(np.uint8)
    h, w = grey.shape

    # 1) Adaptive threshold to find text-like foreground
    block = max(11, (min(h, w) // 16) | 1)
    binary = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=block, C=5)

    # 2) Connected components — filter for glyph-sized blobs
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    total_px = h * w
    min_glyph = total_px * 0.00015
    max_glyph = total_px * 0.04

    text_mask = np.zeros((h, w), dtype=np.float32)

    if n_labels < 2:
        return text_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    glyph_indices = []
    for i in range(len(areas)):
        a = areas[i]
        if min_glyph < a < max_glyph:
            # Check aspect ratio — text glyphs aren't too extreme
            cw = stats[i + 1, cv2.CC_STAT_WIDTH]
            ch = stats[i + 1, cv2.CC_STAT_HEIGHT]
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            if aspect < 8:
                glyph_indices.append(i + 1)

    if len(glyph_indices) < 3:
        return text_mask

    # 3) Mark glyph pixels
    for idx in glyph_indices:
        text_mask[labels == idx] = 1.0

    # 4) Dilate to cover surrounding area (text halo region)
    kernel_size = max(3, min(h, w) // 64) | 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)

    # 5) Smooth for gentle blending
    text_mask = cv2.GaussianBlur(text_mask, (0, 0), sigmaX=kernel_size)
    text_mask = np.clip(text_mask, 0.0, 1.0)

    logger.info(f"Text regions detected: {len(glyph_indices)} glyphs, "
                f"coverage={text_mask.mean():.1%}")
    return text_mask


# ══════════════════════════════════════════════════════════════════════
# Anti-halo pass — compare upscaled vs original to suppress edge halos
# ══════════════════════════════════════════════════════════════════════

def anti_halo_pass(upscaled: np.ndarray,
                   original: np.ndarray,
                   strength: float = 1.0) -> np.ndarray:
    """
    Compare the upscaled image to the original (bicubic-upscaled to same size)
    and suppress halo artifacts — bright/dark rings around edges that AI
    upscalers commonly introduce.

    Parameters
    ----------
    upscaled : np.ndarray
        The AI-upscaled RGB uint8 image.
    original : np.ndarray
        The original (pre-upscale) RGB uint8 image.
    strength : float
        0 = disabled, 1 = full correction.

    Returns
    -------
    Corrected RGB uint8 image.
    """
    if strength <= 0:
        return upscaled

    h_up, w_up = upscaled.shape[:2]
    h_orig, w_orig = original.shape[:2]

    # Resize original to match upscaled dimensions using high-quality bicubic
    if (h_orig, w_orig) != (h_up, w_up):
        ref = cv2.resize(original[:, :, :3], (w_up, h_up),
                         interpolation=cv2.INTER_CUBIC)
    else:
        ref = original[:, :, :3].copy()

    up_f = upscaled[:, :, :3].astype(np.float32)
    ref_f = ref.astype(np.float32)

    # Compute signed difference
    diff = up_f - ref_f

    # Detect edges in the reference (halos cluster around edges)
    ref_grey = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(ref_grey, cv2.CV_32F))
    edge_max = edges.max()
    if edge_max < 1.0:
        return upscaled
    edge_norm = edges / edge_max

    # Expand edge regions where halos form — use a tighter radius so fine
    # grit/detail isn't swept into the "halo zone".
    halo_zone = cv2.GaussianBlur(edge_norm, (0, 0), sigmaX=2.0)
    halo_zone = np.clip(halo_zone * 1.5, 0.0, 1.0)

    # Per-pixel difference magnitude
    diff_mag = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Halo detection: require a larger divergence before treating as halo
    # so subtle grit detail (small diffs near edges) is kept.
    halo_suspicion = halo_zone * np.clip(diff_mag / 60.0, 0.0, 1.0)

    # Correction: blend back toward reference in halo regions
    correction_strength = halo_suspicion * strength * 0.5
    correction_strength = np.clip(correction_strength, 0.0, 0.7)

    # Expand to 3 channels
    cs3 = correction_strength[:, :, np.newaxis]

    corrected = up_f * (1.0 - cs3) + ref_f * cs3
    result = np.clip(corrected, 0, 255).astype(np.uint8)

    # Preserve alpha if present
    if upscaled.ndim == 3 and upscaled.shape[2] == 4:
        result = np.dstack([result, upscaled[:, :, 3]])

    halo_pct = float(halo_suspicion.mean()) * 100
    logger.info(f"Anti-halo pass: {halo_pct:.1f}% halo coverage corrected")
    return result


# ══════════════════════════════════════════════════════════════════════
# Photorealistic conversion
# ══════════════════════════════════════════════════════════════════════

def photorealistic_convert(image: np.ndarray,
                           material: MaterialType,
                           strength: float = 1.0) -> np.ndarray:
    """
    Convert a game-looking texture toward photorealistic appearance.

    Addresses common game texture shortcomings:
    - Flat, hand-painted colour banding → natural gradient transitions
    - Over-saturated / cartoony colours → realistic colour response
    - Missing micro-surface detail → procedural PBR micro-texture
    - Uniform lighting baked in → subtle ambient occlusion variation
    - Hard tonal transitions → smooth tonal curves
    - Overly clean surfaces → realistic imperfections (dust, wear, stains)

    Parameters
    ----------
    image : np.ndarray – RGB uint8
    material : MaterialType
    strength : float – 0 to 1
    """
    if strength <= 0:
        return image

    profile = _PROFILES.get(material, _PROFILES[MaterialType.GENERIC])
    pbr = profile.pbr_response * strength
    if pbr < 0.05:
        return image

    h, w = image.shape[:2]
    img_f = image.astype(np.float32)

    # ── 1) De-band: smooth out quantisation / hand-paint banding ──
    img_f = _deband(img_f, pbr)

    # ── 2) Tonal curve: shift from game-linear to photographic ────
    img_f = _apply_photo_tonecurve(img_f, pbr)

    # ── 3) Colour desaturation toward realistic response ──────────
    #    Game textures are often over-saturated; pull toward natural
    img_norm = np.clip(img_f, 0, 255).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV)
    # hsv[:,:,1] in [0, 1]
    sat = hsv[:, :, 1]
    desat_factor = 1.0 - pbr * 0.15 * sat  # more saturated → more pull
    hsv[:, :, 1] *= desat_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)
    img_f = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0

    # ── 4) Micro-imperfections: dust, wear, subtle stains ─────────
    img_f = _add_surface_imperfections(img_f, material, pbr)

    # ── 5) Ambient occlusion variation ────────────────────────────
    img_f = _add_ao_variation(img_f, pbr)

    # ── 6) Film grain for photographic realism ────────────────────
    img_f = _add_film_grain(img_f, pbr * 0.5)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def _deband(image: np.ndarray, strength: float) -> np.ndarray:
    """Reduce colour banding with edge-preserving smoothing."""
    sigma_color = 15.0 * strength
    sigma_space = 5.0 * strength
    if sigma_color < 2.0:
        return image
    filtered = cv2.bilateralFilter(
        np.clip(image, 0, 255).astype(np.uint8),
        d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    # Blend: keep edges from original, smoothness from filtered
    blend = min(strength * 0.4, 0.6)
    return image * (1.0 - blend) + filtered.astype(np.float32) * blend


def _apply_photo_tonecurve(image: np.ndarray, strength: float) -> np.ndarray:
    """
    Apply a subtle S-curve and toe/shoulder roll-off to mimic photographic
    film response. Lifts shadows slightly, compresses highlights.
    """
    if strength < 0.05:
        return image

    # Build 256-entry LUT with an S-curve
    x = np.arange(256, dtype=np.float32) / 255.0

    # Soft S-curve: contrast in mid-tones, compression at extremes
    midpoint = 0.5
    slope = 1.0 + strength * 0.3  # subtle contrast boost
    curve = midpoint + (x - midpoint) * slope
    # Shoulder roll-off (highlights)
    curve = 1.0 - np.exp(-curve * (1.5 + strength * 0.5))
    # Toe lift (shadows)
    toe_lift = strength * 0.03
    curve = curve * (1.0 - toe_lift) + toe_lift

    # Normalise to 0-255
    curve = np.clip(curve, 0, 1)
    curve_min, curve_max = curve.min(), curve.max()
    if curve_max > curve_min:
        curve = (curve - curve_min) / (curve_max - curve_min)
    lut = (curve * 255.0).astype(np.uint8)

    # Apply per channel
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    result = np.zeros_like(img_u8)
    for c in range(3):
        result[:, :, c] = cv2.LUT(img_u8[:, :, c], lut)

    # Blend with original
    blend = min(strength * 0.6, 0.8)
    return image * (1.0 - blend) + result.astype(np.float32) * blend


def _add_surface_imperfections(image: np.ndarray, material: MaterialType,
                               strength: float) -> np.ndarray:
    """Add subtle surface imperfections (dust, micro-stains, wear marks)."""
    if strength < 0.05:
        return image

    h, w = image.shape[:2]
    rng = np.random.RandomState(77)

    # Dust / speckle layer: very fine, luminance-darkening
    dust = rng.randn(h, w).astype(np.float32)
    dust = cv2.GaussianBlur(dust, (0, 0), sigmaX=0.8)
    dust = np.clip(dust, -1, 0)  # only darkening (dust is dark)

    # Scale by material: rough surfaces get more imperfections
    rough_materials = {
        MaterialType.STONE_ROUGH, MaterialType.CONCRETE, MaterialType.ASPHALT,
        MaterialType.TERRAIN_DIRT, MaterialType.TERRAIN_GRAVEL,
        MaterialType.METAL_RUSTED, MaterialType.WOOD_WEATHERED,
        MaterialType.BARK, MaterialType.BRICK,
    }
    imp_scale = 6.0 if material in rough_materials else 3.0

    imperfection_layer = dust * strength * imp_scale
    for c in range(3):
        image[:, :, c] += imperfection_layer

    return image


def _add_ao_variation(image: np.ndarray, strength: float) -> np.ndarray:
    """Add subtle ambient occlusion variation using local structure."""
    if strength < 0.05:
        return image

    grey = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = grey.shape

    # Create AO-like darkening in concave regions (low local brightness)
    blur_large = cv2.GaussianBlur(grey, (0, 0),
                                   sigmaX=max(h // 32, 3))
    ao = (grey - blur_large) / 255.0
    ao = np.clip(ao * -1.0, 0.0, 0.3)  # negative diff = concavity

    ao_layer = ao * strength * 30.0
    for c in range(3):
        image[:, :, c] -= ao_layer

    return image


def _add_film_grain(image: np.ndarray, strength: float) -> np.ndarray:
    """Add subtle film-like grain for photographic feel."""
    if strength < 0.02:
        return image

    h, w = image.shape[:2]
    rng = np.random.RandomState(999)
    grain = rng.randn(h, w).astype(np.float32)

    # Grain is stronger in mid-tones (like real film)
    grey = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mid_mask = 1.0 - 4.0 * (grey - 0.5) ** 2
    mid_mask = np.clip(mid_mask, 0.1, 1.0)

    grain_layer = grain * mid_mask * strength * 8.0
    for c in range(3):
        image[:, :, c] += grain_layer

    return image


# ══════════════════════════════════════════════════════════════════════
# Main enhancement pipeline
# ══════════════════════════════════════════════════════════════════════

def enhance_texture(image: np.ndarray,
                    material: Optional[MaterialType] = None,
                    strength: float = 1.0,
                    original_image: Optional[np.ndarray] = None,
                    photorealistic: bool = False,
                    text_preserve: bool = False,
                    progress_callback=None,
                    stages: Optional[dict] = None) -> Tuple[np.ndarray, MaterialType, float]:
    """
    Apply material-aware generative enhancement to an upscaled texture.

    Parameters
    ----------
    image : np.ndarray
        RGB or RGBA uint8 image (already AI-upscaled).
    material : MaterialType or str, optional
        Force a specific material type. If None, auto-classify.
        Can be a MaterialType enum or a string matching a MaterialType value.
    strength : float
        Master strength multiplier (0.0 = off, 1.0 = full, 2.0 = extreme).
    original_image : np.ndarray, optional
        The original pre-upscale image. If provided, enables the anti-halo
        pass that compares upscaled vs original to suppress edge ringing.
    photorealistic : bool
        Enable photorealistic conversion mode.
    text_preserve : bool
        If True, detect and protect text/decal regions from enhancement.
    progress_callback : callable, optional
        (progress_float, message_str) callback.
    stages : dict, optional
        Toggle individual pipeline stages on/off. Keys:
        clahe, frequency_split, detail_synth, micro_overlay,
        wavelet, sharpen, structure, colour.
        All default to True if not specified.

    Returns
    -------
    (enhanced_image, detected_material, confidence)
    """
    # Resolve stages — default everything to True
    _stages = {
        "clahe": True,
        "frequency_split": True,
        "detail_synth": True,
        "micro_overlay": True,
        "wavelet": True,
        "sharpen": True,
        "structure": True,
        "colour": True,
    }
    if stages:
        _stages.update(stages)
    if strength <= 0:
        mat, conf = classify_material(image)
        return image, mat, conf

    has_alpha = image.ndim == 3 and image.shape[2] == 4
    alpha = image[:, :, 3] if has_alpha else None
    rgb = image[:, :, :3].copy()

    # ── Step 1: Classify material ────────────────────────────────
    if progress_callback:
        progress_callback(0.0, "Analysing material type...")

    if material is None:
        mat_type, confidence = classify_material(rgb)
    elif isinstance(material, str):
        # String override from UI — match to MaterialType enum
        try:
            mat_type = MaterialType(material)
        except ValueError:
            # Try matching by name
            for mt in MaterialType:
                if mt.value.lower() == material.lower() or mt.name.lower() == material.lower():
                    mat_type = mt
                    break
            else:
                mat_type, _ = classify_material(rgb)
        confidence = 1.0
    else:
        mat_type = material
        confidence = 1.0

    profile = _PROFILES.get(mat_type, _PROFILES[MaterialType.GENERIC])

    # Scale strength by classification confidence — low confidence means
    # less aggressive material-specific enhancement to avoid artifacts
    # from misclassification.  Manual overrides (confidence=1.0) are unaffected.
    conf_scale = 0.5 + 0.5 * min(confidence / 0.7, 1.0)  # 0.5 at conf=0, 1.0 at conf>=0.7
    effective_strength = strength * conf_scale

    if progress_callback:
        progress_callback(0.05, f"Detected: {mat_type.value} — enhancing...")

    # ── Step 2: Detect text regions (only if user marked this texture) ─
    has_text = False
    if text_preserve:
        if progress_callback:
            progress_callback(0.08, "Scanning for text regions...")

        text_mask = detect_text_regions(rgb)
        has_text = text_mask.max() > 0.1
        # Invert: 1 = safe to enhance, 0 = text (protect)
        enhance_mask = 1.0 - text_mask

        if has_text:
            logger.info("Text regions detected — will be protected from enhancement")
            # Save pre-enhancement version for text blending
            rgb_before_enhance = rgb.copy()

    # ── Step 3: Anti-halo pass ───────────────────────────────────
    if original_image is not None:
        if progress_callback:
            progress_callback(0.12, "Removing halo artifacts...")
        rgb = anti_halo_pass(rgb, original_image[:, :, :3], strength=strength)

    # ── Step 4: Adaptive local contrast (CLAHE) ──────────────────
    if _stages["clahe"]:
        if progress_callback:
            progress_callback(0.15, "Enhancing local contrast...")
        rgb = _apply_clahe(rgb, profile, effective_strength)

    # ── Step 5: Frequency-split detail injection ─────────────────
    if _stages["frequency_split"]:
        if progress_callback:
            progress_callback(0.25, "Injecting frequency-split detail...")
        rgb = _frequency_split_enhance(rgb, profile, effective_strength)

    # ── Step 6: High-frequency detail synthesis ──────────────────
    if _stages["detail_synth"]:
        if progress_callback:
            progress_callback(0.35, "Synthesising surface detail...")
        rgb = _synthesise_detail(rgb, profile, effective_strength)

    # ── Step 7: Micro-detail overlay ─────────────────────────────
    if _stages["micro_overlay"]:
        if progress_callback:
            progress_callback(0.42, "Adding micro-detail...")
        rgb = _apply_micro_detail(rgb, profile, effective_strength)

    # ── Step 8: Multi-scale wavelet sharpening ───────────────────
    if _stages["wavelet"]:
        if progress_callback:
            progress_callback(0.50, "Wavelet sharpening...")
        rgb = _wavelet_sharpen(rgb, profile, effective_strength)

    # ── Step 9: Material-specific sharpening ─────────────────────
    if _stages["sharpen"]:
        if progress_callback:
            progress_callback(0.55, "Applying material sharpening...")
        rgb = _apply_sharpening(rgb, profile, effective_strength)

    # ── Step 10: Edge-aware structure enhancement ────────────────
    if _stages["structure"]:
        if progress_callback:
            progress_callback(0.62, "Enhancing surface structure...")
        rgb = _structure_enhance(rgb, profile, effective_strength)

    # ── Step 11: Color enhancement ───────────────────────────────
    if _stages["colour"]:
        if progress_callback:
            progress_callback(0.68, "Enhancing colour response...")
        rgb = _enhance_colour(rgb, profile, effective_strength)

    # ── Step 12: Photorealistic conversion ───────────────────────
    if photorealistic:
        if progress_callback:
            progress_callback(0.78, "Applying photorealistic conversion...")
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = photorealistic_convert(rgb, mat_type, effective_strength)

    # ── Step 13: Text region blending (protect text) ────────────
    if has_text:
        if progress_callback:
            progress_callback(0.90, "Preserving text regions...")
        rgb = np.clip(rgb, 0, 255).astype(np.float32)
        mask3 = enhance_mask[:, :, np.newaxis]
        rgb = rgb * mask3 + rgb_before_enhance.astype(np.float32) * (1.0 - mask3)

    # Clamp
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Recombine alpha
    if alpha is not None:
        rgb = np.dstack([rgb, alpha])

    if progress_callback:
        progress_callback(1.0, f"Enhanced as {mat_type.value}")

    return rgb, mat_type, confidence


# ── Enhancement stages ───────────────────────────────────────────────

def _frequency_split_enhance(image: np.ndarray, profile: MaterialProfile,
                             strength: float) -> np.ndarray:
    """
    Frequency-split detail injection: decompose into low/mid/high frequency
    bands using Gaussian pyramid, selectively boost each band based on
    material profile, then recompose. This gives much finer control than
    simple sharpening.
    """
    detail_str = profile.detail_strength * strength
    if detail_str < 0.05:
        return image

    img_f = image.astype(np.float32)
    h, w = img_f.shape[:2]

    # Decompose into 3 frequency bands
    # Low: large-scale colour/lighting
    sigma_low = max(h, w) / 32.0
    low = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_low)

    # Mid: medium structure (mortar lines, grain patterns, seams)
    sigma_mid = max(h, w) / 128.0
    mid_raw = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma_mid)
    mid = mid_raw - low

    # High: fine detail (pores, scratches, micro-texture)
    high = img_f - mid_raw

    # Material-adaptive boost factors
    # Rough materials get more high-freq boost, smooth materials get mid-boost
    rough = profile.detail_strength
    smooth = 1.0 - rough

    mid_boost = 1.0 + detail_str * (0.15 + smooth * 0.2)
    high_boost = 1.0 + detail_str * (0.1 + rough * 0.3)

    # Suppress boost in flat / solid-colour regions to prevent amplifying
    # tiny AI-tile artefacts into visible contrast swirls.
    # Lower threshold so subtle grit (brick mortar, metal scratches) gets boosted.
    grey = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    ksize = max(5, min(h, w) // 16) | 1
    l_mean = cv2.blur(grey, (ksize, ksize))
    l_sq = cv2.blur(grey ** 2, (ksize, ksize))
    l_var = np.maximum(l_sq - l_mean ** 2, 0)
    l_std = np.sqrt(l_var)
    var_mask = np.clip((l_std - 1.5) / 5.0, 0.0, 1.0)
    var_mask = cv2.GaussianBlur(var_mask, (0, 0), sigmaX=max(2, ksize // 2))
    var3 = var_mask[:, :, np.newaxis]

    # Recompose — boost only in textured regions
    result = low + mid * (1.0 + (mid_boost - 1.0) * var3) + high * (1.0 + (high_boost - 1.0) * var3)
    return np.clip(result, 0, 255).astype(np.uint8)


def _wavelet_sharpen(image: np.ndarray, profile: MaterialProfile,
                     strength: float) -> np.ndarray:
    """
    Multi-scale wavelet-like sharpening using Laplacian pyramid
    reconstruction. Sharpens at multiple scales simultaneously for
    more natural-looking results than single-radius unsharp mask.
    """
    amount = profile.sharpen_amount * strength * 0.5
    if amount < 0.03:
        return image

    img_f = image.astype(np.float32)
    h, w = img_f.shape[:2]

    # Build 3-level Laplacian decomposition
    levels = []
    current = img_f
    sigmas = [1.0, 2.5, 5.0]

    for sigma in sigmas:
        blurred = cv2.GaussianBlur(current, (0, 0), sigmaX=sigma)
        detail = current - blurred
        levels.append(detail)
        current = blurred

    # Boost each level with decreasing strength at coarser scales
    result = current  # base (lowest frequency)
    boosts = [1.0 + amount * 0.6, 1.0 + amount * 0.4, 1.0 + amount * 0.2]

    for detail, boost in zip(levels[::-1], boosts[::-1]):
        result = result + detail * boost

    return np.clip(result, 0, 255).astype(np.uint8)


def _structure_enhance(image: np.ndarray, profile: MaterialProfile,
                       strength: float) -> np.ndarray:
    """
    Edge-aware structure enhancement using guided filter principles.
    Enhances local structure (mortar, grain, seams) while preserving
    smooth regions. More sophisticated than simple edge sharpening.
    """
    struct_str = profile.detail_strength * strength * 0.4
    if struct_str < 0.03:
        return image

    img_f = image.astype(np.float32)
    grey = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)

    h, w = grey.shape

    # Local mean and variance (structure map)
    radius = max(3, min(h, w) // 64) | 1
    local_mean = cv2.blur(grey, (radius, radius))
    local_sq = cv2.blur(grey ** 2, (radius, radius))
    local_var = np.maximum(local_sq - local_mean ** 2, 0)

    # Structure intensity: high variance = strong structure
    var_max = local_var.max()
    if var_max < 1.0:
        return image
    structure_map = np.sqrt(local_var) / np.sqrt(var_max)
    structure_map = cv2.GaussianBlur(structure_map, (0, 0), sigmaX=2.0)

    # Extract local detail (difference from large-scale blur)
    large_blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=max(h // 48, 3))
    detail = img_f - large_blur

    # Amplify detail only in structured regions
    struct3 = structure_map[:, :, np.newaxis]
    enhanced_detail = detail * (1.0 + struct_str * struct3 * 1.0)

    result = large_blur + enhanced_detail
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_clahe(image: np.ndarray, profile: MaterialProfile,
                 strength: float) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) in LAB space.

    Suppressed in regions with low local variance (solid / near-solid colours)
    to prevent tile-boundary brightness swirls.
    """
    clip = profile.clahe_clip * strength
    if clip < 0.5:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    grid = max(2, profile.clahe_grid)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    enhanced_l = clahe.apply(l_channel)

    # Build a variance mask: suppress CLAHE in uniform (solid-colour) areas
    # where the effect creates visible brightness swirl / tile artefacts.
    h, w = l_channel.shape
    ksize = max(h, w) // grid
    ksize = max(ksize, 5) | 1  # ensure odd and at least 5
    l_f = l_channel.astype(np.float32)
    local_mean = cv2.blur(l_f, (ksize, ksize))
    local_sq_mean = cv2.blur(l_f ** 2, (ksize, ksize))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    # Threshold: if local std < 4 the region is effectively solid colour.
    # Ramp from 0 (fully suppressed) to 1 (full CLAHE) between std 4-12.
    var_mask = np.clip((local_std - 4.0) / 8.0, 0.0, 1.0)
    var_mask = cv2.GaussianBlur(var_mask, (0, 0), sigmaX=ksize * 0.5)

    # Blend with original to control intensity, modulated by variance mask.
    # Cap at 0.35 to prevent contrast blowout — CLAHE is very aggressive.
    blend = min(strength * 0.35, 0.35)
    blend_map = (blend * var_mask).astype(np.float32)
    lab[:, :, 0] = np.clip(
        enhanced_l.astype(np.float32) * blend_map +
        l_channel.astype(np.float32) * (1.0 - blend_map),
        0, 255,
    ).astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _synthesise_detail(image: np.ndarray, profile: MaterialProfile,
                       strength: float) -> np.ndarray:
    """
    Synthesise high-frequency surface detail using multi-octave noise
    shaped by the image's local structure.
    """
    detail_str = profile.detail_strength * strength
    if detail_str < 0.01:
        return image

    h, w = image.shape[:2]
    img_f = image.astype(np.float32)

    # Generate base noise at image resolution
    rng = np.random.RandomState(42)  # deterministic for consistency
    accumulated = np.zeros((h, w), dtype=np.float32)

    freq = profile.detail_scale
    amplitude = 1.0
    total_amp = 0.0

    for octave in range(profile.detail_octaves):
        noise_h = max(4, int(h * freq))
        noise_w = max(4, int(w * freq))
        noise = rng.randn(noise_h, noise_w).astype(np.float32)
        # Upscale noise to full resolution with smooth interpolation
        noise_up = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        accumulated += noise_up * amplitude
        total_amp += amplitude
        freq *= 0.5
        amplitude *= 0.5

    if total_amp > 0:
        accumulated /= total_amp

    # Shape noise by local edge structure so detail concentrates in
    # already-textured areas (not flat regions or gradients)
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = cv2.Laplacian(grey, cv2.CV_32F)
    edge_mag = np.abs(edges)
    # Normalise edge magnitude to 0-1
    edge_max = edge_mag.max()
    if edge_max > 0:
        edge_mask = edge_mag / edge_max
    else:
        edge_mask = np.zeros_like(edge_mag)
    # Smooth the mask so transitions are gentle
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), sigmaX=5)

    # Apply noise shaped by edge mask
    detail_layer = accumulated * edge_mask * detail_str * 30.0

    # Add to all channels
    for c in range(3):
        img_f[:, :, c] += detail_layer

    return np.clip(img_f, 0, 255).astype(np.uint8)


def _apply_micro_detail(image: np.ndarray, profile: MaterialProfile,
                        strength: float) -> np.ndarray:
    """
    Add very fine micro-detail overlay simulating surface micro-structure.
    Uses high-pass filtered noise at a finer scale than detail synthesis.
    """
    micro_str = profile.micro_strength * strength
    if micro_str < 0.01:
        return image

    h, w = image.shape[:2]
    img_f = image.astype(np.float32)

    rng = np.random.RandomState(123)
    # Fine noise at close to pixel level
    scale = max(0.1, profile.micro_scale)
    noise_h = max(4, int(h * scale))
    noise_w = max(4, int(w * scale))
    noise = rng.randn(noise_h, noise_w).astype(np.float32)
    noise_up = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)

    # High-pass filter to get only the finest detail
    blurred = cv2.GaussianBlur(noise_up, (0, 0), sigmaX=1.5)
    hi_pass = noise_up - blurred

    # Luminance-adaptive: apply more in mid-tones, less in shadows/highlights
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mid_mask = 1.0 - 4.0 * (grey - 0.5) ** 2  # peaks at 0.5, zero at 0/1
    mid_mask = np.clip(mid_mask, 0.0, 1.0)

    micro_layer = hi_pass * mid_mask * micro_str * 25.0

    for c in range(3):
        img_f[:, :, c] += micro_layer

    return np.clip(img_f, 0, 255).astype(np.uint8)


def _apply_sharpening(image: np.ndarray, profile: MaterialProfile,
                      strength: float) -> np.ndarray:
    """Material-tuned unsharp mask sharpening."""
    amount = profile.sharpen_amount * strength
    if amount < 0.05:
        return image

    radius = max(0.5, profile.sharpen_radius)
    blurred = cv2.GaussianBlur(image.astype(np.float64), (0, 0), sigmaX=radius)
    sharpened = cv2.addWeighted(
        image.astype(np.float64), 1.0 + amount,
        blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _enhance_colour(image: np.ndarray, profile: MaterialProfile,
                    strength: float) -> np.ndarray:
    """
    Material-aware colour enhancement:
      - Saturation boost/cut
      - Selective vibrance (boosts muted colours more than saturated ones)
      - Warmth shift (subtle hue rotation)

    Operates entirely in float32 HSV to avoid uint8 rounding that washes
    out subtle colour differences.
    """
    img_f = image.astype(np.float32) / 255.0  # [0, 1] float RGB
    hsv = cv2.cvtColor(img_f, cv2.COLOR_RGB2HSV)
    # hsv[:,:,0] in [0, 360), [:,:,1] in [0, 1], [:,:,2] in [0, 1]

    # ── Saturation scaling ───────────────────────────────────────
    sat_mult = 1.0 + (profile.saturation_boost - 1.0) * strength
    if abs(sat_mult - 1.0) > 0.01:
        hsv[:, :, 1] *= sat_mult

    # ── Vibrance (selective saturation) ──────────────────────────
    vib = profile.vibrance * strength
    if abs(vib) > 0.01:
        sat = hsv[:, :, 1]
        # Boost factor is stronger for low-saturation pixels
        boost = vib * (1.0 - sat)
        hsv[:, :, 1] *= (1.0 + boost)

    # ── Warmth shift ─────────────────────────────────────────────
    warmth = profile.warmth_shift * strength * 360.0  # full float hue range
    if abs(warmth) > 0.1:
        hsv[:, :, 0] = (hsv[:, :, 0] + warmth) % 360.0

    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0.0, 360.0)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)
