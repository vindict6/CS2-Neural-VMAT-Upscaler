"""
Stage-by-stage image comparison for de_nuke textures.
Saves each stage as a separate PNG so we can visually inspect.
Also tests a 'reduced' parameter set and compares.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from pathlib import Path
from src.core.texture_io import load_texture
from src.core.material_enhancer import (
    classify_material, MaterialType, _PROFILES, MaterialProfile,
    _apply_clahe, _frequency_split_enhance, _synthesise_detail,
    _apply_micro_detail, _wavelet_sharpen, _apply_sharpening,
    _structure_enhance, _enhance_colour, photorealistic_convert,
    _deband, _apply_photo_tonecurve,
)

NUKE_DIR = r"F:\Steam\steamapps\common\Counter-Strike Global Offensive\content\csgo_addons\cs_assault_d\materials\de_nuke"


def hf_energy(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())


def mean_abs_diff(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse < 1e-6: return 99.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def save(img, path):
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def process_texture(filepath, out_dir):
    stem = Path(filepath).stem
    img, _ = load_texture(filepath)
    rgb = img[:, :, :3].copy()
    h, w = rgb.shape[:2]

    mat_type, confidence = classify_material(rgb)
    profile = _PROFILES.get(mat_type, _PROFILES[MaterialType.GENERIC])
    conf_scale = 0.5 + 0.5 * min(confidence / 0.7, 1.0)
    eff = 1.0 * conf_scale

    print(f"\n{'='*60}")
    print(f"{stem}  |  {w}x{h}  |  {mat_type.value} ({confidence:.2f})  |  eff={eff:.2f}")
    print(f"{'='*60}")

    # Simulate upscale
    up = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    save(rgb, out_dir / f"{stem}_00_original.png")
    save(up, out_dir / f"{stem}_01_upscaled.png")

    # --- CURRENT pipeline (cumulative) ---
    current = up.copy()
    stage_fns = [
        ("02_clahe",       lambda i: _apply_clahe(i, profile, eff)),
        ("03_freqsplit",   lambda i: _frequency_split_enhance(i, profile, eff)),
        ("04_detailsynth", lambda i: _synthesise_detail(i, profile, eff)),
        ("05_micro",       lambda i: _apply_micro_detail(i, profile, eff)),
        ("06_wavelet",     lambda i: _wavelet_sharpen(i, profile, eff)),
        ("07_unsharp",     lambda i: _apply_sharpening(i, profile, eff)),
        ("08_structure",   lambda i: _structure_enhance(i, profile, eff)),
        ("09_colour",      lambda i: _enhance_colour(i, profile, eff)),
        ("10_photreal",    lambda i: photorealistic_convert(
            np.clip(i, 0, 255).astype(np.uint8), mat_type, eff)),
    ]

    for tag, fn in stage_fns:
        before_hf = hf_energy(current)
        current = np.clip(fn(current), 0, 255).astype(np.uint8)
        after_hf = hf_energy(current)
        mad = mean_abs_diff(up, current)
        print(f"  {tag:20s}  cumMAD={mad:6.2f}  HF={after_hf:8.1f} (Δ{after_hf-before_hf:+.1f})")
        save(current, out_dir / f"{stem}_{tag}.png")

    total_psnr = psnr(up, current)
    print(f"  TOTAL vs upscaled: PSNR={total_psnr:.1f} dB, MAD={mean_abs_diff(up, current):.2f}")

    # --- REDUCED pipeline: test lower CLAHE cap, skip double-sharpen, lower PBR ---
    print(f"\n  --- REDUCED pipeline test ---")
    reduced = up.copy()

    # 1) CLAHE with lower blend cap
    reduced = _apply_clahe_reduced(reduced, profile, eff)
    print(f"  CLAHE (reduced)     MAD={mean_abs_diff(up, reduced):6.2f}  HF={hf_energy(reduced):.1f}")

    # 2) Freq-split (keep as-is, it's mild)
    reduced = np.clip(_frequency_split_enhance(reduced, profile, eff), 0, 255).astype(np.uint8)

    # 3) Detail synth (keep)
    reduced = np.clip(_synthesise_detail(reduced, profile, eff), 0, 255).astype(np.uint8)

    # 4) Micro (keep)
    reduced = np.clip(_apply_micro_detail(reduced, profile, eff), 0, 255).astype(np.uint8)

    # 5) Wavelet sharpen at reduced strength
    reduced = _wavelet_sharpen_reduced(reduced, profile, eff)
    print(f"  After wavelet(red)  MAD={mean_abs_diff(up, reduced):6.2f}  HF={hf_energy(reduced):.1f}")

    # 6) SKIP unsharp — wavelet already sharpened
    print(f"  Unsharp SKIPPED")

    # 7) Structure (keep, it's mild)
    reduced = np.clip(_structure_enhance(reduced, profile, eff), 0, 255).astype(np.uint8)

    # 8) Colour (keep, it's very mild)
    reduced = np.clip(_enhance_colour(reduced, profile, eff), 0, 255).astype(np.uint8)

    # 9) Photorealistic with reduced bilateral + tonecurve
    reduced = np.clip(reduced, 0, 255).astype(np.uint8)
    reduced = _photorealistic_reduced(reduced, mat_type, eff, profile)
    print(f"  Photorealistic(red) MAD={mean_abs_diff(up, reduced):6.2f}  HF={hf_energy(reduced):.1f}")

    total_psnr_r = psnr(up, reduced)
    total_mad_r = mean_abs_diff(up, reduced)
    print(f"  REDUCED TOTAL: PSNR={total_psnr_r:.1f} dB, MAD={total_mad_r:.2f}")
    save(reduced, out_dir / f"{stem}_99_reduced.png")

    return stem


def _apply_clahe_reduced(image, profile, strength):
    """CLAHE with much lower blend cap (0.18 instead of 0.35)."""
    clip = profile.clahe_clip * strength
    if clip < 0.5:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0]
    grid = max(2, profile.clahe_grid)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    enhanced_l = clahe.apply(l_ch)

    h, w = l_ch.shape
    ksize = max(h, w) // grid
    ksize = max(ksize, 5) | 1
    l_f = l_ch.astype(np.float32)
    local_mean = cv2.blur(l_f, (ksize, ksize))
    local_sq = cv2.blur(l_f ** 2, (ksize, ksize))
    local_var = np.maximum(local_sq - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)
    var_mask = np.clip((local_std - 4.0) / 8.0, 0.0, 1.0)
    var_mask = cv2.GaussianBlur(var_mask, (0, 0), sigmaX=ksize * 0.5)

    # KEY CHANGE: cap at 0.18 instead of 0.35
    blend = min(strength * 0.18, 0.18)
    blend_map = (blend * var_mask).astype(np.float32)
    lab[:, :, 0] = np.clip(
        enhanced_l.astype(np.float32) * blend_map +
        l_ch.astype(np.float32) * (1.0 - blend_map),
        0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _wavelet_sharpen_reduced(image, profile, strength):
    """Wavelet sharpen at 60% of original strength."""
    amount = profile.sharpen_amount * strength * 0.5 * 0.6  # 60% of original
    if amount < 0.03:
        return image
    img_f = image.astype(np.float32)
    levels = []
    current = img_f
    for sigma in [1.0, 2.5, 5.0]:
        blurred = cv2.GaussianBlur(current, (0, 0), sigmaX=sigma)
        levels.append(current - blurred)
        current = blurred
    result = current
    boosts = [1.0 + amount * 0.6, 1.0 + amount * 0.4, 1.0 + amount * 0.2]
    for detail, boost in zip(levels[::-1], boosts[::-1]):
        result = result + detail * boost
    return np.clip(result, 0, 255).astype(np.uint8)


def _photorealistic_reduced(image, material, strength, profile):
    """Photorealistic with reduced bilateral strength and tonecurve blend."""
    pbr = profile.pbr_response * strength
    if pbr < 0.05:
        return image

    img_f = image.astype(np.float32)

    # 1) Deband: reduce sigma_color and blend
    sigma_color = 10.0 * pbr  # was 15.0
    sigma_space = 3.0 * pbr   # was 5.0
    if sigma_color >= 2.0:
        filtered = cv2.bilateralFilter(
            np.clip(img_f, 0, 255).astype(np.uint8),
            d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        blend = min(pbr * 0.25, 0.35)  # was 0.4, cap 0.6
        img_f = img_f * (1.0 - blend) + filtered.astype(np.float32) * blend

    # 2) Tonecurve: reduced blend
    img_f = _apply_photo_tonecurve_reduced(img_f, pbr)

    # 3) Colour desat (keep same, it's very mild)
    img_norm = np.clip(img_f, 0, 255).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    desat_factor = 1.0 - pbr * 0.15 * sat
    hsv[:, :, 1] *= desat_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)
    img_f = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0

    # 4-6) Skip surface imperfs, AO, film grain (they're mild but cumulative)
    # Actually keep them but at reduced strength
    from src.core.material_enhancer import _add_surface_imperfections, _add_ao_variation, _add_film_grain
    img_f = _add_surface_imperfections(img_f, material, pbr * 0.6)
    img_f = _add_ao_variation(img_f, pbr * 0.6)
    img_f = _add_film_grain(img_f, pbr * 0.3)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def _apply_photo_tonecurve_reduced(image, strength):
    """Tonecurve with lower blend (0.35 cap instead of 0.8)."""
    if strength < 0.05:
        return image
    x = np.arange(256, dtype=np.float32) / 255.0
    midpoint = 0.5
    slope = 1.0 + strength * 0.3
    curve = midpoint + (x - midpoint) * slope
    curve = 1.0 - np.exp(-curve * (1.5 + strength * 0.5))
    toe_lift = strength * 0.03
    curve = curve * (1.0 - toe_lift) + toe_lift
    curve = np.clip(curve, 0, 1)
    curve_min, curve_max = curve.min(), curve.max()
    if curve_max > curve_min:
        curve = (curve - curve_min) / (curve_max - curve_min)
    lut = (curve * 255.0).astype(np.uint8)
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    result = np.zeros_like(img_u8)
    for c in range(3):
        result[:, :, c] = cv2.LUT(img_u8[:, :, c], lut)
    # KEY: lower blend cap
    blend = min(strength * 0.35, 0.45)  # was 0.6, cap 0.8
    return image * (1.0 - blend) + result.astype(np.float32) * blend


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "debug_nuke_output"
    out_dir.mkdir(exist_ok=True)

    for f in sorted(Path(NUKE_DIR).glob("*_color.tga")):
        process_texture(str(f), out_dir)

    print("\n\nImages saved to debug_nuke_output/")
    print("Compare *_07_unsharp.png (current) vs *_99_reduced.png (proposed)")
