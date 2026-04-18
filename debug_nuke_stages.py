"""
Diagnostic script: process de_nuke textures through every enhancement stage
individually, measuring how much each stage smooths / changes the image.

Outputs a per-stage report with:
  - PSNR vs previous stage
  - Mean absolute difference
  - High-frequency energy (Laplacian variance) before & after
  - Structural similarity (SSIM) vs input
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from pathlib import Path
from src.core.texture_io import load_texture
from src.core.material_enhancer import (
    classify_material, enhance_texture, MaterialType, _PROFILES, MaterialProfile,
    anti_halo_pass, _apply_clahe, _frequency_split_enhance, _synthesise_detail,
    _apply_micro_detail, _wavelet_sharpen, _apply_sharpening, _structure_enhance,
    _enhance_colour, photorealistic_convert,
)

NUKE_DIR = r"F:\Steam\steamapps\common\Counter-Strike Global Offensive\content\csgo_addons\cs_assault_d\materials\de_nuke"


def hf_energy(img):
    """High-frequency energy via Laplacian variance."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    return float(lap.var())


def mean_abs_diff(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse < 1e-6:
        return 99.0
    return float(10 * np.log10(255.0 ** 2 / mse))


def analyze_texture(filepath):
    print(f"\n{'='*70}")
    print(f"TEXTURE: {Path(filepath).name}")
    print(f"{'='*70}")

    img, info = load_texture(filepath)
    rgb = img[:, :, :3].copy()
    h, w = rgb.shape[:2]
    print(f"  Size: {w}x{h}, channels: {img.shape[2]}")

    # Classify
    mat_type, confidence = classify_material(rgb)
    profile = _PROFILES.get(mat_type, _PROFILES[MaterialType.GENERIC])
    print(f"  Classification: {mat_type.value} (confidence={confidence:.2f})")

    # Confidence scaling (as used in enhance_texture)
    conf_scale = 0.5 + 0.5 * min(confidence / 0.7, 1.0)
    effective_strength = 1.0 * conf_scale
    print(f"  Confidence scale: {conf_scale:.2f}, effective strength: {effective_strength:.2f}")
    print(f"  Profile: clahe_clip={profile.clahe_clip}, sharpen={profile.sharpen_amount}, "
          f"detail={profile.detail_strength}, micro={profile.micro_strength}, "
          f"pbr={profile.pbr_response}")
    print()

    # Simulate AI upscale with bicubic 2x (since we don't want to run the model)
    upscaled = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    baseline_hf = hf_energy(upscaled)
    print(f"  Baseline HF energy (after 2x bicubic): {baseline_hf:.1f}")
    print()

    # Run each stage individually and measure impact
    stages = [
        ("CLAHE", lambda img: _apply_clahe(img, profile, effective_strength)),
        ("Freq-split", lambda img: _frequency_split_enhance(img, profile, effective_strength)),
        ("Detail synth", lambda img: _synthesise_detail(img, profile, effective_strength)),
        ("Micro overlay", lambda img: _apply_micro_detail(img, profile, effective_strength)),
        ("Wavelet sharpen", lambda img: _wavelet_sharpen(img, profile, effective_strength)),
        ("Unsharp sharpen", lambda img: _apply_sharpening(img, profile, effective_strength)),
        ("Structure", lambda img: _structure_enhance(img, profile, effective_strength)),
        ("Colour", lambda img: _enhance_colour(img, profile, effective_strength)),
        ("Photorealistic", lambda img: photorealistic_convert(
            np.clip(img, 0, 255).astype(np.uint8), mat_type, effective_strength)),
    ]

    current = upscaled.copy()
    print(f"  {'Stage':<20} {'MAD':>6} {'PSNR':>7} {'HF before':>10} {'HF after':>10} {'HF Δ%':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")

    for name, fn in stages:
        before = current.copy()
        hf_before = hf_energy(before)
        after = fn(before)
        after = np.clip(after, 0, 255).astype(np.uint8)
        hf_after = hf_energy(after)
        mad = mean_abs_diff(before, after)
        p = psnr(before, after)
        hf_delta = ((hf_after - hf_before) / max(hf_before, 1)) * 100

        marker = ""
        if hf_delta < -5:
            marker = " *** SMOOTHING"
        elif hf_delta < -2:
            marker = " * mild smooth"

        print(f"  {name:<20} {mad:>6.2f} {p:>7.1f} {hf_before:>10.1f} {hf_after:>10.1f} {hf_delta:>+7.1f}%{marker}")
        current = after

    # Now run the full cumulative pipeline
    print(f"\n  --- Cumulative pipeline ---")
    cumulative_hf = hf_energy(current)
    total_delta = ((cumulative_hf - baseline_hf) / max(baseline_hf, 1)) * 100
    total_mad = mean_abs_diff(upscaled, current)
    total_psnr = psnr(upscaled, current)
    print(f"  Total: MAD={total_mad:.2f}, PSNR={total_psnr:.1f}, "
          f"HF: {baseline_hf:.1f} -> {cumulative_hf:.1f} ({total_delta:+.1f}%)")

    # Now analyze the photorealistic sub-stages individually
    print(f"\n  --- Photorealistic sub-stages breakdown ---")
    pbr_input = upscaled.copy()
    pbr_strength = profile.pbr_response * effective_strength

    from src.core.material_enhancer import _deband, _apply_photo_tonecurve, _add_surface_imperfections, _add_ao_variation, _add_film_grain

    pbr_substages = [
        ("Deband (bilateral)", lambda img: _deband(img.astype(np.float32), pbr_strength)),
        ("Photo tonecurve", lambda img: _apply_photo_tonecurve(img.astype(np.float32), pbr_strength)),
        ("Colour desat", lambda img: _pbr_colour_desat(img.astype(np.float32), pbr_strength)),
        ("Surface imperfs", lambda img: _add_surface_imperfections(img.astype(np.float32), mat_type, pbr_strength).clip(0, 255).astype(np.uint8)),
        ("AO variation", lambda img: _add_ao_variation(img.astype(np.float32), pbr_strength).clip(0, 255).astype(np.uint8)),
        ("Film grain", lambda img: _add_film_grain(img.astype(np.float32), pbr_strength * 0.5).clip(0, 255).astype(np.uint8)),
    ]

    pbr_current = pbr_input.copy()
    print(f"  {'Sub-stage':<22} {'MAD':>6} {'PSNR':>7} {'HF before':>10} {'HF after':>10} {'HF Δ%':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")

    for name, fn in pbr_substages:
        before = pbr_current.copy()
        hf_before = hf_energy(before)
        try:
            after = fn(before)
            after = np.clip(after, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
            continue
        hf_after = hf_energy(after)
        mad = mean_abs_diff(before, after)
        p = psnr(before, after)
        hf_delta = ((hf_after - hf_before) / max(hf_before, 1)) * 100

        marker = ""
        if hf_delta < -5:
            marker = " *** SMOOTHING"
        elif hf_delta < -2:
            marker = " * mild smooth"

        print(f"  {name:<22} {mad:>6.2f} {p:>7.1f} {hf_before:>10.1f} {hf_after:>10.1f} {hf_delta:>+7.1f}%{marker}")
        pbr_current = after

    # Save debug outputs
    debug_dir = Path(__file__).parent / "debug_nuke_output"
    debug_dir.mkdir(exist_ok=True)
    stem = Path(filepath).stem
    cv2.imwrite(str(debug_dir / f"{stem}_00_original.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"{stem}_01_bicubic2x.png"),
                cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(debug_dir / f"{stem}_02_enhanced.png"),
                cv2.cvtColor(current, cv2.COLOR_RGB2BGR))


def _pbr_colour_desat(img_f, pbr_strength):
    """Reproduce the colour desaturation step from photorealistic_convert."""
    img_norm = np.clip(img_f, 0, 255).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    desat_factor = 1.0 - pbr_strength * 0.15 * sat
    hsv[:, :, 1] *= desat_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    nuke_path = Path(NUKE_DIR)
    tga_files = sorted(nuke_path.glob("*_color.tga"))
    if not tga_files:
        print(f"No *_color.tga files found in {NUKE_DIR}")
        sys.exit(1)

    print(f"Found {len(tga_files)} texture(s) in {NUKE_DIR}")
    for f in tga_files:
        analyze_texture(str(f))

    print("\n\nDone. Debug images saved to debug_nuke_output/")
