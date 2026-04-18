"""
Per-stage cumulative image saves for visual inspection.
Saves after EACH stage so we can see exactly which stage introduces artifacts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from pathlib import Path
from src.core.texture_io import load_texture
from src.core.material_enhancer import (
    classify_material, MaterialType, _PROFILES,
    anti_halo_pass, _apply_clahe, _frequency_split_enhance, _synthesise_detail,
    _apply_micro_detail, _wavelet_sharpen, _apply_sharpening, _structure_enhance,
    _enhance_colour, photorealistic_convert,
)

NUKE_DIR = r"F:\Steam\steamapps\common\Counter-Strike Global Offensive\content\csgo_addons\cs_assault_d\materials\de_nuke"

def save(img, path):
    cv2.imwrite(str(path), cv2.cvtColor(img[:,:,:3], cv2.COLOR_RGB2BGR))

def process(filepath, out_dir):
    stem = Path(filepath).stem
    img, _ = load_texture(filepath)
    rgb = img[:,:,:3].copy()
    h, w = rgb.shape[:2]
    
    mat_type, confidence = classify_material(rgb)
    profile = _PROFILES.get(mat_type, _PROFILES[MaterialType.GENERIC])
    conf_scale = 0.5 + 0.5 * min(confidence / 0.7, 1.0)
    eff = conf_scale
    
    print(f"{stem}: {mat_type.value} (conf={confidence:.2f}, eff={eff:.2f})")
    
    up = cv2.resize(rgb, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    save(rgb, out_dir / f"{stem}_00_original.png")
    save(up, out_dir / f"{stem}_01_bicubic2x.png")
    
    current = up.copy()
    
    # Anti-halo
    current = anti_halo_pass(current, rgb, strength=1.0)
    save(current, out_dir / f"{stem}_02_antihalo.png")
    
    # CLAHE
    current = np.clip(_apply_clahe(current, profile, eff), 0, 255).astype(np.uint8)
    save(current, out_dir / f"{stem}_03_clahe.png")
    
    # Freq-split
    current = np.clip(_frequency_split_enhance(current, profile, eff), 0, 255).astype(np.uint8)
    save(current, out_dir / f"{stem}_04_freqsplit.png")
    
    # Detail synth
    current = np.clip(_synthesise_detail(current, profile, eff), 0, 255).astype(np.uint8)
    save(current, out_dir / f"{stem}_05_detailsynth.png")
    
    # Micro overlay
    current = np.clip(_apply_micro_detail(current, profile, eff), 0, 255).astype(np.uint8)
    save(current, out_dir / f"{stem}_06_micro.png")
    
    # Wavelet sharpen
    current = np.clip(_wavelet_sharpen(current, profile, eff), 0, 255).astype(np.uint8)
    save(current, out_dir / f"{stem}_07_wavelet.png")
    
    # Structure
    current = np.clip(_structure_enhance(current, profile, eff), 0, 255).astype(np.uint8)
    save(current, out_dir / f"{stem}_08_structure.png")
    
    # Colour
    current = np.clip(_enhance_colour(current, profile, eff), 0, 255).astype(np.uint8)
    save(current, out_dir / f"{stem}_09_colour.png")
    
    # Photorealistic
    current = np.clip(current, 0, 255).astype(np.uint8)
    current = photorealistic_convert(current, mat_type, eff)
    save(current, out_dir / f"{stem}_10_photorealistic.png")
    
    print(f"  Saved 11 stage images")

if __name__ == "__main__":
    out = Path(__file__).parent / "debug_nuke_stages"
    out.mkdir(exist_ok=True)
    for f in sorted(Path(NUKE_DIR).glob("*_color.tga")):
        process(str(f), out)
