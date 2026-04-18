"""Test the actual enhance_texture pipeline end-to-end on de_nuke textures."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from pathlib import Path
from src.core.texture_io import load_texture
from src.core.material_enhancer import classify_material, enhance_texture

NUKE_DIR = r"F:\Steam\steamapps\common\Counter-Strike Global Offensive\content\csgo_addons\cs_assault_d\materials\de_nuke"

def hf_energy(img):
    grey = cv2.cvtColor(img[:,:,:3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(grey, cv2.CV_32F).var())

def mad(a, b):
    return float(np.abs(a[:,:,:3].astype(np.float32) - b[:,:,:3].astype(np.float32)).mean())

def psnr(a, b):
    mse = float(np.mean((a[:,:,:3].astype(np.float32) - b[:,:,:3].astype(np.float32)) ** 2))
    if mse < 1e-6: return 99.0
    return float(10 * np.log10(255**2 / mse))

out = Path(__file__).parent / "debug_nuke_output"
out.mkdir(exist_ok=True)

for f in sorted(Path(NUKE_DIR).glob("*_color.tga")):
    img, _ = load_texture(str(f))
    rgb = img[:,:,:3]
    h, w = rgb.shape[:2]
    
    # Simulate upscale
    up = cv2.resize(rgb, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    
    mat, conf = classify_material(rgb)
    print(f"\n{f.stem}: {w}x{h} → {mat.value} (conf={conf:.2f})")
    
    # Run full pipeline with photorealistic=True
    enhanced, _, _ = enhance_texture(
        up.copy(), material=None, strength=1.0,
        original_image=rgb, photorealistic=True,
        text_preserve=False)
    
    hf_before = hf_energy(up)
    hf_after = hf_energy(enhanced)
    m = mad(up, enhanced)
    p = psnr(up, enhanced)
    
    print(f"  HF: {hf_before:.1f} → {hf_after:.1f} ({(hf_after-hf_before)/hf_before*100:+.1f}%)")
    print(f"  MAD: {m:.2f}, PSNR: {p:.1f} dB")
    
    cv2.imwrite(str(out / f"{f.stem}_pipeline_result.png"),
                cv2.cvtColor(enhanced[:,:,:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / f"{f.stem}_input_2x.png"),
                cv2.cvtColor(up, cv2.COLOR_RGB2BGR))
    
    print(f"  Saved to debug_nuke_output/")
