"""
Core AI upscaling engine using Real-ESRGAN and RRDB architectures.
Supports multiple model variants, GPU/CPU inference, half-precision,
and specialised texture processing modes (diffuse, normal, roughness, etc.).
"""

import gc
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("CS2Upscaler.Upscaler")


class TextureType(Enum):
    """Classification of texture maps that determines processing behaviour."""
    DIFFUSE = "diffuse"
    NORMAL = "normal"
    ROUGHNESS = "roughness"
    METALLIC = "metallic"
    AO = "ao"
    EMISSIVE = "emissive"
    HEIGHT = "height"
    OPACITY = "opacity"
    AUTO = "auto"


class ModelVariant(Enum):
    """Available upscaling model variants."""
    REALESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REALESRGAN_X4PLUS_ANIME = "RealESRGAN_x4plus_anime_6B"
    REALESRGAN_X2PLUS = "RealESRGAN_x2plus"
    REALESRNET_X4PLUS = "RealESRNet_x4plus"


@dataclass
class UpscaleSettings:
    """Complete configuration for an upscale operation."""
    scale_factor: int = 4
    model_variant: ModelVariant = ModelVariant.REALESRGAN_X4PLUS
    texture_type: TextureType = TextureType.AUTO
    tile_size: int = 0
    tile_overlap: int = 32
    half_precision: bool = True
    preserve_alpha: bool = True
    seamless_mode: bool = False
    denoise_strength: float = 0.5
    sharpen_amount: float = 0.0
    color_correction: bool = True
    gpu_id: int = 0
    output_format: str = "png"
    output_quality: int = 95
    generate_mipmaps: bool = False
    max_mipmap_levels: int = 8
    max_output_size: int = 4096  # clamp longest edge (0 = unlimited)
    compression_quality: int = 90  # JPEG/WebP quality for output (1-100)
    compression_enabled: bool = True  # apply output compression
    generative_enhance: bool = False  # material-aware generative enhancement
    enhance_strength: float = 1.0  # enhancement intensity (0.0-2.0)
    photorealistic_mode: bool = False  # convert game textures to photorealistic
    anti_halo: bool = True  # suppress AI upscale halo artifacts
    text_preserve: bool = False  # protect text regions (user-marked textures only)
    material_override: str = "auto"  # force a specific material type or "auto"
    enhance_clahe: bool = True  # adaptive local contrast
    enhance_frequency_split: bool = True  # frequency-split detail injection
    enhance_detail_synth: bool = True  # high-frequency detail synthesis
    enhance_micro_overlay: bool = True  # micro-detail overlay
    enhance_wavelet: bool = True  # multi-scale wavelet sharpening
    enhance_sharpen: bool = True  # material-specific sharpening
    enhance_structure: bool = True  # edge-aware structure enhancement
    enhance_colour: bool = True  # colour enhancement
    # PBR map generation
    pbr_generate: bool = False  # enable PBR map generation
    pbr_roughness: bool = True  # generate roughness map
    pbr_metalness: bool = True  # generate metalness map
    pbr_normals: bool = True  # generate normal map
    pbr_strength: float = 1.0  # PBR generation intensity
    pbr_update_vmat: bool = True  # update VMAT files with new maps
    pbr_auto_values: bool = True  # auto-adjust roughness/metalness scalar values


@dataclass
class UpscaleResult:
    """Result of an upscale operation."""
    image: np.ndarray
    original_size: tuple
    upscaled_size: tuple
    model_used: str
    processing_time: float = 0.0
    mipmaps: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    detected_material: str = ""  # material type detected by generative enhance
    material_confidence: float = 0.0
    # PBR generation results
    pbr_roughness_map: Optional[np.ndarray] = None
    pbr_metalness_map: Optional[np.ndarray] = None
    pbr_normal_map: Optional[np.ndarray] = None
    pbr_global_metalness: float = 0.0
    pbr_global_roughness: float = 0.7


class TextureUpscaler:
    """
    High-performance AI texture upscaler built on Real-ESRGAN.

    Handles loading/unloading of models, GPU memory management,
    tile-based processing for large textures, and specialised processing
    pipelines for different texture map types.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device: Optional[torch.device] = None
        self._current_model = None
        self._current_model_name: Optional[str] = None
        self._upsampler = None
        self._gpu_gen: str = "unknown"  # "turing", "ampere", "ada", "blackwell", "cpu"
        self._gpu_cc: tuple = (0, 0)  # compute capability
        self._detect_device()

    # ------------------------------------------------------------------
    # GPU generation detection and per-gen optimizations
    # ------------------------------------------------------------------

    _GPU_GENS = {
        # (major, minor) -> generation label
        # Turing: RTX 20xx, GTX 16xx  (SM 7.5)
        (7, 5): "turing",
        # Ampere: RTX 30xx  (SM 8.0, 8.6)
        (8, 0): "ampere",
        (8, 6): "ampere",
        (8, 7): "ampere",  # some laptop chips
        (8, 9): "ada",     # Ada Lovelace: RTX 40xx (SM 8.9)
        # Blackwell: RTX 50xx  (SM 10.0+, 12.0)
        (10, 0): "blackwell",
        (10, 2): "blackwell",
        (12, 0): "blackwell",
    }

    def _classify_gpu_gen(self, major: int, minor: int) -> str:
        """Classify GPU generation from compute capability."""
        cc = (major, minor)
        if cc in self._GPU_GENS:
            return self._GPU_GENS[cc]
        # Fallback by major version
        if major >= 10:
            return "blackwell"
        if major == 8 and minor >= 9:
            return "ada"
        if major == 8:
            return "ampere"
        if major == 7:
            return "turing"
        return "pre_turing"

    def _detect_device(self):
        """Detect GPU, classify generation, and apply per-generation optimizations."""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram = props.total_memory / (1024 ** 3)
            self._gpu_cc = (props.major, props.minor)
            self._gpu_gen = self._classify_gpu_gen(props.major, props.minor)

            logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM, "
                        f"SM {props.major}.{props.minor}, gen={self._gpu_gen})")

            # ── Common optimizations (all NVIDIA GPUs) ───────────
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            if self._gpu_gen == "turing":
                # ── RTX 20xx / GTX 16xx (Turing, SM 7.5) ────────
                # No TF32, no BF16. FP16 works well.
                # cuDNN benchmark is the main speed lever.
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                logger.info("Turing optimizations: cuDNN benchmark, FP16")

            elif self._gpu_gen == "ampere":
                # ── RTX 30xx (Ampere, SM 8.x) ───────────────────
                # TF32 gives ~2x matmul speedup on 3rd-gen tensor cores.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Ampere optimizations: cuDNN benchmark, TF32")

            elif self._gpu_gen == "ada":
                # ── RTX 40xx (Ada Lovelace, SM 8.9) ─────────────
                # TF32 + 4th-gen tensor cores + FP8 support.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Ada Lovelace optimizations: cuDNN benchmark, TF32")

            elif self._gpu_gen == "blackwell":
                # ── RTX 50xx (Blackwell, SM 10.0+) ──────────────
                # TF32 + 5th-gen tensor cores + enhanced FP16/BF16 throughput.
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Blackwell benefits from larger workloads to saturate cores
                logger.info("Blackwell optimizations: cuDNN benchmark, TF32")

            else:
                # ── Older / unrecognised GPUs ────────────────────
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                logger.info("Legacy GPU: cuDNN benchmark only")

        else:
            self.device = torch.device("cpu")
            self._gpu_gen = "cpu"
            logger.info("No GPU detected – using CPU (slower)")

    @property
    def gpu_available(self) -> bool:
        return self.device is not None and self.device.type == "cuda"

    @property
    def gpu_info(self) -> dict:
        if not self.gpu_available:
            return {"name": "CPU", "vram_total": 0, "vram_used": 0, "vram_free": 0,
                    "generation": "cpu", "compute_capability": "N/A"}
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / (1024 ** 3)
        vram_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_total": round(vram_total, 2),
            "vram_used": round(vram_used, 2),
            "vram_free": round(vram_total - vram_used, 2),
            "generation": self._gpu_gen,
            "compute_capability": f"{self._gpu_cc[0]}.{self._gpu_cc[1]}",
        }

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _build_rrdb_net(self, variant: ModelVariant, scale: int):
        """Build the RRDBNet architecture matching the requested variant."""
        from basicsr.archs.rrdbnet_arch import RRDBNet

        if variant == ModelVariant.REALESRGAN_X4PLUS_ANIME:
            return RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=scale,
            )
        else:
            return RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=scale,
            )

    def load_model(self, settings: UpscaleSettings):
        """Load or switch to the requested model variant."""
        variant = settings.model_variant
        model_name = variant.value

        if self._current_model_name == model_name:
            logger.debug("Model already loaded – skipping.")
            return

        self.unload_model()

        from realesrgan import RealESRGANer

        scale = settings.scale_factor
        if variant in (ModelVariant.REALESRGAN_X4PLUS,
                       ModelVariant.REALESRGAN_X4PLUS_ANIME,
                       ModelVariant.REALESRNET_X4PLUS):
            native_scale = 4
        else:
            native_scale = 2

        model = self._build_rrdb_net(variant, native_scale)
        model_path = self.models_dir / f"{model_name}.pth"

        if not model_path.exists():
            logger.info(f"Downloading model: {model_name} ...")
            self._download_model(variant, model_path)

        half = settings.half_precision and self.gpu_available
        gpu_id = settings.gpu_id if self.gpu_available else None

        self._upsampler = RealESRGANer(
            scale=native_scale,
            model_path=str(model_path),
            model=model,
            tile=settings.tile_size,
            tile_pad=settings.tile_overlap,
            pre_pad=10,
            half=half,
            gpu_id=gpu_id,
        )

        # Try to use channels-last memory format for better GPU throughput
        if self.gpu_available:
            try:
                self._upsampler.model = self._upsampler.model.to(
                    memory_format=torch.channels_last)
                logger.info("Channels-last memory format enabled")
            except Exception:
                pass  # not all models support this

            # torch.compile for Ada Lovelace+ (RTX 40xx/50xx) — JIT fuses ops
            if self._gpu_gen in ("ada", "blackwell") and hasattr(torch, "compile"):
                try:
                    # The inductor backend (default) requires a working triton
                    # installation with CUDA toolkit access.  Test that first;
                    # if it fails fall back to the eager backend which still
                    # benefits from graph-break elimination.
                    _triton_ok = False
                    try:
                        import triton  # noqa: F401
                        # triton imported, but check it can actually find CUDA
                        from triton.runtime.driver import CudaUtils  # noqa: F401
                        _triton_ok = True
                    except Exception:
                        pass

                    if _triton_ok:
                        self._upsampler.model = torch.compile(
                            self._upsampler.model, mode="reduce-overhead")
                        logger.info(f"torch.compile enabled ({self._gpu_gen})")
                    else:
                        # eager backend: no triton/inductor, still fuses some ops
                        self._upsampler.model = torch.compile(
                            self._upsampler.model, backend="eager")
                        logger.info(f"torch.compile enabled (eager, no triton/CUDA toolkit) ({self._gpu_gen})")
                except Exception as e:
                    logger.debug(f"torch.compile skipped: {e}")

        self._current_model_name = model_name
        logger.info(f"Model loaded: {model_name}  (half={half})")

    def _download_model(self, variant: ModelVariant, dest: Path):
        """Download a pretrained model from the Real-ESRGAN release assets."""
        from basicsr.utils.download_util import load_file_from_url

        urls = {
            ModelVariant.REALESRGAN_X4PLUS:
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            ModelVariant.REALESRGAN_X4PLUS_ANIME:
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            ModelVariant.REALESRGAN_X2PLUS:
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            ModelVariant.REALESRNET_X4PLUS:
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        }
        url = urls.get(variant)
        if url is None:
            raise ValueError(f"No download URL for variant: {variant}")
        load_file_from_url(url, model_dir=str(dest.parent), file_name=dest.name)

    def unload_model(self):
        """Free GPU memory occupied by the current model."""
        if self._upsampler is not None:
            del self._upsampler
            self._upsampler = None
        self._current_model = None
        self._current_model_name = None
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model unloaded – GPU memory freed.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def upscale(self, image: np.ndarray, settings: UpscaleSettings,
                progress_callback=None) -> UpscaleResult:
        """
        Upscale a single image (H, W, C) uint8 numpy array.

        Handles alpha channel separation, texture-type-specific preprocessing,
        seamless padding, and post-processing.
        """
        import time
        t0 = time.perf_counter()

        self.load_model(settings)

        original_h, original_w = image.shape[:2]

        # Skip images that are too small for Real-ESRGAN's reflect padding
        # (pre_pad default is 10, image must be > pre_pad on each spatial dim)
        _MIN_DIM = 16
        if original_h < _MIN_DIM or original_w < _MIN_DIM:
            from PIL import Image as PILImage
            logger.warning(
                f"Image too small ({original_w}×{original_h}) for AI upscale "
                f"— resizing with Lanczos instead.")
            pil = PILImage.fromarray(image)
            new_w = original_w * settings.scale_factor
            new_h = original_h * settings.scale_factor
            pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
            output = np.array(pil)
            import time
            return UpscaleResult(
                image=output,
                original_size=(original_w, original_h),
                upscaled_size=(new_w, new_h),
                model_used="Lanczos (too small for AI)",
                processing_time=time.perf_counter() - t0,
            )

        has_alpha = image.shape[2] == 4 if image.ndim == 3 else False

        # Detect texture type if auto
        tex_type = settings.texture_type
        if tex_type == TextureType.AUTO:
            tex_type = self._detect_texture_type(image)

        # Separate alpha
        alpha_channel = None
        if has_alpha and settings.preserve_alpha:
            alpha_channel = image[:, :, 3]
            rgb = image[:, :, :3]
        else:
            rgb = image[:, :, :3] if image.ndim == 3 else np.stack([image]*3, axis=-1)

        # Pre-process based on texture type
        rgb = self._preprocess(rgb, tex_type, settings)

        # Save unpadded original for postprocess (before seamless padding)
        original_rgb = rgb

        # Seamless border padding
        if settings.seamless_mode:
            rgb = self._apply_seamless_padding(rgb, pad=settings.tile_overlap)

        # Run Real-ESRGAN (expects BGR input, returns BGR output)
        if progress_callback:
            progress_callback(0.1, "Running AI upscale...")

        # Auto-select tile size based on VRAM if set to 0 (auto)
        if settings.tile_size == 0 and self.gpu_available:
            auto_tile = self._auto_tile_size(rgb.shape, settings)
            self._upsampler.tile = auto_tile
            logger.info(f"Auto tile size: {auto_tile}")

        bgr_in = rgb[:, :, ::-1]  # RGB \u2192 BGR for Real-ESRGAN
        with torch.inference_mode():
            output_bgr, _ = self._upsampler.enhance(bgr_in, outscale=settings.scale_factor)
        output = output_bgr[:, :, ::-1].copy()  # BGR → RGB back

        if progress_callback:
            progress_callback(0.75, "Post-processing...")

        # Remove seamless padding
        if settings.seamless_mode:
            pad_scaled = settings.tile_overlap * settings.scale_factor
            output = output[pad_scaled:-pad_scaled, pad_scaled:-pad_scaled]

        # Post-process (flat-artifact suppression, sharpen, normal renorm — no color correction yet)
        output = self._postprocess(output, tex_type, settings, original_rgb=original_rgb)

        # Generative material-aware enhancement
        detected_material = ""
        material_confidence = 0.0
        if settings.generative_enhance:
            if progress_callback:
                progress_callback(0.80, "Applying generative enhancement...")
            from .material_enhancer import enhance_texture
            # Pass original image for anti-halo if enabled
            original_for_halo = image[:, :, :3] if settings.anti_halo else None
            # Material override: "auto" → None (auto-detect), else force type
            mat_override = None if settings.material_override == "auto" else settings.material_override
            output, mat_type, mat_conf = enhance_texture(
                output,
                material=mat_override,
                strength=settings.enhance_strength,
                original_image=original_for_halo,
                photorealistic=settings.photorealistic_mode,
                text_preserve=settings.text_preserve,
                progress_callback=None,
                stages={
                    "clahe": settings.enhance_clahe,
                    "frequency_split": settings.enhance_frequency_split,
                    "detail_synth": settings.enhance_detail_synth,
                    "micro_overlay": settings.enhance_micro_overlay,
                    "wavelet": settings.enhance_wavelet,
                    "sharpen": settings.enhance_sharpen,
                    "structure": settings.enhance_structure,
                    "colour": settings.enhance_colour,
                },
            )
            detected_material = mat_type.value
            material_confidence = mat_conf
            logger.info(f"Generative enhance: {detected_material} (conf={mat_conf:.2f})")

        # Final colour correction — runs AFTER all enhancement so nothing
        # can undo it.  Locks per-channel mean & std to the original image.
        if settings.color_correction and original_rgb is not None:
            output = self._match_color_stats(
                output[:, :, :3] if output.ndim == 3 and output.shape[2] >= 3 else output,
                original_rgb,
            )
            if has_alpha and alpha_channel is not None:
                pass  # alpha recombined later
            # output is now RGB uint8, no alpha yet

        # PBR map generation
        pbr_roughness_map = None
        pbr_metalness_map = None
        pbr_normal_map = None
        pbr_global_met = 0.0
        pbr_global_rough = 0.7
        if settings.pbr_generate:
            if progress_callback:
                progress_callback(0.88, "Generating PBR maps...")
            from .pbr_generator import generate_pbr_maps
            # Use detected material type for PBR generation
            pbr_mat = detected_material if detected_material else "generic"
            if settings.material_override != "auto":
                pbr_mat = settings.material_override
            pbr_result = generate_pbr_maps(
                output[:, :, :3] if output.ndim == 3 else output,
                material_type=pbr_mat,
                strength=settings.pbr_strength,
                generate_roughness=settings.pbr_roughness,
                generate_metalness=settings.pbr_metalness,
                generate_normals=settings.pbr_normals,
            )
            pbr_roughness_map = pbr_result.roughness_map
            pbr_metalness_map = pbr_result.metalness_map
            pbr_normal_map = pbr_result.normal_map
            pbr_global_met = pbr_result.global_metalness
            pbr_global_rough = pbr_result.global_roughness
            logger.info(f"PBR maps generated for {pbr_mat}: "
                        f"metalness={pbr_global_met:.3f}, roughness={pbr_global_rough:.3f}")

        # Upscale and recombine alpha
        if alpha_channel is not None:
            alpha_up = self._upscale_alpha(alpha_channel, settings)
            # Ensure same spatial size
            if alpha_up.shape[:2] != output.shape[:2]:
                from PIL import Image as PILImage
                alpha_pil = PILImage.fromarray(alpha_up)
                alpha_pil = alpha_pil.resize(
                    (output.shape[1], output.shape[0]), PILImage.LANCZOS
                )
                alpha_up = np.array(alpha_pil)
            output = np.dstack([output, alpha_up])

        # Clamp to max output resolution
        if settings.max_output_size > 0:
            output = self._clamp_resolution(output, settings.max_output_size)

        # Generate mipmaps
        mipmaps = []
        if settings.generate_mipmaps:
            mipmaps = self._generate_mipmaps(output, settings.max_mipmap_levels)

        elapsed = time.perf_counter() - t0

        if progress_callback:
            progress_callback(1.0, "Done")

        return UpscaleResult(
            image=output,
            original_size=(original_w, original_h),
            upscaled_size=(output.shape[1], output.shape[0]),
            model_used=self._current_model_name or "unknown",
            processing_time=elapsed,
            mipmaps=mipmaps,
            detected_material=detected_material,
            material_confidence=material_confidence,
            pbr_roughness_map=pbr_roughness_map,
            pbr_metalness_map=pbr_metalness_map,
            pbr_normal_map=pbr_normal_map,
            pbr_global_metalness=pbr_global_met,
            pbr_global_roughness=pbr_global_rough,
            metadata={
                "texture_type": tex_type.value,
                "seamless": settings.seamless_mode,
                "scale": settings.scale_factor,
                "generative_enhance": settings.generative_enhance,
                "detected_material": detected_material,
                "material_confidence": material_confidence,
                "pbr_generated": settings.pbr_generate,
            },
        )

    # ------------------------------------------------------------------
    # Texture type detection
    # ------------------------------------------------------------------

    def _detect_texture_type(self, image: np.ndarray) -> TextureType:
        """Heuristic detection of texture map type from pixel statistics."""
        if image.ndim == 2:
            return TextureType.ROUGHNESS

        rgb = image[:, :, :3].astype(np.float32)
        mean = rgb.mean(axis=(0, 1))
        std = rgb.std(axis=(0, 1))

        # Normal maps typically have dominant blue, with R/G centred ~128
        if (120 < mean[0] < 140 and 120 < mean[1] < 140 and mean[2] > 200):
            return TextureType.NORMAL

        # Very low variance grey → roughness / metallic
        if std.mean() < 15 and abs(mean[0] - mean[1]) < 5 and abs(mean[1] - mean[2]) < 5:
            return TextureType.ROUGHNESS

        return TextureType.DIFFUSE

    # ------------------------------------------------------------------
    # Pre / post processing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray, tex_type: TextureType,
                    settings: UpscaleSettings) -> np.ndarray:
        """Texture-type-specific preprocessing before AI inference."""
        if tex_type == TextureType.NORMAL:
            # Convert from [0,255] to normalised vectors, process in linear space
            return image  # Real-ESRGAN handles uint8 directly

        return image

    def _postprocess(self, image: np.ndarray, tex_type: TextureType,
                     settings: UpscaleSettings,
                     original_rgb: np.ndarray = None) -> np.ndarray:
        """Post-processing pipeline: flat-region fix, sharpen, normalise.

        Note: colour correction (mean/std matching) is deliberately NOT done
        here — it runs as the very last step in upscale() so that generative
        enhancement cannot undo it.
        """
        # Suppress AI tile-boundary artifacts on flat / solid-colour regions
        if original_rgb is not None:
            image = self._suppress_flat_artifacts(image, original_rgb)

        if settings.sharpen_amount > 0:
            image = self._unsharp_mask(image, settings.sharpen_amount)

        if tex_type == TextureType.NORMAL:
            image = self._renormalise_normal_map(image)

        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _suppress_flat_artifacts(upscaled: np.ndarray,
                                 original: np.ndarray) -> np.ndarray:
        """In flat regions, blend AI output with Lanczos resize to remove tile artifacts."""
        import cv2
        from PIL import Image as PILImage

        h_o, w_o = original.shape[:2]
        h_u, w_u = upscaled.shape[:2]

        grey = cv2.cvtColor(original[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        ksize = max(5, min(h_o, w_o) // 16) | 1
        local_mean = cv2.blur(grey, (ksize, ksize))
        local_sq = cv2.blur(grey ** 2, (ksize, ksize))
        local_var = np.maximum(local_sq - local_mean ** 2, 0)
        local_std = np.sqrt(local_var)

        # 0 = flat (prefer Lanczos), 1 = textured (keep AI)
        mask = np.clip((local_std - 2.0) / 6.0, 0.0, 1.0)

        # Early exit if the whole image is textured
        if mask.min() > 0.95:
            return upscaled

        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, ksize * 0.5))
        mask_up = cv2.resize(mask, (w_u, h_u), interpolation=cv2.INTER_LINEAR)

        simple = np.array(PILImage.fromarray(original[:, :, :3]).resize(
            (w_u, h_u), PILImage.LANCZOS))

        mask3 = mask_up[:, :, np.newaxis]
        result = (upscaled[:, :, :3].astype(np.float32) * mask3
                  + simple.astype(np.float32) * (1.0 - mask3))
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _match_color_stats(upscaled: np.ndarray,
                           original: np.ndarray) -> np.ndarray:
        """Match per-channel mean and standard deviation of *upscaled* to *original*.

        This corrects the colour drift / wash-out that Real-ESRGAN (and FP16
        quantisation) introduces without altering spatial detail.
        """
        up_f = upscaled.astype(np.float32)
        orig_f = original.astype(np.float32)
        for c in range(min(up_f.shape[2], orig_f.shape[2], 3)):
            o_mean = orig_f[:, :, c].mean()
            o_std = orig_f[:, :, c].std()
            u_mean = up_f[:, :, c].mean()
            u_std = up_f[:, :, c].std()
            if u_std < 1e-6:
                continue
            up_f[:, :, c] = (up_f[:, :, c] - u_mean) * (o_std / u_std) + o_mean
        return np.clip(up_f, 0, 255).astype(np.uint8)

    def _unsharp_mask(self, image: np.ndarray, amount: float) -> np.ndarray:
        """Apply unsharp mask sharpening."""
        import cv2
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(image.astype(np.float64), 1.0 + amount,
                                     blurred.astype(np.float64), -amount, 0)
        return sharpened

    def _renormalise_normal_map(self, image: np.ndarray) -> np.ndarray:
        """Ensure normal map vectors are unit length after upscaling."""
        normals = image.astype(np.float32) / 127.5 - 1.0
        length = np.sqrt(np.sum(normals ** 2, axis=-1, keepdims=True))
        length = np.maximum(length, 1e-6)
        normals = normals / length
        return ((normals + 1.0) * 127.5).astype(np.uint8)

    # ------------------------------------------------------------------
    # Resolution clamping
    # ------------------------------------------------------------------

    def _clamp_resolution(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """Downscale image if either dimension exceeds max_size, preserving aspect ratio."""
        h, w = image.shape[:2]
        if h <= max_size and w <= max_size:
            return image
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(image)
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        logger.info(f"Clamped output {w}x{h} → {new_w}x{new_h} (max {max_size})")
        return np.array(pil)

    # ------------------------------------------------------------------
    # Alpha channel
    # ------------------------------------------------------------------

    def _upscale_alpha(self, alpha: np.ndarray, settings: UpscaleSettings) -> np.ndarray:
        """Upscale alpha channel using bicubic interpolation (avoids AI artefacts)."""
        from PIL import Image as PILImage
        h, w = alpha.shape[:2]
        new_w = w * settings.scale_factor
        new_h = h * settings.scale_factor
        pil = PILImage.fromarray(alpha)
        pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
        return np.array(pil)

    # ------------------------------------------------------------------
    # Seamless tiling support
    # ------------------------------------------------------------------

    def _auto_tile_size(self, image_shape: tuple, settings: UpscaleSettings) -> int:
        """Pick the largest tile size that fits in available VRAM, gen-aware."""
        if not self.gpu_available:
            return 256
        try:
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            vram_free = vram_total - torch.cuda.memory_allocated(0) / (1024 ** 3)
            scale = settings.scale_factor
            half_mult = 0.5 if settings.half_precision else 1.0

            # Generation-specific tile candidates and headroom:
            # Newer GPUs have better memory bandwidth → can push larger tiles
            if self._gpu_gen == "blackwell":
                candidates = [1536, 1280, 1024, 768, 512, 384, 256]
                headroom = 0.50  # aggressive — Blackwell has great bandwidth
            elif self._gpu_gen == "ada":
                candidates = [1280, 1024, 768, 512, 384, 256]
                headroom = 0.55
            elif self._gpu_gen == "ampere":
                candidates = [1024, 768, 512, 384, 256, 192]
                headroom = 0.60
            elif self._gpu_gen == "turing":
                candidates = [768, 512, 384, 256, 192, 128]
                headroom = 0.65  # Turing has less bandwidth; more conservative
            else:
                candidates = [512, 384, 256, 192, 128]
                headroom = 0.70

            for tile in candidates:
                tile_vram = (tile ** 2 * 3 * 4 * scale ** 2 * 8 * half_mult) / (1024 ** 3)
                if tile_vram < vram_free * (1.0 - headroom):
                    return tile
            return 128
        except Exception:
            return 512

    def _apply_seamless_padding(self, image: np.ndarray, pad: int) -> np.ndarray:
        """Mirror-pad borders so the upscaled result tiles seamlessly."""
        return np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='wrap')

    # ------------------------------------------------------------------
    # Mipmap generation
    # ------------------------------------------------------------------

    def _generate_mipmaps(self, image: np.ndarray, max_levels: int) -> list:
        """Generate a mipmap chain from the upscaled image."""
        from PIL import Image as PILImage
        mipmaps = []
        h, w = image.shape[:2]
        current = PILImage.fromarray(image)
        for level in range(max_levels):
            w //= 2
            h //= 2
            if w < 1 or h < 1:
                break
            mip = current.resize((w, h), PILImage.LANCZOS)
            mipmaps.append(np.array(mip))
            current = mip
        return mipmaps
