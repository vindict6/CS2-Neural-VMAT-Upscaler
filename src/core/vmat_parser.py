"""
Comprehensive VMAT (Source 2 Material) parser for CS2.

Handles the text-based KeyValues format used by CS2 content tools,
including both "Layer0 { }" style and KV3 header style.

Parses all known shader parameters, texture references, feature flags,
float/int/vector params, and string attributes.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("CS2Upscaler.VmatParser")


# ── Known CS2 Shaders ────────────────────────────────────────────────────

KNOWN_SHADERS = {
    "csgo_complex.vfx": "Complex PBR",
    "csgo_simple.vfx": "Simple",
    "csgo_weapon.vfx": "Weapon",
    "csgo_character.vfx": "Character",
    "csgo_environment.vfx": "Environment",
    "csgo_environment_blend.vfx": "Environment Blend",
    "csgo_lightmapped.vfx": "Lightmapped",
    "csgo_lightmappedgeneric.vfx": "Lightmapped Generic",
    "csgo_glass.vfx": "Glass",
    "csgo_foliage.vfx": "Foliage",
    "csgo_decal.vfx": "Decal",
    "csgo_projected_decals.vfx": "Projected Decals",
    "csgo_vertexlitgeneric.vfx": "Vertex Lit Generic",
    "csgo_unlitgeneric.vfx": "Unlit Generic",
    "csgo_effects.vfx": "Effects",
    "csgo_static_overlay.vfx": "Static Overlay",
    "csgo_spritecard.vfx": "Sprite Card",
    "csgo_sky.vfx": "Sky",
    "csgo_water.vfx": "Water",
    "hero.vfx": "Hero",
    "vr_complex.vfx": "VR Complex",
    "vr_simple.vfx": "VR Simple",
    "vr_glass.vfx": "VR Glass",
    "generic.vfx": "Generic",
    "tools_wireframe.vfx": "Wireframe",
}


# ── Texture Parameter Mapping ────────────────────────────────────────────
# Maps VMAT texture keys to (canonical_name, texture_type, description)

class TextureRole(Enum):
    COLOR = "Color"
    NORMAL = "Normal"
    ROUGHNESS = "Roughness"
    METALNESS = "Metalness"
    AO = "Ambient Occlusion"
    SELF_ILLUM = "Self Illumination"
    EMISSIVE = "Emissive"
    DETAIL = "Detail"
    DETAIL_MASK = "Detail Mask"
    TINT_MASK = "Tint Mask"
    TRANSLUCENCY = "Translucency"
    HEIGHT = "Height"
    BLEND_MASK = "Blend Mask"
    COLOR2 = "Color 2"
    NORMAL2 = "Normal 2"
    ROUGHNESS2 = "Roughness 2"
    METALNESS2 = "Metalness 2"
    AO2 = "AO 2"
    SPECULAR = "Specular"
    ANISOTROPY = "Anisotropy"
    COAT_NORMAL = "Coat Normal"
    COAT_ROUGHNESS = "Coat Roughness"
    CUBEMAP = "Cubemap"
    OPACITY = "Opacity"
    THICKNESS = "Thickness"
    UNKNOWN = "Unknown"


# Both TextureXxx and g_tXxx styles
_TEXTURE_KEY_MAP: Dict[str, TextureRole] = {
    # TextureLayer1Xxx form (CS2 imported materials)
    "texturelayer1color": TextureRole.COLOR,
    "texturelayer1normal": TextureRole.NORMAL,
    "texturelayer1roughness": TextureRole.ROUGHNESS,
    "texturelayer1metalness": TextureRole.METALNESS,
    "texturelayer1ambientocclusion": TextureRole.AO,
    "texturelayer1detail": TextureRole.DETAIL,
    "texturelayer1selfillum": TextureRole.SELF_ILLUM,
    "texturelayer1tintmask": TextureRole.TINT_MASK,
    "texturelayer1translucency": TextureRole.TRANSLUCENCY,
    # Layer 2 (blend materials)
    "texturelayer2color": TextureRole.COLOR2,
    "texturelayer2normal": TextureRole.NORMAL2,
    "texturelayer2roughness": TextureRole.ROUGHNESS2,
    "texturelayer2metalness": TextureRole.METALNESS2,
    "texturelayer2ambientocclusion": TextureRole.AO2,
    # TextureXxx form
    "texturecolor": TextureRole.COLOR,
    "texturenormal": TextureRole.NORMAL,
    "textureroughness": TextureRole.ROUGHNESS,
    "texturemetalness": TextureRole.METALNESS,
    "textureambientocclusion": TextureRole.AO,
    "textureselfillummask": TextureRole.SELF_ILLUM,
    "textureemission": TextureRole.EMISSIVE,
    "textureemissive": TextureRole.EMISSIVE,
    "texturedetail": TextureRole.DETAIL,
    "texturedetailmask": TextureRole.DETAIL_MASK,
    "texturetintmask": TextureRole.TINT_MASK,
    "texturetranslucency": TextureRole.TRANSLUCENCY,
    "textureheight": TextureRole.HEIGHT,
    "textureblendmask": TextureRole.BLEND_MASK,
    "textureblendmodulation": TextureRole.BLEND_MASK,
    "texturecolor2": TextureRole.COLOR2,
    "texturenormal2": TextureRole.NORMAL2,
    "textureroughness2": TextureRole.ROUGHNESS2,
    "texturemetalness2": TextureRole.METALNESS2,
    "textureao2": TextureRole.AO2,
    "texturespecular": TextureRole.SPECULAR,
    "textureanisotropy": TextureRole.ANISOTROPY,
    "texturecoatnormal": TextureRole.COAT_NORMAL,
    "texturecoatroughness": TextureRole.COAT_ROUGHNESS,
    "textureopacity": TextureRole.OPACITY,
    "texturethickness": TextureRole.THICKNESS,
    "texturecubemap": TextureRole.CUBEMAP,
    # g_tXxx form
    "g_tcolor": TextureRole.COLOR,
    "g_tcolor1": TextureRole.COLOR,
    "g_tnormal": TextureRole.NORMAL,
    "g_troughness": TextureRole.ROUGHNESS,
    "g_tmetalness": TextureRole.METALNESS,
    "g_tambientocclusion": TextureRole.AO,
    "g_tselfillummask": TextureRole.SELF_ILLUM,
    "g_tselfillum": TextureRole.SELF_ILLUM,
    "g_temissive": TextureRole.EMISSIVE,
    "g_tdetail": TextureRole.DETAIL,
    "g_tdetailmask": TextureRole.DETAIL_MASK,
    "g_ttintmask": TextureRole.TINT_MASK,
    "g_ttranslucency": TextureRole.TRANSLUCENCY,
    "g_theight": TextureRole.HEIGHT,
    "g_tblendmodulation": TextureRole.BLEND_MASK,
    "g_tcolor2": TextureRole.COLOR2,
    "g_tnormal2": TextureRole.NORMAL2,
    "g_troughness2": TextureRole.ROUGHNESS2,
    "g_tmetalness2": TextureRole.METALNESS2,
    "g_tspecular": TextureRole.SPECULAR,
    "g_tanisotropy": TextureRole.ANISOTROPY,
    "g_tcoatnormal": TextureRole.COAT_NORMAL,
    "g_tcoatroughness": TextureRole.COAT_ROUGHNESS,
    "g_topacity": TextureRole.OPACITY,
    "g_tthickness": TextureRole.THICKNESS,
    "g_tcubemap": TextureRole.CUBEMAP,
}


# ── Feature Flags ────────────────────────────────────────────────────────

KNOWN_FEATURE_FLAGS = {
    "F_METALNESS_TEXTURE",
    "F_SELF_ILLUM",
    "F_TRANSLUCENT",
    "F_ALPHA_TEST",
    "F_ADDITIVE_BLEND",
    "F_SPECULAR",
    "F_DETAIL_TEXTURE",
    "F_AMBIENT_OCCLUSION_TEXTURE",
    "F_TINT_MASK",
    "F_SECONDARY_UV",
    "F_DO_NOT_CAST_SHADOWS",
    "F_RENDER_BACKFACES",
    "F_OVERLAY",
    "F_GLASS",
    "F_TEXTURE_ANIMATION",
    "F_RETRO_REFLECTIVE",
    "F_BLEND",
    "F_MORPH_SUPPORTED",
    "F_WRINKLE",
    "F_DISABLE_TONE_MAPPING",
    "F_FULLBRIGHT",
    "F_USE_BENT_NORMALS",
    "F_ANISOTROPIC_GLOSS",
    "F_SSS",
    "F_CLOTH_SHADING",
    "F_HIGH_QUALITY_GLOSS",
    "F_SPECULAR_CUBE_MAP",
    "F_SPECULAR_CUBE_MAP_PROJECTION",
    "F_ENABLE_NORMAL_SELF_SHADOW",
    "F_FORCE_UV2",
    "F_NO_TINT",
    "F_PAINT",
    "F_MASKS_1",
    "F_MASKS_2",
    "F_PREPASS_ALPHA_TEST",
}


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class VmatTexture:
    """A single texture reference in a VMAT."""
    key: str                  # Original key from the file
    role: TextureRole         # Canonical role
    path: str                 # Relative texture path
    resolved_path: str = ""   # Absolute path on disk (if found)
    exists: bool = False      # Whether the file exists on disk


@dataclass
class VmatMaterial:
    """Parsed representation of a complete VMAT file."""
    path: str                              # Absolute path to the .vmat file
    filename: str                          # Just the filename
    relative_path: str = ""                # Path relative to materials root
    shader: str = ""                       # Shader name (e.g. csgo_complex.vfx)
    shader_description: str = ""           # Human-readable shader name
    textures: List[VmatTexture] = field(default_factory=list)
    feature_flags: Dict[str, int] = field(default_factory=dict)
    float_params: Dict[str, float] = field(default_factory=dict)
    int_params: Dict[str, int] = field(default_factory=dict)
    vector_params: Dict[str, List[float]] = field(default_factory=dict)
    string_params: Dict[str, str] = field(default_factory=dict)
    dynamic_params: Dict[str, str] = field(default_factory=dict)
    render_attributes: List[str] = field(default_factory=list)
    layer: str = ""                        # Layer name (e.g. "Layer0")
    surface_property: str = ""             # PhysicsSurfaceProperties (e.g. "brick", "metal")
    raw_keyvalues: Dict[str, str] = field(default_factory=dict)
    raw_lines: List[str] = field(default_factory=list)  # Original file lines for write-back
    parse_errors: List[str] = field(default_factory=list)

    @property
    def texture_count(self) -> int:
        return len(self.textures)

    @property
    def found_texture_count(self) -> int:
        return sum(1 for t in self.textures if t.exists)

    def get_texture(self, role: TextureRole) -> Optional[VmatTexture]:
        for t in self.textures:
            if t.role == role:
                return t
        return None


# ── Parser ───────────────────────────────────────────────────────────────

_RE_KV3_HEADER = re.compile(
    r"<!--\s*kv3\s+encoding:.*?-->", re.DOTALL
)
_RE_VECTOR = re.compile(
    r"\[\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)(?:\s+([-\d.eE+]+))?\s*\]"
)


def parse_vmat(filepath: str, materials_root: str = "") -> VmatMaterial:
    """
    Parse a VMAT file and return a VmatMaterial.

    Args:
        filepath: Absolute path to the .vmat file.
        materials_root: Root directory for resolving relative texture paths.
                        If empty, uses the parent of the .vmat directory.
    """
    p = Path(filepath)
    mat = VmatMaterial(
        path=str(p.resolve()),
        filename=p.name,
    )

    if materials_root:
        try:
            mat.relative_path = str(p.relative_to(materials_root))
        except ValueError:
            mat.relative_path = p.name
    else:
        mat.relative_path = p.name

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        mat.parse_errors.append(f"Failed to read file: {e}")
        return mat

    # Store raw lines for write-back support
    mat.raw_lines = text.splitlines(keepends=True)

    # Strip KV3 header if present
    text = _RE_KV3_HEADER.sub("", text)

    # Parse the content
    _parse_keyvalues(text, mat, materials_root or str(p.parent.parent))

    # Resolve shader description
    mat.shader_description = KNOWN_SHADERS.get(mat.shader, mat.shader)

    # Resolve texture paths on disk
    _resolve_texture_paths(mat, materials_root or str(p.parent.parent))

    logger.debug(f"Parsed {mat.filename}: shader={mat.shader}, "
                 f"{mat.texture_count} textures, "
                 f"{len(mat.feature_flags)} flags")
    return mat


def _parse_keyvalues(text: str, mat: VmatMaterial, root: str):
    """Parse Valve KeyValues text format (both KV1 and simplified KV3)."""
    lines = text.splitlines()
    in_block = False
    block_depth = 0
    current_section = ""
    # For KV3-style array parsing
    in_array = False
    array_key = ""
    array_items: list = []
    current_array_item: dict = {}

    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("//"):
            continue

        # Handle block opening
        if line == "{":
            block_depth += 1
            if block_depth == 1 and current_section:
                in_block = True
            continue

        # Handle block closing
        if line == "}":
            if in_array and current_array_item:
                array_items.append(current_array_item)
                current_array_item = {}
            block_depth -= 1
            if block_depth == 0:
                in_block = False
            continue

        # Handle section headers like "Layer0" or Layer0
        stripped = line.strip('"')
        if not in_block and block_depth == 0 and '=' not in line:
            if re.match(r'^"?[\w]+"?$', line):
                current_section = stripped
                mat.layer = stripped
                continue

        # Parse key-value pairs
        _parse_kv_line(line, mat, root, line_num)


def _parse_kv_line(line: str, mat: VmatMaterial, root: str, line_num: int):
    """Parse a single key-value line."""
    # Try: "Key" "Value" (both quoted — standard CS2 VMAT format)
    m = re.match(r'^"([^"]+)"\s+"(.*?)"$', line)
    if m:
        key, value = m.group(1), m.group(2)
        _process_kv(key, value, mat, root)
        return

    # Try: "Key" "[x y z w]" (quoted key, vector value)
    m = re.match(r'^"([^"]+)"\s+"(\[.*?\])"$', line)
    if m:
        key, value = m.group(1), m.group(2)
        _process_kv(key, value, mat, root)
        return

    # Try: key "value" (unquoted key, quoted value — KV3 style)
    m = re.match(r'^(\S+)\s+"(.*?)"$', line)
    if m:
        key, value = m.group(1), m.group(2)
        _process_kv(key, value, mat, root)
        return

    # Try: key = "value" (KV3 style with equals)
    m = re.match(r'^(\S+)\s*=\s*"(.*?)"$', line)
    if m:
        key, value = m.group(1), m.group(2)
        _process_kv(key, value, mat, root)
        return

    # Try: key value (bare int/float value)
    m = re.match(r'^(\S+)\s+(\S+)$', line)
    if m:
        key, value = m.group(1), m.group(2)
        _process_kv(key, value, mat, root)
        return

    # Try: key = value (KV3 bare value)
    m = re.match(r'^(\S+)\s*=\s*(\S+)$', line)
    if m:
        key, value = m.group(1), m.group(2)
        _process_kv(key, value, mat, root)
        return

    # Try: key = [x, y, z, w] (KV3 vector)
    m = re.match(r'^(\S+)\s*=\s*(\[.*?\])$', line)
    if m:
        key, value = m.group(1), m.group(2)
        _process_kv(key, value, mat, root)
        return


def _process_kv(key: str, value: str, mat: VmatMaterial, root: str):
    """Process a single key-value pair into the material."""
    # Strip any remaining quotes from key (belt-and-suspenders)
    key = key.strip('"')
    key_lower = key.lower()
    mat.raw_keyvalues[key] = value

    # Shader
    if key_lower == "shader":
        mat.shader = value
        return

    # PhysicsSurfaceProperties
    if key_lower == "physicssurfaceproperties":
        mat.surface_property = value.lower()
        mat.string_params[key] = value
        return

    # Texture parameters — try exact match first, then strip layer number
    role = _TEXTURE_KEY_MAP.get(key_lower)
    if role is None:
        # Also try matching without quotes that may have leaked
        role = _TEXTURE_KEY_MAP.get(key_lower.strip('"'))
    if role is not None:
        mat.textures.append(VmatTexture(
            key=key,
            role=role,
            path=value,
        ))
        return

    # Feature flags (F_*)
    if key.startswith("F_") or key.startswith("f_"):
        try:
            mat.feature_flags[key] = int(value)
        except ValueError:
            mat.feature_flags[key] = 1 if value.lower() in ("true", "1") else 0
        return

    # Float params (g_fl*)
    if key_lower.startswith("g_fl"):
        try:
            mat.float_params[key] = float(value)
        except ValueError:
            mat.string_params[key] = value
        return

    # Integer params (g_n*)
    if key_lower.startswith("g_n"):
        try:
            mat.int_params[key] = int(value)
        except ValueError:
            mat.string_params[key] = value
        return

    # Vector params (g_v*) or any value that looks like a vector
    if key_lower.startswith("g_v") or value.startswith("["):
        vec = _parse_vector(value)
        if vec is not None:
            mat.vector_params[key] = vec
            return

    # Check if it looks like a texture path even with unknown key
    if _looks_like_texture_path(value):
        mat.textures.append(VmatTexture(
            key=key,
            role=_TEXTURE_KEY_MAP.get(key_lower, TextureRole.UNKNOWN),
            path=value,
        ))
        return

    # String attributes
    mat.string_params[key] = value


def _parse_vector(value: str) -> Optional[List[float]]:
    """Parse a vector string like '[1.0 2.0 3.0 4.0]' or '[1.0, 2.0, 3.0]'."""
    # Normalize commas to spaces
    cleaned = value.replace(",", " ")
    m = _RE_VECTOR.match(cleaned)
    if m:
        components = [float(m.group(i)) for i in range(1, 5) if m.group(i)]
        return components
    return None


def _looks_like_texture_path(value: str) -> bool:
    """Heuristic: does this value look like a texture file path?"""
    if not value:
        return False
    lower = value.lower()
    tex_extensions = (".tga", ".png", ".jpg", ".jpeg", ".bmp",
                      ".tiff", ".vtex", ".psd", ".dds", ".exr")
    if any(lower.endswith(ext) for ext in tex_extensions):
        return True
    if "materials/" in lower and ("/" in value or "\\" in value):
        return True
    return False


def _resolve_texture_paths(mat: VmatMaterial, root: str):
    """Try to find texture files on disk."""
    root_path = Path(root)
    for tex in mat.textures:
        if not tex.path:
            continue

        # Try direct path relative to root
        candidates = [
            root_path / tex.path,
            root_path / tex.path.replace("/", os.sep),
        ]

        # Also try stripping "materials/" prefix
        stripped = tex.path
        if stripped.lower().startswith("materials/"):
            stripped = stripped[len("materials/"):]
            candidates.append(root_path / "materials" / stripped)
            candidates.append(root_path / stripped)

        # Try common extensions if path has no extension
        p = Path(tex.path)
        if not p.suffix:
            for ext in (".tga", ".png", ".jpg", ".vtex"):
                candidates.append(root_path / (tex.path + ext))

        for cand in candidates:
            if cand.exists():
                tex.resolved_path = str(cand.resolve())
                tex.exists = True
                break


# ── Batch Scanning ───────────────────────────────────────────────────────

def scan_vmats(directory: str, materials_root: str = "") -> List[VmatMaterial]:
    """
    Recursively scan a directory for .vmat files and parse them all.
    """
    root = Path(directory)
    if not root.exists():
        return []

    vmats = []
    mat_root = materials_root or str(root)

    for vmat_path in sorted(root.rglob("*.vmat")):
        try:
            mat = parse_vmat(str(vmat_path), mat_root)
            vmats.append(mat)
        except Exception as e:
            logger.warning(f"Failed to parse {vmat_path.name}: {e}")

    logger.info(f"Scanned {len(vmats)} VMAT files in {directory}")
    return vmats


def get_all_texture_paths(vmats: List[VmatMaterial]) -> List[Tuple[str, TextureRole, str]]:
    """
    Extract all unique texture file paths from parsed VMATs.

    Returns: list of (resolved_path, role, vmat_relative_path)
    """
    seen = set()
    results = []
    for mat in vmats:
        for tex in mat.textures:
            if tex.resolved_path and tex.resolved_path not in seen:
                seen.add(tex.resolved_path)
                results.append((tex.resolved_path, tex.role, mat.relative_path))
    return results


def scan_textures_recursive(directory: str) -> List[str]:
    """
    Recursively scan for all texture image files in a directory tree.
    Returns absolute paths.
    """
    root = Path(directory)
    if not root.exists():
        return []

    texture_exts = {
        ".png", ".jpg", ".jpeg", ".tga", ".bmp",
        ".tiff", ".tif", ".webp", ".dds", ".exr",
    }

    files = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in texture_exts:
            files.append(str(f.resolve()))

    logger.info(f"Found {len(files)} texture files in {directory}")
    return files


# ── Surface Property → Material Type Mapping ─────────────────────────────

# Maps CS2 PhysicsSurfaceProperties values to material_enhancer MaterialType values
SURFACE_TO_MATERIAL: Dict[str, str] = {
    # Direct mappings
    "brick": "brick",
    "concrete": "concrete",
    "glass": "glass",
    "plastic": "plastic",
    "wood": "wood_raw",
    "tile": "tile_ceramic",
    "plaster": "plaster",
    "dirt": "terrain_dirt",
    "sand": "terrain_sand",
    "grass": "foliage",
    "gravel": "terrain_gravel",
    # Metal variants
    "metal": "metal_bare",
    "metalgrate": "metal_bare",
    "metalpanel": "metal_painted",
    "metalvent": "metal_bare",
    "solidmetal": "metal_bare",
    # Fabric / soft
    "cloth": "fabric_woven",
    "carpet": "carpet",
    "leather": "fabric_leather",
    "rubber": "rubber",
    # Construction
    "cardboard": "generic",
    "ceiling_tile": "tile_ceramic",
    "chainlink": "metal_bare",
    "sheetrock": "plaster",
    "drywall": "plaster",
    # Organic
    "mud": "terrain_dirt",
    "snow": "terrain_sand",
    "ice": "glass",
    "foliage": "foliage",
    "bark": "bark",
    "flesh": "skin",
    # Pavement
    "asphalt": "asphalt",
    "rock": "stone_rough",
    "boulder": "stone_rough",
}


def surface_property_to_material(surface_prop: str) -> Optional[str]:
    """
    Convert a CS2 PhysicsSurfaceProperties value to a MaterialType enum value.

    Returns None if the surface property is empty, "default", or unmapped —
    in which case the caller should fall back to AI-based material detection.
    """
    if not surface_prop or surface_prop.lower() in ("default", ""):
        return None
    return SURFACE_TO_MATERIAL.get(surface_prop.lower())


# ── VMAT Modification / Write-back ───────────────────────────────────────

def modify_vmat(vmat: VmatMaterial, changes: Dict[str, str]) -> str:
    """
    Apply key-value changes to a VMAT file and return the new file text.

    Parameters
    ----------
    vmat : VmatMaterial
        A parsed VMAT with raw_lines.
    changes : dict
        Key-value pairs to update or insert.
        Examples:
          {"TextureLayer1Roughness": "materials/brick/brick_rough.tga",
           "g_flMetalness": "0.100"}
        To set an inline roughness value:
          {"TextureLayer1Roughness": "[0.5 0.5 0.5 1.0]"}

    Returns
    -------
    str : The modified VMAT file text.
    """
    lines = list(vmat.raw_lines) if vmat.raw_lines else []
    if not lines:
        # Fallback: read from disk
        try:
            lines = Path(vmat.path).read_text(
                encoding="utf-8", errors="replace").splitlines(keepends=True)
        except Exception:
            return ""

    remaining = dict(changes)
    new_lines = []

    for line in lines:
        stripped = line.strip()

        # Try to match this line to a change key
        matched_key = None
        for change_key in list(remaining.keys()):
            # Match both "Key" "Value" and Key "Value" and Key Value
            patterns = [
                re.compile(
                    rf'^(\s*)"?{re.escape(change_key)}"?\s+".*?"(.*)$',
                    re.IGNORECASE),
                re.compile(
                    rf'^(\s*)"?{re.escape(change_key)}"?\s+\S+(.*)$',
                    re.IGNORECASE),
            ]
            for pat in patterns:
                m = pat.match(line.rstrip('\n').rstrip('\r'))
                if m:
                    matched_key = change_key
                    indent = m.group(1)
                    val = remaining.pop(change_key)
                    # Determine if value needs quoting
                    if val.startswith("[") or val.replace(".", "").replace("-", "").isdigit():
                        new_lines.append(f'{indent}"{change_key}"\t\t"{val}"\n')
                    else:
                        new_lines.append(f'{indent}"{change_key}"\t\t"{val}"\n')
                    break
            if matched_key:
                break
        else:
            new_lines.append(line if line.endswith('\n') else line + '\n')

    # Insert remaining changes before the last closing brace at depth 0
    if remaining:
        insert_idx = len(new_lines) - 1
        # Find the outermost closing brace
        for i in range(len(new_lines) - 1, -1, -1):
            if new_lines[i].strip() == "}":
                insert_idx = i
                break
        for k, v in remaining.items():
            new_lines.insert(insert_idx, f'\t"{k}"\t\t"{v}"\n')

    return "".join(new_lines)


def write_vmat(vmat: VmatMaterial, changes: Dict[str, str],
               output_path: Optional[str] = None) -> str:
    """
    Modify a VMAT and write it to disk.

    Parameters
    ----------
    vmat : VmatMaterial
        Parsed VMAT to modify.
    changes : dict
        Key-value changes to apply.
    output_path : str, optional
        Where to write. Defaults to the original VMAT path (in-place edit).

    Returns
    -------
    str : Path written to.
    """
    text = modify_vmat(vmat, changes)
    out = output_path or vmat.path
    Path(out).write_text(text, encoding="utf-8")
    logger.info(f"VMAT written: {out} ({len(changes)} changes)")
    return out
