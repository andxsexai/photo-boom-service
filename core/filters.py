"""Five premium photo processing styles for Photo Boom."""

from __future__ import annotations

import io
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

LUTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "luts"


class Style(str, Enum):
    UPGRADE = "upgrade"
    QUIET_LUXURY = "quiet_luxury"
    CINEMATIC_TWILIGHT = "cinematic_twilight"
    BW_NOIR = "bw_noir"
    MAGAZINE_COVER = "magazine_cover"


STYLE_META = {
    Style.UPGRADE: {"num": 1, "name": "Upgrade", "description": "Чистая ретушь + детализация без изменения цвета"},
    Style.QUIET_LUXURY: {"num": 2, "name": "Quiet Luxury", "description": "Бежево-графитовый LUT, премиум-эстетика"},
    Style.CINEMATIC_TWILIGHT: {"num": 3, "name": "Cinematic Twilight", "description": "Глубокие тени, голливудский контраст"},
    Style.BW_NOIR: {"num": 4, "name": "B&W Noir", "description": "Монохром, лёгкое зерно, глянцевый портрет"},
    Style.MAGAZINE_COVER: {"num": 5, "name": "Magazine Cover", "description": "Quiet Luxury + минималистичный оверлей"},
}


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _adjust_luminance(bgr: np.ndarray, contrast: float = 1.0, brightness: float = 0.0) -> np.ndarray:
    """Adjust in LAB space — preserves color, avoids muddy shifts."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]
    l = np.clip((l - 128) * contrast + 128 + brightness, 0, 255)
    lab[:, :, 0] = l
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def load_cube_lut(cube_path: Path) -> np.ndarray | None:
    if not cube_path.exists():
        return None
    size = 0
    table: list[list[float]] = []
    with cube_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if line[0].isdigit() or line[0] == "-":
                parts = line.split()
                if len(parts) >= 3:
                    table.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if size == 0 or len(table) != size**3:
        return None
    return np.array(table, dtype=np.float32).reshape(size, size, size, 3)


def apply_3d_lut(image: Image.Image, lut: np.ndarray, strength: float = 1.0) -> Image.Image:
    """Apply 3D LUT with optional blend back to original."""
    size = lut.shape[0]
    img = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    orig = img.copy()
    scaled = img * (size - 1)

    r, g, b = scaled[..., 0], scaled[..., 1], scaled[..., 2]
    r0, g0, b0 = np.floor(r).astype(int), np.floor(g).astype(int), np.floor(b).astype(int)
    r1 = np.clip(r0 + 1, 0, size - 1)
    g1 = np.clip(g0 + 1, 0, size - 1)
    b1 = np.clip(b0 + 1, 0, size - 1)
    dr, dg, db = r - r0, g - g0, b - b0
    dr, dg, db = dr[..., None], dg[..., None], db[..., None]

    c000, c100 = lut[r0, g0, b0], lut[r1, g0, b0]
    c010, c110 = lut[r0, g1, b0], lut[r1, g1, b0]
    c001, c101 = lut[r0, g0, b1], lut[r1, g0, b1]
    c011, c111 = lut[r0, g1, b1], lut[r1, g1, b1]

    c00 = c000 * (1 - dr) + c100 * dr
    c01 = c001 * (1 - dr) + c101 * dr
    c10 = c010 * (1 - dr) + c110 * dr
    c11 = c011 * (1 - dr) + c111 * dr
    c0 = c00 * (1 - dg) + c10 * dg
    c1 = c01 * (1 - dg) + c11 * dg
    graded = c0 * (1 - db) + c1 * db

    if strength < 1.0:
        graded = orig * (1 - strength) + graded * strength

    out = np.clip(graded * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _generate_quiet_luxury_lut(size: int = 17) -> np.ndarray:
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                r, g, b = ri / (size - 1), gi / (size - 1), bi / (size - 1)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                # Subtle warm beige lift, soft graphite shadows
                warm = 1.0 + 0.04 * (1 - abs(lum - 0.45) * 2)
                lut[ri, gi, bi] = [
                    np.clip(r * warm + 0.02 * (1 - lum), 0, 1),
                    np.clip(g * warm * 0.98 + 0.015, 0, 1),
                    np.clip(b * (1 - 0.03 * lum), 0, 1),
                ]
    return lut


def _generate_cinematic_twilight_lut(size: int = 17) -> np.ndarray:
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                r, g, b = ri / (size - 1), gi / (size - 1), bi / (size - 1)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                # Gentle S-curve, not aggressive contrast
                lum_c = lum + 0.08 * (lum - 0.5) * (1 - abs(lum - 0.5) * 2)
                scale = lum_c / (lum + 1e-6)
                scale = np.clip(scale, 0.85, 1.15)
                te = max(0, 0.35 - lum) * 0.06
                tw = max(0, lum - 0.65) * 0.05
                lut[ri, gi, bi] = [
                    np.clip(r * scale + tw - te * 0.2, 0, 1),
                    np.clip(g * scale * 0.98 + tw * 0.3, 0, 1),
                    np.clip(b * scale + te * 0.5 + tw * 0.15, 0, 1),
                ]
    return lut


def _get_lut(name: str) -> np.ndarray:
    loaded = load_cube_lut(LUTS_DIR / f"{name}.cube")
    if loaded is not None:
        return loaded
    if name == "quiet_luxury":
        return _generate_quiet_luxury_lut()
    if name == "cinematic_twilight":
        return _generate_cinematic_twilight_lut()
    raise ValueError(f"No LUT for {name}")


def apply_upgrade(image: Image.Image) -> Image.Image:
    """Variant 1: clean retouch, preserve natural colors."""
    bgr = _pil_to_bgr(image)
    smooth = cv2.bilateralFilter(bgr, d=7, sigmaColor=35, sigmaSpace=35)
    result = _adjust_luminance(smooth, contrast=1.04, brightness=2)
    return _bgr_to_pil(result)


def apply_quiet_luxury(image: Image.Image) -> Image.Image:
    base = apply_upgrade(image)
    return apply_3d_lut(base, _get_lut("quiet_luxury"), strength=0.75)


def apply_cinematic_twilight(image: Image.Image) -> Image.Image:
    base = apply_upgrade(image)
    graded = apply_3d_lut(base, _get_lut("cinematic_twilight"), strength=0.7)
    bgr = _pil_to_bgr(graded)
    h, w = bgr.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h * 0.42  # face-centered vignette
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1 - 0.18 * (dist / max_dist) ** 1.8
    bgr = (bgr.astype(np.float32) * vignette[..., None]).astype(np.uint8)
    return _bgr_to_pil(bgr)


def apply_bw_noir(image: Image.Image) -> Image.Image:
    bgr = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = np.clip((gray - 128) * 1.12 + 128, 0, 255)
    # Subtle film grain
    rng = np.random.default_rng(42)
    grain = rng.normal(0, 2.5, gray.shape)
    gray = np.clip(gray + grain, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _bgr_to_pil(result)


def apply_magazine_cover(image: Image.Image, brand: str = "ANDX") -> Image.Image:
    base = apply_quiet_luxury(image)
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", max(14, w // 22))
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", max(9, w // 38))
    except OSError:
        font_large = ImageFont.load_default()
        font_small = font_large
    draw.rectangle([0, 0, w, h // 8], fill=(0, 0, 0, 50))
    draw.text((w // 2, h // 16), brand.upper(), fill=(255, 255, 255, 200), font=font_large, anchor="mm")
    draw.text((w // 2, h - h // 12), "PHOTO BOOM", fill=(255, 255, 255, 160), font=font_small, anchor="mm")
    draw.rectangle([8, 8, w - 8, h - 8], outline=(255, 255, 255, 80), width=1)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


STYLE_HANDLERS = {
    Style.UPGRADE: apply_upgrade,
    Style.QUIET_LUXURY: apply_quiet_luxury,
    Style.CINEMATIC_TWILIGHT: apply_cinematic_twilight,
    Style.BW_NOIR: apply_bw_noir,
    Style.MAGAZINE_COVER: apply_magazine_cover,
}


def apply_style(image: Image.Image, style: Style) -> Image.Image:
    return STYLE_HANDLERS[style](image)


def image_to_bytes(image: Image.Image, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        image.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        image.save(buf, format=fmt)
    return buf.getvalue()


def all_styles() -> list[Style]:
    return list(Style)
