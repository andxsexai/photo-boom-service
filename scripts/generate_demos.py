"""Generate static demo previews from sample portrait."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from config import DEMO_DIR
from core.enhancer import PhotoEnhancer
from core.filters import Style, STYLE_META

# Free portrait from Unsplash (license: free to use)
SAMPLE_URL = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80"


def download_sample(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading sample → {dest}")
        urllib.request.urlretrieve(SAMPLE_URL, dest)
    return dest


def generate():
    sample = download_sample(DEMO_DIR / "original.jpg")
    image = Image.open(sample).convert("RGB")
    enhancer = PhotoEnhancer(use_local_models=True, use_hf_fallback=False)

    previews = enhancer.preview_all(image)
    for p in previews:
        out = DEMO_DIR / f"{p.style.value}.jpg"
        p.image.save(out, format="JPEG", quality=88)
        print(f"  ✓ {STYLE_META[p.style]['name']} → {out.name}")

    image.save(DEMO_DIR / "original.jpg", quality=90)
    print("Demo samples ready in", DEMO_DIR)


if __name__ == "__main__":
    generate()
