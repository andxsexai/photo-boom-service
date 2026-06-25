"""Generate diverse demo previews — one unique portrait per style."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from config import DEMO_DIR
from core.enhancer import PhotoEnhancer
from core.filters import STYLE_META, Style

# Free Unsplash portraits — diverse women, commercial use allowed
SAMPLES: dict[Style, dict] = {
    Style.UPGRADE: {
        "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=900&q=85",
        "label": "Studio portrait",
    },
    Style.QUIET_LUXURY: {
        "url": "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=900&q=85",
        "label": "Soft elegance",
    },
    Style.CINEMATIC_TWILIGHT: {
        "url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=900&q=85",
        "label": "Natural light",
    },
    Style.BW_NOIR: {
        "url": "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=900&q=85",
        "label": "Classic beauty",
    },
    Style.MAGAZINE_COVER: {
        "url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=900&q=85",
        "label": "Fashion editorial",
    },
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {dest.name}")
    urllib.request.urlretrieve(url, dest)


def generate(force: bool = False) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    enhancer = PhotoEnhancer(use_local_models=True, use_hf_fallback=False)
    manifest: list[dict] = []

    for style, meta in SAMPLES.items():
        orig_path = DEMO_DIR / f"original_{style.value}.jpg"
        out_path = DEMO_DIR / f"{style.value}.jpg"

        if force or not orig_path.exists():
            download(meta["url"], orig_path)

        image = Image.open(orig_path).convert("RGB")
        # Full pipeline: CodeFormer + style
        result = enhancer.process_single(image, style)
        result.image.save(out_path, format="JPEG", quality=90)
        image.save(orig_path, format="JPEG", quality=90)

        manifest.append({
            "style": style.value,
            "name": STYLE_META[style]["name"],
            "num": STYLE_META[style]["num"],
            "label": meta["label"],
            "original": f"original_{style.value}.jpg",
            "processed": f"{style.value}.jpg",
        })
        print(f"  ✓ {STYLE_META[style]['name']} — {meta['label']}")

    (DEMO_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nDemo ready → {DEMO_DIR} ({len(manifest)} styles)")


if __name__ == "__main__":
    force = "--force" in sys.argv
    generate(force=force)
