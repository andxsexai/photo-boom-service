"""Download scripts for GFPGAN, CodeFormer, and Real-ESRGAN weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

DOWNLOADS = {
    "gfpgan": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "path": "gfpgan/GFPGANv1.4.pth",
    },
    "codeformer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "path": "codeformer/codeformer.pth",
    },
    "realesrgan": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "path": "realesrgan/RealESRGAN_x4plus.pth",
    },
}


def download(name: str, force: bool = False) -> Path:
    if name not in DOWNLOADS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(DOWNLOADS)}")

    meta = DOWNLOADS[name]
    dest = WEIGHTS_DIR / meta["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[skip] {name} already at {dest}")
        return dest

    print(f"[download] {name} → {dest}")
    urlretrieve(meta["url"], dest)
    print(f"[done] {name}")
    return dest


def download_all(force: bool = False) -> None:
    for name in DOWNLOADS:
        download(name, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Photo Boom model weights")
    parser.add_argument(
        "models",
        nargs="*",
        choices=[*DOWNLOADS, "all"],
        default=["all"],
        help="Models to download (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    targets = list(DOWNLOADS) if "all" in args.models else args.models
    for name in targets:
        try:
            download(name, force=args.force)
        except Exception as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
