"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = ""
    hf_space: str = "avans06/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN"
    use_hf_fallback: bool = True
    use_local_models: bool = False
    session_limit: int = 5
    output_dir: Path = Path("outputs")
    host: str = "0.0.0.0"
    port: int = int(os.environ.get("PORT", "8000"))
    codeformer_fidelity: float = 0.6
    public_url: str = ""

    # Local ComfyUI models (Mac only — auto-disabled in cloud)
    comfyui_root: Path = Path.home() / "ComfyUI"
    comfy_python: Path = Path.home() / "ComfyUI" / ".venv" / "bin" / "python"
    facerestore_model: str = "codeformer-v0.1.0.pth"
    facerestore_fallback: str = "GFPGANv1.4.pth"
    face_restore_visibility: float = 0.85


settings = Settings()

# Auto-enable local ComfyUI when available on this machine
_comfy_ready = (
    settings.comfy_python.exists()
    and (settings.comfyui_root / "models" / "facerestore_models" / settings.facerestore_model).exists()
)
if _comfy_ready and os.environ.get("USE_LOCAL_MODELS", "").lower() != "false":
    settings.use_local_models = True
    settings.use_hf_fallback = False

settings.output_dir.mkdir(parents=True, exist_ok=True)

COMFY_FACE_MODELS = settings.comfyui_root / "models" / "facerestore_models"
COMFY_FACE_DET = settings.comfyui_root / "models" / "facedetection"
DEMO_DIR = Path(__file__).resolve().parent / "static" / "demo"
