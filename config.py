"""Application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = ""
    hf_space: str = "avans06/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN"
    use_hf_fallback: bool = False
    use_local_models: bool = True
    session_limit: int = 5
    output_dir: Path = Path("outputs")
    host: str = "0.0.0.0"
    port: int = 8000
    codeformer_fidelity: float = 0.6

    # Local ComfyUI models (auto-detected on Mac)
    comfyui_root: Path = Path.home() / "ComfyUI"
    comfy_python: Path = Path.home() / "ComfyUI" / ".venv" / "bin" / "python"
    facerestore_model: str = "codeformer-v0.1.0.pth"
    facerestore_fallback: str = "GFPGANv1.4.pth"
    face_restore_visibility: float = 0.85


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)

COMFY_FACE_MODELS = settings.comfyui_root / "models" / "facerestore_models"
COMFY_FACE_DET = settings.comfyui_root / "models" / "facedetection"
DEMO_DIR = Path(__file__).resolve().parent / "static" / "demo"
