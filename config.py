"""Application configuration."""

from __future__ import annotations

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
    port: int = 8000
    codeformer_fidelity: float = 0.6


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
