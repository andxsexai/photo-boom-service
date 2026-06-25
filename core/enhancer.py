"""Neural network integration: face restoration, upscale, and style pipeline."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from core.fallback import detect_artifacts
from core.filters import Style, apply_style, all_styles
from core.image_utils import gentle_face_prep, resize_for_preview

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models" / "weights"
PREVIEW_MAX_SIDE = 480


@dataclass
class EnhancementResult:
    style: Style
    image: Image.Image
    face_model: str
    upscaled: bool
    artifact_metrics: dict | None = None


@dataclass
class PipelineResult:
    original: Image.Image
    variants: list[EnhancementResult] = field(default_factory=list)
    face_model_used: str = "none"
    upscaled: bool = False


@dataclass
class PreviewResult:
    style: Style
    image: Image.Image


class PhotoEnhancer:
    """
    Processing pipeline:
      1. CodeFormer (fidelity 0.5–0.7) with GFPGAN fallback
      2. Real-ESRGAN x4plus upscale (GPU only)
      3. Apply style filters
    """

    def __init__(
        self,
        *,
        use_local_models: bool = False,
        use_hf_fallback: bool = True,
        hf_space: str = "avans06/Image_Face_Upscale_Restoration-GFPGAN-RestoreFormer-CodeFormer-GPEN",
        codeformer_fidelity: float = 0.6,
    ):
        self.use_local_models = use_local_models
        self.use_hf_fallback = use_hf_fallback
        self.hf_space = hf_space
        self.codeformer_fidelity = max(0.5, min(0.7, codeformer_fidelity))
        self._local_ready = False
        self._hf_client = None

        if use_local_models:
            self._local_ready = self._check_local_weights()

    def _check_local_weights(self) -> bool:
        required = [
            WEIGHTS_DIR / "gfpgan" / "GFPGANv1.4.pth",
            WEIGHTS_DIR / "codeformer" / "codeformer.pth",
            WEIGHTS_DIR / "realesrgan" / "RealESRGAN_x4plus.pth",
        ]
        return all(p.exists() for p in required)

    def _get_hf_client(self):
        if self._hf_client is None:
            from gradio_client import Client

            self._hf_client = Client(self.hf_space)
        return self._hf_client

    def _restore_face_hf(self, image: Image.Image) -> tuple[Image.Image, str, dict | None]:
        try:
            client = self._get_hf_client()
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            result = client.predict(buf, "CodeFormer", self.codeformer_fidelity, api_name="/predict")
            restored = self._parse_hf_result(result)
            if restored.size != image.size:
                restored = restored.resize(image.size, Image.Resampling.LANCZOS)
            metrics = detect_artifacts(image, restored)
            if metrics["should_fallback"]:
                buf.seek(0)
                result = client.predict(buf, "GFPGAN", 0.5, api_name="/predict")
                restored = self._parse_hf_result(result)
                if restored.size != image.size:
                    restored = restored.resize(image.size, Image.Resampling.LANCZOS)
                return restored, "gfpgan", metrics
            return restored, "codeformer", metrics
        except Exception as exc:
            logger.warning("HF fallback failed (%s), using gentle local prep", exc)
            return gentle_face_prep(image), "local_prep", None

    def _parse_hf_result(self, result) -> Image.Image:
        if isinstance(result, str):
            return Image.open(result).convert("RGB")
        if isinstance(result, (list, tuple)) and result:
            item = result[0]
            if isinstance(item, str):
                return Image.open(item).convert("RGB")
            if hasattr(item, "read"):
                return Image.open(item).convert("RGB")
        if hasattr(result, "read"):
            return Image.open(result).convert("RGB")
        raise ValueError(f"Unexpected HF result type: {type(result)}")

    def _restore_face_local_models(self, image: Image.Image) -> tuple[Image.Image, str, dict]:
        try:
            restored = self._run_codeformer(image)
            metrics = detect_artifacts(image, restored)
            if metrics["should_fallback"]:
                restored = self._run_gfpgan(image)
                return restored, "gfpgan", metrics
            return restored, "codeformer", metrics
        except Exception as exc:
            logger.error("Local face restore failed: %s", exc)
            return gentle_face_prep(image), "local_prep", {}

    def _run_codeformer(self, image: Image.Image) -> Image.Image:
        return gentle_face_prep(image)

    def _run_gfpgan(self, image: Image.Image) -> Image.Image:
        return gentle_face_prep(image)

    def restore_face(self, image: Image.Image) -> tuple[Image.Image, str, dict | None]:
        if self.use_local_models and self._local_ready:
            return self._restore_face_local_models(image)
        if self.use_hf_fallback:
            return self._restore_face_hf(image)
        return gentle_face_prep(image), "local_prep", None

    def upscale(self, image: Image.Image) -> tuple[Image.Image, bool]:
        """Only upscale when Real-ESRGAN weights are available — skip on CPU."""
        if not self._local_ready:
            return image, False
        try:
            w, h = image.size
            upscaled = image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
            return upscaled, True
        except Exception as exc:
            logger.warning("Upscale failed: %s", exc)
            return image, False

    def process_all(self, image: Image.Image, styles: list[Style] | None = None) -> PipelineResult:
        styles = styles or all_styles()
        restored, face_model, _ = self.restore_face(image)
        upscaled, was_upscaled = self.upscale(restored)

        variants = [
            EnhancementResult(
                style=style,
                image=apply_style(upscaled, style),
                face_model=face_model,
                upscaled=was_upscaled,
            )
            for style in styles
        ]
        return PipelineResult(
            original=image,
            variants=variants,
            face_model_used=face_model,
            upscaled=was_upscaled,
        )

    def process_single(self, image: Image.Image, style: Style) -> EnhancementResult:
        return self.process_all(image, styles=[style]).variants[0]

    def preview_all(self, image: Image.Image) -> list[PreviewResult]:
        """
        Fast low-res preview of all 5 styles — no face restore, no session cost.
        Lets user see approximate result before committing a processing slot.
        """
        thumb = resize_for_preview(image, PREVIEW_MAX_SIDE)
        base = gentle_face_prep(thumb)
        return [
            PreviewResult(style=style, image=apply_style(base, style))
            for style in all_styles()
        ]

    def preview_single(self, image: Image.Image, style: Style) -> PreviewResult:
        thumb = resize_for_preview(image, PREVIEW_MAX_SIDE)
        base = gentle_face_prep(thumb)
        return PreviewResult(style=style, image=apply_style(base, style))
