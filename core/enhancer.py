"""Neural network integration: face restoration, upscale, and style pipeline."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from PIL import Image

from core.comfy_restore import ComfyFaceRestorer
from core.filters import Style, apply_style, all_styles
from core.image_utils import gentle_face_prep, resize_for_preview

logger = logging.getLogger(__name__)
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
    def __init__(
        self,
        *,
        use_local_models: bool = True,
        use_hf_fallback: bool = False,
        hf_space: str = "",
        codeformer_fidelity: float = 0.6,
    ):
        self.use_hf_fallback = use_hf_fallback
        self.hf_space = hf_space
        self.codeformer_fidelity = max(0.5, min(0.7, codeformer_fidelity))
        self._hf_client = None
        self._comfy = ComfyFaceRestorer()
        self._local_ready = use_local_models and self._comfy.available
        logger.info("PhotoEnhancer: local_models=%s comfy=%s", self._local_ready, self._comfy.available)

    def restore_face(self, image: Image.Image) -> tuple[Image.Image, str, dict | None]:
        if self._local_ready:
            try:
                return self._comfy.restore(image)
            except Exception as exc:
                logger.warning("ComfyUI restore failed: %s → gentle prep", exc)
        if self.use_hf_fallback:
            return self._restore_face_hf(image)
        return gentle_face_prep(image), "local_prep", None

    def _restore_face_hf(self, image: Image.Image):
        try:
            from gradio_client import Client

            if self._hf_client is None:
                self._hf_client = Client(self.hf_space)
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            result = self._hf_client.predict(buf, "CodeFormer", self.codeformer_fidelity, api_name="/predict")
            if isinstance(result, str):
                restored = Image.open(result).convert("RGB")
            elif isinstance(result, (list, tuple)) and result:
                item = result[0]
                restored = Image.open(item).convert("RGB") if isinstance(item, str) else Image.open(item).convert("RGB")
            else:
                raise ValueError("Unexpected HF result")
            if restored.size != image.size:
                restored = restored.resize(image.size, Image.Resampling.LANCZOS)
            return restored, "codeformer_hf", None
        except Exception as exc:
            logger.warning("HF fallback failed: %s", exc)
            return gentle_face_prep(image), "local_prep", None

    def process_all(self, image: Image.Image, styles: list[Style] | None = None) -> PipelineResult:
        styles = styles or all_styles()
        restored, face_model, _ = self.restore_face(image)

        variants = [
            EnhancementResult(
                style=style,
                image=apply_style(restored, style),
                face_model=face_model,
                upscaled=False,
            )
            for style in styles
        ]
        return PipelineResult(original=image, variants=variants, face_model_used=face_model, upscaled=False)

    def process_single(self, image: Image.Image, style: Style) -> EnhancementResult:
        return self.process_all(image, styles=[style]).variants[0]

    def preview_all(self, image: Image.Image) -> list[PreviewResult]:
        thumb = resize_for_preview(image, PREVIEW_MAX_SIDE)
        try:
            base, _, _ = self.restore_face(thumb)
        except Exception:
            base = gentle_face_prep(thumb)
        return [PreviewResult(style=s, image=apply_style(base, s)) for s in all_styles()]

    def preview_single(self, image: Image.Image, style: Style) -> PreviewResult:
        return self.preview_all(image)[next(i for i, s in enumerate(all_styles()) if s == style)]
