"""Local face restoration via ComfyUI models (subprocess)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import settings
from core.fallback import detect_artifacts

logger = logging.getLogger(__name__)

INFERENCE_SCRIPT = Path(__file__).resolve().parent.parent / "models" / "inference" / "face_restore.py"


class ComfyFaceRestorer:
    def __init__(self):
        self.python = settings.comfy_python
        self.script = INFERENCE_SCRIPT
        self.primary = settings.facerestore_model
        self.fallback = settings.facerestore_fallback
        self.weight = settings.codeformer_fidelity
        self.visibility = settings.face_restore_visibility
        self.available = self._check()

    def _check(self) -> bool:
        if not self.python.exists():
            logger.warning("ComfyUI python not found: %s", self.python)
            return False
        if not self.script.exists():
            return False
        model = settings.comfyui_root / "models" / "facerestore_models" / self.primary
        if not model.exists():
            logger.warning("Face model not found: %s", model)
            return False
        return True

    def _run(self, image: Image.Image, model_name: str) -> tuple[Image.Image, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "in.jpg"
            out = Path(tmp) / "out.jpg"
            image.save(inp, format="JPEG", quality=95)

            cmd = [
                str(self.python),
                str(self.script),
                str(inp),
                str(out),
                "--model",
                model_name,
                "--weight",
                str(self.weight),
                "--visibility",
                str(self.visibility),
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "COMFYUI_ROOT": str(settings.comfyui_root)},
            )

            # Script prints JSON on last line
            lines = [ln for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
            meta = {"model": model_name, "stderr": proc.stderr[-500:] if proc.stderr else ""}
            if lines:
                try:
                    meta.update(json.loads(lines[-1]))
                except json.JSONDecodeError:
                    pass

            if proc.returncode != 0 or not out.exists():
                raise RuntimeError(meta.get("error") or proc.stderr or "Face restore failed")

            bgr = cv2.imread(str(out))
            if bgr is None:
                raise RuntimeError("Restored image unreadable")
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), meta

    def restore(self, image: Image.Image) -> tuple[Image.Image, str, dict | None]:
        if not self.available:
            raise RuntimeError("ComfyUI models not available")

        restored, meta = self._run(image, self.primary)
        metrics = detect_artifacts(image, restored)

        if meta.get("fallback") or metrics.get("should_fallback"):
            logger.info("CodeFormer fallback → GFPGAN (%s)", metrics)
            restored, meta = self._run(image, self.fallback)
            return restored, "gfpgan", metrics

        model_label = "codeformer" if "codeformer" in self.primary.lower() else "gfpgan"
        return restored, model_label, metrics
