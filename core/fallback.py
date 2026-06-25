"""Artifact detection for face restoration fallback (CodeFormer → GFPGAN)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _to_gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def laplacian_variance(gray: np.ndarray) -> float:
    """Lower variance often indicates over-smoothed / plastic skin."""
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(lap))


def color_cast_score(original: Image.Image, restored: Image.Image) -> float:
    """Detect unnatural color shift between original and restored face region."""
    orig = np.asarray(original.convert("RGB"), dtype=np.float32)
    rest = np.asarray(restored.convert("RGB"), dtype=np.float32)
    h, w = min(orig.shape[0], rest.shape[0]), min(orig.shape[1], rest.shape[1])
    orig, rest = orig[:h, :w], rest[:h, :w]
    diff = np.abs(orig - rest).mean(axis=2)
    return float(diff.mean())


def blockiness_score(image: Image.Image) -> float:
    """Detect grid/block compression artifacts amplified by restoration."""
    gray = _to_gray_array(image)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0

    block = 8
    edges_h = np.abs(gray[:, block::block] - gray[:, block - 1 : -1 : block]).mean()
    edges_v = np.abs(gray[block::block, :] - gray[block - 1 : -1 : block, :]).mean()
    return float((edges_h + edges_v) / 2)


def detect_artifacts(
    original: Image.Image,
    restored: Image.Image,
    *,
    laplacian_threshold: float = 0.0008,
    color_threshold: float = 0.18,
    blockiness_threshold: float = 0.04,
) -> dict:
    """
    Returns quality metrics and whether fallback to GFPGAN is recommended.
    Pattern inspired by facerestore_advanced artifact detection.
    """
    gray = _to_gray_array(restored)
    lap_var = laplacian_variance(gray)
    color_score = color_cast_score(original, restored)
    block_score = blockiness_score(restored)

    over_smooth = lap_var < laplacian_threshold
    color_drift = color_score > color_threshold
    blocky = block_score > blockiness_threshold

    should_fallback = over_smooth or color_drift or blocky

    return {
        "laplacian_variance": lap_var,
        "color_cast_score": color_score,
        "blockiness_score": block_score,
        "over_smooth": over_smooth,
        "color_drift": color_drift,
        "blocky": blocky,
        "should_fallback": should_fallback,
    }
