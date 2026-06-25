#!/usr/bin/env python3
"""
Face restoration using local ComfyUI models (CodeFormer / GFPGAN).
Run with ComfyUI venv: ComfyUI/venv/bin/python face_restore.py ...

Uses ComfyUI-ReActor embedded r_facelib / r_basicsr — no ComfyUI server needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

# --- paths ---
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", Path.home() / "ComfyUI"))
REACTOR_ROOT = COMFY_ROOT / "custom_nodes" / "ComfyUI-ReActor"
MODELS_FACE = COMFY_ROOT / "models" / "facerestore_models"
MODELS_DET = COMFY_ROOT / "models" / "facedetection"

sys.path.insert(0, str(REACTOR_ROOT))
os.chdir(str(REACTOR_ROOT))

import scripts.r_archs.codeformer_arch  # noqa: F401 — register CodeFormer in ARCH_REGISTRY
from r_basicsr.utils.registry import ARCH_REGISTRY  # noqa: E402
from r_chainner import model_loading  # noqa: E402
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper  # noqa: E402


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def img2tensor(img: np.ndarray, bgr2rgb: bool = True, float32: bool = True) -> torch.Tensor:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img.transpose(2, 0, 1))
    if float32:
        t = t.float() / 255.0
    return t


def tensor2img(tensor: torch.Tensor, rgb2bgr: bool = True, min_max=(-1, 1)) -> np.ndarray:
    t = tensor.squeeze(0).float().cpu().clamp_(*min_max)
    t = (t - min_max[0]) / (min_max[1] - min_max[0] + 1e-8)
    img = t.numpy().transpose(1, 2, 0)
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return (img * 255.0).round().astype(np.uint8)


def load_restore_net(model_name: str, device: torch.device):
    model_path = MODELS_FACE / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if "codeformer" in model_name.lower():
        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["params_ema"])
        net.eval()
        return net, "codeformer"

    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    net = model_loading.load_state_dict(sd).eval().to(device)
    return net, "gfpgan"


def restore_cropped(cropped: np.ndarray, net, model_type: str, weight: float, device: torch.device) -> np.ndarray:
    face_size = 512
    cropped = cv2.resize(cropped, (face_size, face_size), interpolation=cv2.INTER_CUBIC)
    t = img2tensor(cropped, bgr2rgb=True, float32=True)
    normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    t = t.unsqueeze(0).to(device)

    with torch.no_grad():
        if model_type == "codeformer":
            out = net(t, w=weight)[0]
        else:
            out = net(t)[0]
    return tensor2img(out, rgb2bgr=True, min_max=(-1, 1))


def patch_detection_paths():
    """Use local ComfyUI facedetection weights instead of downloading."""
    det_file = MODELS_DET / "detection_Resnet50_Final.pth"
    parse_file = MODELS_DET / "parsing_parsenet.pth"
    if not det_file.exists():
        raise FileNotFoundError(f"Face detector not found: {det_file}")

    import r_facelib.detection as det_mod

    def _init_retinaface(model_name, half=False, device="cuda"):
        from copy import deepcopy
        from r_facelib.detection.retinaface.retinaface import RetinaFace

        model = RetinaFace(network_name="resnet50", half=half)
        load_net = torch.load(det_file, map_location="cpu", weights_only=False)
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)
        model.load_state_dict(load_net, strict=True)
        model.eval()
        return model.to(device)

    det_mod.init_retinaface_model = _init_retinaface

    if parse_file.exists():
        import r_facelib.parsing as parse_mod

        def _init_parsing(model_name="parsenet", device="cuda"):
            from r_facelib.parsing.parsenet import ParseNet

            model = ParseNet()
            ckpt = torch.load(parse_file, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt, strict=True)
            model.eval()
            return model.to(device)

        parse_mod.init_parsing_model = _init_parsing


def restore_image(
    input_path: Path,
    output_path: Path,
    model_name: str,
    codeformer_weight: float = 0.6,
    visibility: float = 0.85,
) -> dict:
    device = get_device()
    patch_detection_paths()

    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    net, model_type = load_restore_net(model_name, device)

    helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=False,
        device=device,
    )

    helper.read_image(img)
    num_faces = helper.get_face_landmarks_5(only_center_face=True, eye_dist_threshold=5)
    if num_faces == 0:
        helper.clean_all()
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {"faces": 0, "model": model_name, "device": str(device), "fallback": True}

    helper.align_warp_face()
    for cropped in helper.cropped_faces:
        restored = restore_cropped(cropped, net, model_type, codeformer_weight, device)
        if visibility < 1.0:
            restored = (cropped * (1 - visibility) + restored * visibility).astype(np.uint8)
        helper.add_restored_face(restored)

    helper.get_inverse_affine(None)
    result = helper.paste_faces_to_input_image(upsample_img=img)
    helper.clean_all()

    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"faces": num_faces, "model": model_name, "device": str(device), "fallback": False}


def main():
    parser = argparse.ArgumentParser(description="Face restore with ComfyUI local models")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="codeformer-v0.1.0.pth")
    parser.add_argument("--weight", type=float, default=0.6)
    parser.add_argument("--visibility", type=float, default=0.85)
    args = parser.parse_args()

    try:
        meta = restore_image(args.input, args.output, args.model, args.weight, args.visibility)
        print(json.dumps({"ok": True, **meta}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
