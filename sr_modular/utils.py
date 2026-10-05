from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import config as cfg

RGB_WEIGHTS = torch.tensor([65.481, 128.553, 24.966], dtype=torch.float32, device=cfg.DEVICE)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
IMAGENET_MEAN_BATCH = IMAGENET_MEAN.to(cfg.DEVICE).unsqueeze(0)
IMAGENET_STD_BATCH = IMAGENET_STD.to(cfg.DEVICE).unsqueeze(0)


def set_seed(seed: int | None = None) -> None:
    cfg.set_seed(cfg.SEED if seed is None else seed)


def convert_image(img: torch.Tensor | Image.Image, source: str, target: str):
    assert source in {"pil", "[0, 1]", "[-1, 1]"}
    assert target in {"pil", "[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm", "y-channel"}

    if source == "pil":
        np_img = np.asarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(np_img).permute(2, 0, 1)
    elif source == "[-1, 1]":
        img = (img + 1.0) / 2.0
    elif source == "[0, 1]":
        pass

    if target == "pil":
        img01 = img.detach().cpu().clamp(0, 1)
        img_hwc = img01.permute(1, 2, 0).numpy()
        return Image.fromarray(np.round(img_hwc * 255.0).astype(np.uint8))
    if target == "[0, 255]":
        return 255.0 * img
    if target == "[0, 1]":
        return img
    if target == "[-1, 1]":
        return 2.0 * img - 1.0
    if target == "imagenet-norm":
        if img.ndim == 3:
            return (img - IMAGENET_MEAN) / IMAGENET_STD
        if img.ndim == 4:
            return (img - IMAGENET_MEAN_BATCH) / IMAGENET_STD_BATCH
        raise ValueError("Expected 3D or 4D tensor for imagenet normalization.")
    if target == "y-channel":
        if img.ndim != 4:
            raise ValueError("y-channel conversion expects a 4D tensor [N, C, H, W].")
        return torch.matmul((255.0 * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :]), RGB_WEIGHTS) / 255.0 + 16.0

    raise ValueError(f"Unsupported target: {target}")


def imagenet_denormalize(imgs: torch.Tensor) -> torch.Tensor:
    if imgs.ndim == 3:
        return imgs * IMAGENET_STD + IMAGENET_MEAN
    if imgs.ndim == 4:
        return imgs * IMAGENET_STD_BATCH + IMAGENET_MEAN_BATCH
    raise ValueError("Expected 3D or 4D tensor.")


def tensor_to_uint8(img: torch.Tensor, source: str) -> np.ndarray:
    if source == "[-1, 1]":
        img01 = ((img.detach().cpu() + 1.0) / 2.0).clamp(0, 1)
    elif source == "imagenet-norm":
        img01 = imagenet_denormalize(img.detach().cpu()).clamp(0, 1)
    elif source == "[0, 1]":
        img01 = img.detach().cpu().clamp(0, 1)
    else:
        raise ValueError(f"Unsupported source for tensor_to_uint8: {source}")

    img_hwc = img01.permute(1, 2, 0).numpy()
    return np.round(img_hwc * 255.0).astype(np.uint8)


def upsample_tensor_for_display(lr_tensor_01: torch.Tensor, out_hw: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    kwargs = {}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(
        lr_tensor_01.unsqueeze(0),
        size=out_hw,
        mode=mode,
        **kwargs,
    ).squeeze(0).clamp(0, 1)


def compute_psnr_ssim_from_tensors(sr_imgs: torch.Tensor, hr_imgs: torch.Tensor, source: str) -> tuple[float, float]:
    if source == "imagenet-norm":
        sr_01 = imagenet_denormalize(sr_imgs).clamp(0, 1)
        hr_01 = imagenet_denormalize(hr_imgs).clamp(0, 1)
    elif source == "[-1, 1]":
        sr_01 = ((sr_imgs + 1.0) / 2.0).clamp(0, 1)
        hr_01 = ((hr_imgs + 1.0) / 2.0).clamp(0, 1)
    else:
        raise ValueError(f"Unsupported source: {source}")

    sr_y = convert_image(sr_01, source="[0, 1]", target="y-channel")
    hr_y = convert_image(hr_01, source="[0, 1]", target="y-channel")

    psnr_values = []
    ssim_values = []
    for i in range(sr_y.size(0)):
        sr_np = sr_y[i].detach().cpu().numpy()
        hr_np = hr_y[i].detach().cpu().numpy()
        psnr_values.append(peak_signal_noise_ratio(hr_np, sr_np, data_range=255.0))
        ssim_values.append(structural_similarity(hr_np, sr_np, data_range=255.0))

    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def infer_image_array_from_namespace(namespace: dict[str, Any]) -> np.ndarray:
    if "IMAGE_ARRAY" in namespace and namespace["IMAGE_ARRAY"] is not None:
        return np.asarray(namespace["IMAGE_ARRAY"])

    candidates = []
    for name, value in namespace.items():
        if isinstance(value, np.ndarray) and value.ndim == 4 and value.shape[1:] == (256, 448, 3):
            candidates.append((name, value))

    if len(candidates) == 0:
        raise ValueError("No matching image array found. Assign IMAGE_ARRAY = your_array.")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple candidate arrays found {[name for name, _ in candidates]}. "
            "Assign the desired one explicitly with IMAGE_ARRAY = your_array."
        )

    name, arr = candidates[0]
    print(f"Auto-detected image array: {name}")
    return np.asarray(arr)


def validate_image_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4 or arr.shape[1:] != (256, 448, 3):
        raise ValueError(f"Expected [N, 256, 448, 3], got {arr.shape}.")
    if arr.size == 0:
        raise ValueError("IMAGE_ARRAY is empty.")
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError(f"Array values must lie in [0, 255], got [{arr.min()}, {arr.max()}].")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr
