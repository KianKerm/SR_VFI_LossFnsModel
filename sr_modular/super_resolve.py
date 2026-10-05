from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as cfg
from .models import Generator
from .train_srresnet import make_srresnet_model
from .utils import (
    AverageMeter,
    compute_psnr_ssim_from_tensors,
    imagenet_denormalize,
    load_json,
    tensor_to_uint8,
    upsample_tensor_for_display,
)


def load_srresnet_model_from_checkpoint(path: str | Path):
    ckpt = torch.load(path, map_location="cpu")
    model = make_srresnet_model()
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(cfg.DEVICE).eval()


def load_srgan_generator_from_checkpoint(path: str | Path):
    ckpt = torch.load(path, map_location="cpu")
    generator = Generator(
        large_kernel_size=cfg.LARGE_KERNEL_SIZE,
        small_kernel_size=cfg.SMALL_KERNEL_SIZE,
        n_channels=cfg.N_CHANNELS,
        n_blocks=cfg.N_BLOCKS,
        scaling_factor=cfg.SCALING_FACTOR,
    )
    generator.load_state_dict(ckpt["generator_state_dict"])
    return generator.to(cfg.DEVICE).eval()


def discover_best_model_paths(loss_names: list[str] | None = None):
    loss_names = cfg.SRRESNET_LOSS_FUNCTIONS if loss_names is None else loss_names

    srresnet_paths = {}
    srgan_paths = {}
    for loss_name in loss_names:
        suffix = loss_name.upper()

        srresnet_best = cfg.CHECKPOINT_DIR / f"SRResNet_{suffix}_best.pt"
        srresnet_last = cfg.CHECKPOINT_DIR / f"SRResNet_{suffix}.pt"
        if srresnet_best.exists():
            srresnet_paths[suffix] = srresnet_best
        elif srresnet_last.exists():
            srresnet_paths[suffix] = srresnet_last

        srgan_best = cfg.CHECKPOINT_DIR / f"SRGAN_ResNet_{suffix}_best.pt"
        srgan_last = cfg.CHECKPOINT_DIR / f"SRGAN_ResNet_{suffix}.pt"
        if srgan_best.exists():
            srgan_paths[suffix] = srgan_best
        elif srgan_last.exists():
            srgan_paths[suffix] = srgan_last

    return srresnet_paths, srgan_paths


@torch.no_grad()
def evaluate_sr_model(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    mse = nn.MSELoss().to(cfg.DEVICE)

    for lr_imgs, hr_imgs in loader:
        lr_imgs = lr_imgs.to(cfg.DEVICE, non_blocking=True)
        hr_imgs = hr_imgs.to(cfg.DEVICE, non_blocking=True)

        sr_imgs = model(lr_imgs)
        loss = mse(sr_imgs, hr_imgs)
        psnr, ssim = compute_psnr_ssim_from_tensors(sr_imgs, hr_imgs, source="[-1, 1]")

        batch_size = lr_imgs.size(0)
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(psnr, batch_size)
        ssim_meter.update(ssim, batch_size)

    return {"mse_loss": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def build_metrics_table(
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_names: list[str] | None = None,
) -> pd.DataFrame:
    srresnet_paths, srgan_paths = discover_best_model_paths(loss_names=loss_names)
    rows = []

    for suffix, path in srresnet_paths.items():
        model = load_srresnet_model_from_checkpoint(path)
        val_metrics = evaluate_sr_model(model, val_loader)
        test_metrics = evaluate_sr_model(model, test_loader)
        rows.append(
            {
                "model": f"SRResNet_{suffix}",
                "family": suffix,
                "type": "SRResNet",
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "test_psnr": test_metrics["psnr"],
                "test_ssim": test_metrics["ssim"],
            }
        )

    for suffix, path in srgan_paths.items():
        generator = load_srgan_generator_from_checkpoint(path)
        val_metrics = evaluate_sr_model(generator, val_loader)
        test_metrics = evaluate_sr_model(generator, test_loader)
        rows.append(
            {
                "model": f"SRGAN_ResNet_{suffix}",
                "family": suffix,
                "type": "SRGAN",
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
                "test_psnr": test_metrics["psnr"],
                "test_ssim": test_metrics["ssim"],
            }
        )

    if not rows:
        raise FileNotFoundError("No model checkpoints found for evaluation.")

    return pd.DataFrame(rows).sort_values(["family", "type"]).reset_index(drop=True)


def get_history_if_present(model_name: str) -> dict | None:
    history_path = cfg.METRICS_DIR / f"{model_name}_history.json"
    if history_path.exists():
        return load_json(history_path)
    return None


def plot_srresnet_histories(loss_names: list[str] | None = None) -> None:
    loss_names = cfg.SRRESNET_LOSS_FUNCTIONS if loss_names is None else loss_names
    histories = {}
    for loss_name in loss_names:
        model_name = f"SRResNet_{loss_name.upper()}"
        hist = get_history_if_present(model_name)
        if hist is not None:
            histories[loss_name.upper()] = hist
    if not histories:
        raise FileNotFoundError("No SRResNet history JSON files found.")

    plt.figure(figsize=(10, 5))
    for suffix, hist in histories.items():
        plt.plot(hist["train_loss"], label=f"{suffix} train")
        plt.plot(hist["val_loss"], linestyle="--", label=f"{suffix} val")
    plt.title("SRResNet: train/val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for suffix, hist in histories.items():
        plt.plot(hist["train_psnr"], label=f"{suffix} train")
        plt.plot(hist["val_psnr"], linestyle="--", label=f"{suffix} val")
    plt.title("SRResNet: train/val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (Y channel)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for suffix, hist in histories.items():
        plt.plot(hist["train_ssim"], label=f"{suffix} train")
        plt.plot(hist["val_ssim"], linestyle="--", label=f"{suffix} val")
    plt.title("SRResNet: train/val SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM (Y channel)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_srgan_histories(loss_names: list[str] | None = None) -> None:
    loss_names = cfg.SRRESNET_LOSS_FUNCTIONS if loss_names is None else loss_names
    histories = {}
    for loss_name in loss_names:
        model_name = f"SRGAN_ResNet_{loss_name.upper()}"
        hist = get_history_if_present(model_name)
        if hist is not None:
            histories[loss_name.upper()] = hist
    if not histories:
        raise FileNotFoundError("No SRGAN history JSON files found.")

    plt.figure(figsize=(10, 5))
    for suffix, hist in histories.items():
        plt.plot(hist["train_content_loss"], label=f"{suffix} train")
        plt.plot(hist["val_content_loss"], linestyle="--", label=f"{suffix} val")
    plt.title("SRGAN: train/val content loss")
    plt.xlabel("Epoch")
    plt.ylabel("Content loss (VGG-space MSE)")
    plt.legend()
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def get_test_sample_for_visualization(loader: DataLoader, sample_index: int = 0):
    return loader.dataset[sample_index]


@torch.no_grad()
def generate_family_visuals(loss_name: str, test_loader: DataLoader, sample_index: int = 0):
    suffix = loss_name.upper()
    srresnet_paths, srgan_paths = discover_best_model_paths([suffix])

    if suffix not in srresnet_paths:
        raise FileNotFoundError(f"No SRResNet checkpoint found for {suffix}.")
    if suffix not in srgan_paths:
        raise FileNotFoundError(f"No SRGAN checkpoint found for {suffix}.")

    srresnet = load_srresnet_model_from_checkpoint(srresnet_paths[suffix])
    srgan = load_srgan_generator_from_checkpoint(srgan_paths[suffix])
    lr_img_imagenet, hr_img_minus1_to1 = get_test_sample_for_visualization(test_loader, sample_index=sample_index)

    lr_img_imagenet_b = lr_img_imagenet.unsqueeze(0).to(cfg.DEVICE)
    srresnet_out = srresnet(lr_img_imagenet_b).squeeze(0).cpu()
    srgan_out = srgan(lr_img_imagenet_b).squeeze(0).cpu()

    hr_uint8 = tensor_to_uint8(hr_img_minus1_to1.cpu(), source="[-1, 1]")
    lr_01 = imagenet_denormalize(lr_img_imagenet.cpu()).clamp(0, 1)
    lr_display_01 = upsample_tensor_for_display(lr_01, out_hw=(hr_uint8.shape[0], hr_uint8.shape[1]), mode="nearest")
    bicubic_01 = upsample_tensor_for_display(lr_01, out_hw=(hr_uint8.shape[0], hr_uint8.shape[1]), mode="bicubic")

    lr_up_uint8 = np.round(lr_display_01.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    bicubic_uint8 = np.round(bicubic_01.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    srresnet_uint8 = tensor_to_uint8(srresnet_out, source="[-1, 1]")
    srgan_uint8 = tensor_to_uint8(srgan_out, source="[-1, 1]")

    return {
        "suffix": suffix,
        "LR": lr_up_uint8,
        "Bicubic": bicubic_uint8,
        "SRResNet": srresnet_uint8,
        "SRGAN": srgan_uint8,
        "Ground Truth": hr_uint8,
    }


def show_comparison_strips(test_loader: DataLoader, loss_names: list[str] | None = None, sample_index: int = 0) -> None:
    loss_names = cfg.SRRESNET_LOSS_FUNCTIONS if loss_names is None else loss_names
    visuals = [generate_family_visuals(loss_name, test_loader=test_loader, sample_index=sample_index) for loss_name in loss_names]
    titles = ["LR", "Bicubic", "SRResNet", "SRGAN", "Ground Truth"]

    for idx, family in enumerate(visuals, start=1):
        print(f"Comparison strip {idx}: SRResNet training loss = {family['suffix']}")

        fig, axes = plt.subplots(1, len(titles), figsize=(4 * len(titles), 4.8))
        if len(titles) == 1:
            axes = [axes]

        for ax, title in zip(axes, titles):
            ax.imshow(family[title])
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        fig.suptitle(
            f"Test image comparison strip | SRResNet training loss: {family['suffix']}",
            y=0.98,
            fontsize=15,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        plt.show()


def absolute_difference_map(pred_uint8: np.ndarray, gt_uint8: np.ndarray) -> np.ndarray:
    """Return the per-pixel mean absolute difference normalized to [0, 1]."""
    pred = pred_uint8.astype(np.float32) / 255.0
    gt = gt_uint8.astype(np.float32) / 255.0
    return np.abs(pred - gt).mean(axis=2).clip(0.0, 1.0)


def show_absolute_difference_heatmaps(test_loader: DataLoader, loss_names: list[str] | None = None, sample_index: int = 0) -> None:
    loss_names = cfg.SRRESNET_LOSS_FUNCTIONS if loss_names is None else loss_names
    visuals = [generate_family_visuals(loss_name, test_loader=test_loader, sample_index=sample_index) for loss_name in loss_names]
    heatmap_titles = ["Bicubic", "SRResNet", "SRGAN"]

    fig, axes = plt.subplots(len(visuals), len(heatmap_titles), figsize=(5 * len(heatmap_titles), 4 * len(visuals)))
    if len(visuals) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, family in enumerate(visuals):
        gt = family["Ground Truth"]
        for col_idx, title in enumerate(heatmap_titles):
            diff = absolute_difference_map(family[title], gt)
            ax = axes[row_idx, col_idx]
            im = ax.imshow(diff, cmap="hot", vmin=0.0, vmax=1.0)
            ax.set_title(f"{family['suffix']} - {title} | abs diff vs GT")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
