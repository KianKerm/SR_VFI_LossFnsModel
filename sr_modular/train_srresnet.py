from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as cfg
from .models import SRResNet
from .utils import AverageMeter, clip_gradient, compute_psnr_ssim_from_tensors, save_json
from .loss_functions import build_loss_function


def get_srresnet_loss(loss_name: str) -> nn.Module:
    return build_loss_function(
        loss_name,
        scaling_factor=cfg.SCALING_FACTOR,
        data_range=2.0,
        huber_delta=cfg.HUBER_DELTA,
    )


def make_srresnet_model() -> SRResNet:
    return SRResNet(
        large_kernel_size=cfg.LARGE_KERNEL_SIZE,
        small_kernel_size=cfg.SMALL_KERNEL_SIZE,
        n_channels=cfg.N_CHANNELS,
        n_blocks=cfg.N_BLOCKS,
        scaling_factor=cfg.SCALING_FACTOR,
    )


def run_srresnet_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    print_freq: int | None = None,
    epoch: int | None = None,
    split_name: str = "train",
):
    print_freq = cfg.SRRESNET_PRINT_FREQ if print_freq is None else print_freq
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(loader):
        lr_imgs = lr_imgs.to(cfg.DEVICE, non_blocking=True)
        hr_imgs = hr_imgs.to(cfg.DEVICE, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.SRRESNET_GRAD_CLIP is not None:
                    clip_gradient(optimizer, cfg.SRRESNET_GRAD_CLIP)
                optimizer.step()

        batch_size = lr_imgs.size(0)
        psnr, ssim = compute_psnr_ssim_from_tensors(sr_imgs, hr_imgs, source="[-1, 1]")
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(psnr, batch_size)
        ssim_meter.update(ssim, batch_size)

        if is_train and batch_idx % print_freq == 0:
            ep = "?" if epoch is None else epoch
            print(
                f"[SRResNet][{split_name}] epoch={ep} batch={batch_idx}/{len(loader)} "
                f"loss={loss_meter.avg:.4f} psnr={psnr_meter.avg:.3f} ssim={ssim_meter.avg:.4f}"
            )

    return {"loss": loss_meter.avg, "psnr": psnr_meter.avg, "ssim": ssim_meter.avg}


def save_srresnet_checkpoint(
    model_name: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: dict,
) -> Path:
    path = cfg.CHECKPOINT_DIR / f"{model_name}.pt"
    payload = {
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": {
            "scaling_factor": cfg.SCALING_FACTOR,
            "crop_size": cfg.CROP_SIZE,
            "large_kernel_size": cfg.LARGE_KERNEL_SIZE,
            "small_kernel_size": cfg.SMALL_KERNEL_SIZE,
            "n_channels": cfg.N_CHANNELS,
            "n_blocks": cfg.N_BLOCKS,
        },
    }
    torch.save(payload, path)
    return path


def train_single_srresnet(
    loss_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int | None = None,
    lr: float | None = None,
) -> dict:
    num_epochs = cfg.SRRESNET_EPOCHS if num_epochs is None else num_epochs
    lr = cfg.SRRESNET_LR if lr is None else lr

    model_name = f"SRResNet_{loss_name.upper()}"
    print(f"\n===== Training {model_name} =====")

    model = make_srresnet_model().to(cfg.DEVICE)
    criterion = get_srresnet_loss(loss_name).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_psnr": [],
        "train_ssim": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    best_val_psnr = -float("inf")
    best_ckpt_path = None

    for epoch in range(num_epochs):
        train_metrics = run_srresnet_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            print_freq=cfg.SRRESNET_PRINT_FREQ,
            epoch=epoch,
            split_name="train",
        )
        val_metrics = run_srresnet_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            split_name="val",
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_psnr"].append(train_metrics["psnr"])
        history["train_ssim"].append(train_metrics["ssim"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["val_ssim"].append(val_metrics["ssim"])

        print(
            f"[{model_name}] epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_psnr={val_metrics['psnr']:.3f} "
            f"val_ssim={val_metrics['ssim']:.4f}"
        )

        ckpt_path = save_srresnet_checkpoint(
            model_name=model_name,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            history=history,
        )

        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            best_ckpt_path = cfg.CHECKPOINT_DIR / f"{model_name}_best.pt"
            torch.save(torch.load(ckpt_path, map_location="cpu"), best_ckpt_path)

    history_path = cfg.METRICS_DIR / f"{model_name}_history.json"
    save_json(history, history_path)

    return {
        "model_name": model_name,
        "loss_name": loss_name.upper(),
        "checkpoint_path": str(cfg.CHECKPOINT_DIR / f"{model_name}.pt"),
        "best_checkpoint_path": str(best_ckpt_path),
        "history_path": str(history_path),
        "history": history,
    }


def train_multiple_srresnet(
    loss_names: list[str] | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int | None = None,
    lr: float | None = None,
) -> dict[str, dict]:
    loss_names = cfg.SRRESNET_LOSS_FUNCTIONS if loss_names is None else loss_names
    results = {}
    for loss_name in loss_names:
        result = train_single_srresnet(
            loss_name=loss_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
        )
        results[result["loss_name"]] = result
    return results
