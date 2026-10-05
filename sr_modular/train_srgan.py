from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as cfg
from .models import Discriminator, Generator, TruncatedVGG19
from .utils import AverageMeter, clip_gradient, compute_psnr_ssim_from_tensors, convert_image, save_json


def discover_srresnet_checkpoints(loss_names: list[str] | None = None) -> dict[str, Path]:
    loss_names = cfg.SRRESNET_LOSS_FUNCTIONS if loss_names is None else loss_names
    ckpts = {}
    for loss_name in loss_names:
        suffix = loss_name.upper()
        preferred = cfg.CHECKPOINT_DIR / f"SRResNet_{suffix}_best.pt"
        fallback = cfg.CHECKPOINT_DIR / f"SRResNet_{suffix}.pt"
        if preferred.exists():
            ckpts[suffix] = preferred
        elif fallback.exists():
            ckpts[suffix] = fallback
        else:
            print(f"Warning: no SRResNet checkpoint found for loss '{suffix}'.")
    return ckpts


def load_srresnet_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu")


def build_generator_from_srresnet_checkpoint(path: str | Path) -> Generator:
    ckpt = load_srresnet_checkpoint(path)
    generator = Generator(
        large_kernel_size=cfg.LARGE_KERNEL_SIZE,
        small_kernel_size=cfg.SMALL_KERNEL_SIZE,
        n_channels=cfg.N_CHANNELS,
        n_blocks=cfg.N_BLOCKS,
        scaling_factor=cfg.SCALING_FACTOR,
    )
    generator.initialize_with_srresnet_state_dict(ckpt["model_state_dict"])
    return generator


def save_srgan_checkpoint(
    model_name: str,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    history: dict,
    base_srresnet_checkpoint: str,
) -> Path:
    path = cfg.CHECKPOINT_DIR / f"{model_name}.pt"
    payload = {
        "model_name": model_name,
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "history": history,
        "base_srresnet_checkpoint": base_srresnet_checkpoint,
        "config": {
            "scaling_factor": cfg.SCALING_FACTOR,
            "crop_size": cfg.CROP_SIZE,
            "large_kernel_size": cfg.LARGE_KERNEL_SIZE,
            "small_kernel_size": cfg.SMALL_KERNEL_SIZE,
            "n_channels": cfg.N_CHANNELS,
            "n_blocks": cfg.N_BLOCKS,
            "beta": cfg.SRGAN_BETA,
            "vgg19_i": cfg.VGG19_I,
            "vgg19_j": cfg.VGG19_J,
        },
    }
    torch.save(payload, path)
    return path


def run_srgan_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    truncated_vgg19: nn.Module,
    loader: DataLoader,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    beta: float | None = None,
    print_freq: int | None = None,
    epoch: int | None = None,
    split_name: str = "train",
):
    beta = cfg.SRGAN_BETA if beta is None else beta
    print_freq = cfg.SRGAN_PRINT_FREQ if print_freq is None else print_freq

    is_train = optimizer_g is not None and optimizer_d is not None
    generator.train() if is_train else generator.eval()
    discriminator.train() if is_train else discriminator.eval()

    content_criterion = nn.MSELoss().to(cfg.DEVICE)
    adv_criterion = nn.BCEWithLogitsLoss().to(cfg.DEVICE)

    content_meter = AverageMeter()
    adv_meter = AverageMeter()
    disc_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for batch_idx, (lr_imgs, hr_imgs_minus1_to_1) in enumerate(loader):
        lr_imgs = lr_imgs.to(cfg.DEVICE, non_blocking=True)
        hr_imgs_minus1_to_1 = hr_imgs_minus1_to_1.to(cfg.DEVICE, non_blocking=True)
        hr_imgs_imagenet = convert_image(hr_imgs_minus1_to_1, source="[-1, 1]", target="imagenet-norm")

        with torch.set_grad_enabled(is_train):
            sr_imgs_minus1_to_1 = generator(lr_imgs)
            sr_imgs_imagenet = convert_image(sr_imgs_minus1_to_1, source="[-1, 1]", target="imagenet-norm")

            sr_features = truncated_vgg19(sr_imgs_imagenet)
            hr_features = truncated_vgg19(hr_imgs_imagenet).detach()

            sr_logits_for_g = discriminator(sr_imgs_imagenet)
            content_loss = content_criterion(sr_features, hr_features)
            adversarial_loss_g = adv_criterion(sr_logits_for_g, torch.ones_like(sr_logits_for_g))
            perceptual_loss = content_loss + beta * adversarial_loss_g

            if is_train:
                optimizer_g.zero_grad(set_to_none=True)
                perceptual_loss.backward()
                if cfg.SRGAN_GRAD_CLIP is not None:
                    clip_gradient(optimizer_g, cfg.SRGAN_GRAD_CLIP)
                optimizer_g.step()

                sr_imgs_minus1_to_1 = generator(lr_imgs)
                sr_imgs_imagenet = convert_image(sr_imgs_minus1_to_1, source="[-1, 1]", target="imagenet-norm")

                hr_logits = discriminator(hr_imgs_imagenet)
                sr_logits_for_d = discriminator(sr_imgs_imagenet.detach())

                adversarial_loss_d = (
                    adv_criterion(sr_logits_for_d, torch.zeros_like(sr_logits_for_d))
                    + adv_criterion(hr_logits, torch.ones_like(hr_logits))
                )

                optimizer_d.zero_grad(set_to_none=True)
                adversarial_loss_d.backward()
                if cfg.SRGAN_GRAD_CLIP is not None:
                    clip_gradient(optimizer_d, cfg.SRGAN_GRAD_CLIP)
                optimizer_d.step()
            else:
                hr_logits = discriminator(hr_imgs_imagenet)
                sr_logits_for_d = discriminator(sr_imgs_imagenet.detach())
                adversarial_loss_d = (
                    adv_criterion(sr_logits_for_d, torch.zeros_like(sr_logits_for_d))
                    + adv_criterion(hr_logits, torch.ones_like(hr_logits))
                )

        batch_size = lr_imgs.size(0)
        psnr, ssim = compute_psnr_ssim_from_tensors(sr_imgs_minus1_to_1, hr_imgs_minus1_to_1, source="[-1, 1]")
        content_meter.update(content_loss.item(), batch_size)
        adv_meter.update(adversarial_loss_g.item(), batch_size)
        disc_meter.update(adversarial_loss_d.item(), batch_size)
        psnr_meter.update(psnr, batch_size)
        ssim_meter.update(ssim, batch_size)

        if is_train and batch_idx % print_freq == 0:
            ep = "?" if epoch is None else epoch
            print(
                f"[SRGAN][{split_name}] epoch={ep} batch={batch_idx}/{len(loader)} "
                f"content={content_meter.avg:.4f} adv_g={adv_meter.avg:.4f} "
                f"disc={disc_meter.avg:.4f} psnr={psnr_meter.avg:.3f} ssim={ssim_meter.avg:.4f}"
            )

    return {
        "content_loss": content_meter.avg,
        "adv_loss_g": adv_meter.avg,
        "disc_loss": disc_meter.avg,
        "psnr": psnr_meter.avg,
        "ssim": ssim_meter.avg,
    }


def train_single_srgan_from_srresnet(
    loss_name: str,
    srresnet_checkpoint_path: str | Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int | None = None,
    lr: float | None = None,
) -> dict:
    num_epochs = cfg.SRGAN_EPOCHS if num_epochs is None else num_epochs
    lr = cfg.SRGAN_LR if lr is None else lr

    suffix = loss_name.upper()
    model_name = f"SRGAN_ResNet_{suffix}"
    print(f"\n===== Training {model_name} from {Path(srresnet_checkpoint_path).name} =====")

    generator = build_generator_from_srresnet_checkpoint(srresnet_checkpoint_path).to(cfg.DEVICE)
    discriminator = Discriminator().to(cfg.DEVICE)
    truncated_vgg19 = TruncatedVGG19(i=cfg.VGG19_I, j=cfg.VGG19_J).to(cfg.DEVICE).eval()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    history = {
        "train_content_loss": [],
        "train_adv_loss_g": [],
        "train_disc_loss": [],
        "train_psnr": [],
        "train_ssim": [],
        "val_content_loss": [],
        "val_adv_loss_g": [],
        "val_disc_loss": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    best_val_content = float("inf")
    best_ckpt_path = None

    for epoch in range(num_epochs):
        train_metrics = run_srgan_epoch(
            generator=generator,
            discriminator=discriminator,
            truncated_vgg19=truncated_vgg19,
            loader=train_loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            beta=cfg.SRGAN_BETA,
            print_freq=cfg.SRGAN_PRINT_FREQ,
            epoch=epoch,
            split_name="train",
        )
        val_metrics = run_srgan_epoch(
            generator=generator,
            discriminator=discriminator,
            truncated_vgg19=truncated_vgg19,
            loader=val_loader,
            optimizer_g=None,
            optimizer_d=None,
            beta=cfg.SRGAN_BETA,
            split_name="val",
        )

        history["train_content_loss"].append(train_metrics["content_loss"])
        history["train_adv_loss_g"].append(train_metrics["adv_loss_g"])
        history["train_disc_loss"].append(train_metrics["disc_loss"])
        history["train_psnr"].append(train_metrics["psnr"])
        history["train_ssim"].append(train_metrics["ssim"])
        history["val_content_loss"].append(val_metrics["content_loss"])
        history["val_adv_loss_g"].append(val_metrics["adv_loss_g"])
        history["val_disc_loss"].append(val_metrics["disc_loss"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["val_ssim"].append(val_metrics["ssim"])

        print(
            f"[{model_name}] epoch={epoch} train_content={train_metrics['content_loss']:.4f} "
            f"val_content={val_metrics['content_loss']:.4f} val_psnr={val_metrics['psnr']:.3f} "
            f"val_ssim={val_metrics['ssim']:.4f}"
        )

        ckpt_path = save_srgan_checkpoint(
            model_name=model_name,
            epoch=epoch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            history=history,
            base_srresnet_checkpoint=str(srresnet_checkpoint_path),
        )

        if val_metrics["content_loss"] < best_val_content:
            best_val_content = val_metrics["content_loss"]
            best_ckpt_path = cfg.CHECKPOINT_DIR / f"{model_name}_best.pt"
            torch.save(torch.load(ckpt_path, map_location="cpu"), best_ckpt_path)

    history_path = cfg.METRICS_DIR / f"{model_name}_history.json"
    save_json(history, history_path)

    return {
        "model_name": model_name,
        "loss_name": suffix,
        "checkpoint_path": str(cfg.CHECKPOINT_DIR / f"{model_name}.pt"),
        "best_checkpoint_path": str(best_ckpt_path),
        "history_path": str(history_path),
        "history": history,
        "base_srresnet_checkpoint": str(srresnet_checkpoint_path),
    }


def train_multiple_srgan(
    loss_names: list[str] | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int | None = None,
    lr: float | None = None,
) -> dict[str, dict]:
    srresnet_ckpts = discover_srresnet_checkpoints(loss_names=loss_names)
    if len(srresnet_ckpts) == 0:
        raise FileNotFoundError(
            "No SRResNet checkpoints found. Train SRResNet first or place checkpoints in CHECKPOINT_DIR."
        )

    results = {}
    for suffix, ckpt_path in srresnet_ckpts.items():
        result = train_single_srgan_from_srresnet(
            loss_name=suffix,
            srresnet_checkpoint_path=ckpt_path,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
        )
        results[suffix] = result
    return results
