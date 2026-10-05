from __future__ import annotations

import random

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from .utils import convert_image


def train_val_test_split_indices(
    n: int,
    train_fraction: float | None = None,
    val_fraction: float | None = None,
    test_fraction: float | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    train_fraction = cfg.TRAIN_FRACTION if train_fraction is None else train_fraction
    val_fraction = cfg.VAL_FRACTION if val_fraction is None else val_fraction
    test_fraction = cfg.TEST_FRACTION if test_fraction is None else test_fraction
    seed = cfg.SEED if seed is None else seed

    assert abs(train_fraction + val_fraction + test_fraction - 1.0) < 1e-8

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


class InMemorySRDataset(Dataset):
    def __init__(
        self,
        image_array: np.ndarray,
        indices: np.ndarray,
        split: str,
        crop_size: int | None = None,
        scaling_factor: int | None = None,
        lr_img_type: str = "imagenet-norm",
        hr_img_type: str = "[-1, 1]",
        min_size: int | None = None,
    ):
        self.image_array = np.asarray(image_array)
        self.indices = np.asarray(indices)
        self.split = split.lower()
        self.crop_size = cfg.CROP_SIZE if crop_size is None else int(crop_size)
        self.scaling_factor = cfg.SCALING_FACTOR if scaling_factor is None else int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.min_size = cfg.MIN_SIZE if min_size is None else int(min_size)

        assert self.split in {"train", "val", "test"}
        assert lr_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}
        assert hr_img_type in {"[0, 255]", "[0, 1]", "[-1, 1]", "imagenet-norm"}

        if self.split == "train" and self.crop_size % self.scaling_factor != 0:
            raise ValueError("crop_size must be divisible by scaling_factor for training.")

        valid_indices = []
        for idx in self.indices:
            h, w = self.image_array[int(idx)].shape[:2]
            if min(h, w) >= self.min_size:
                valid_indices.append(int(idx))
        self.indices = np.asarray(valid_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_pil_image(self, idx: int) -> Image.Image:
        return Image.fromarray(self.image_array[idx].astype(np.uint8), mode="RGB")

    def _random_train_crop(self, img: Image.Image) -> Image.Image:
        if img.width < self.crop_size or img.height < self.crop_size:
            raise ValueError(
                f"Image is smaller than crop size. Image size={(img.width, img.height)}, crop_size={self.crop_size}"
            )
        left = random.randint(0, img.width - self.crop_size)
        top = random.randint(0, img.height - self.crop_size)
        return img.crop((left, top, left + self.crop_size, top + self.crop_size))

    def _center_divisible_crop(self, img: Image.Image) -> Image.Image:
        x_remainder = img.width % self.scaling_factor
        y_remainder = img.height % self.scaling_factor
        left = x_remainder // 2
        top = y_remainder // 2
        right = left + (img.width - x_remainder)
        bottom = top + (img.height - y_remainder)
        return img.crop((left, top, right, bottom))

    def __getitem__(self, i: int):
        img = self._get_pil_image(int(self.indices[i]))

        if self.split == "train":
            hr_img = self._random_train_crop(img)
        else:
            hr_img = self._center_divisible_crop(img)

        lr_img = hr_img.resize(
            (hr_img.width // self.scaling_factor, hr_img.height // self.scaling_factor),
            Image.BICUBIC,
        )

        assert hr_img.width == lr_img.width * self.scaling_factor
        assert hr_img.height == lr_img.height * self.scaling_factor

        lr_tensor = convert_image(lr_img, source="pil", target=self.lr_img_type)
        hr_tensor = convert_image(hr_img, source="pil", target=self.hr_img_type)

        return lr_tensor, hr_tensor


def build_dataloaders(
    image_array: np.ndarray,
    split_indices: dict[str, np.ndarray],
    crop_size: int | None = None,
    scaling_factor: int | None = None,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
):
    crop_size = cfg.CROP_SIZE if crop_size is None else crop_size
    scaling_factor = cfg.SCALING_FACTOR if scaling_factor is None else scaling_factor
    train_batch_size = cfg.TRAIN_BATCH_SIZE if train_batch_size is None else train_batch_size
    eval_batch_size = cfg.EVAL_BATCH_SIZE if eval_batch_size is None else eval_batch_size
    num_workers = cfg.NUM_WORKERS if num_workers is None else num_workers
    pin_memory = cfg.PIN_MEMORY if pin_memory is None else pin_memory

    train_dataset = InMemorySRDataset(
        image_array=image_array,
        indices=split_indices["train"],
        split="train",
        crop_size=crop_size,
        scaling_factor=scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )
    val_dataset = InMemorySRDataset(
        image_array=image_array,
        indices=split_indices["val"],
        split="val",
        crop_size=crop_size,
        scaling_factor=scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )
    test_dataset = InMemorySRDataset(
        image_array=image_array,
        indices=split_indices["test"],
        split="test",
        crop_size=crop_size,
        scaling_factor=scaling_factor,
        lr_img_type="imagenet-norm",
        hr_img_type="[-1, 1]",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
