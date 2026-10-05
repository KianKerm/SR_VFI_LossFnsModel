from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

# ---- Reproducibility ----
SEED = 42

# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data / split parameters ----
SCALING_FACTOR = 4
CROP_SIZE = 96
MIN_SIZE = 100
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# ---- Dataloader parameters ----
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 4
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

# ---- Training parameters: SRResNet ----
SRRESNET_LOSS_FUNCTIONS = [
    "MSE",
    "L1",
    "GLOSS",
    "SSIM-Loss",
    "MS-SSIM",
    "MS-SSIM + L1",
    "Huber",
]
HUBER_DELTA = 1.35
SRRESNET_EPOCHS = 3
SRRESNET_LR = 1e-4
SRRESNET_PRINT_FREQ = 50
SRRESNET_GRAD_CLIP = None

# ---- Training parameters: SRGAN ----
SRGAN_EPOCHS = 3
SRGAN_LR = 1e-4
SRGAN_PRINT_FREQ = 50
SRGAN_GRAD_CLIP = None
SRGAN_BETA = 1e-3
VGG19_I = 5
VGG19_J = 4

# ---- Model architecture parameters ----
LARGE_KERNEL_SIZE = 9
SMALL_KERNEL_SIZE = 3
N_CHANNELS = 64
N_BLOCKS = 16

# ---- Output paths ----
PROJECT_ROOT = Path("./sr_project_outputs")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
METRICS_DIR = PROJECT_ROOT / "metrics"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_output_dirs() -> None:
    for path in [PROJECT_ROOT, CHECKPOINT_DIR, METRICS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int | None = None) -> None:
    seed = SEED if seed is None else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ensure_output_dirs()
set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
