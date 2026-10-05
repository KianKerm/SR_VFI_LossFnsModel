from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# 8 fixed 3x3 kernels from Eq. (5) of the G-Loss paper.
_GRADIENT_KERNELS = torch.tensor(
    [
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    ],
    dtype=torch.float32,
).unsqueeze(1)  # [8, 1, 3, 3]

_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_DEFAULT_MS_LEVELS = 5
_DEFAULT_MIX_ALPHA = 0.84
_DEFAULT_DATA_RANGE = 2.0  # repo tensors are in [-1, 1]


# -------------------------------
# Shared helpers
# -------------------------------
def _legacy_get_reduction(
    size_average: bool | None,
    reduce: bool | None,
    reduction: str,
) -> str:
    if size_average is None and reduce is None:
        return reduction
    if reduce is False:
        return "none"
    if size_average is False:
        return "sum"
    return "mean"


def _validate_reduction(reduction: str) -> str:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")
    return reduction


def _reduce_batch_values(values: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")


def _as_4d(x: Tensor) -> tuple[Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(x.shape)}.")


def _normalize_loss_name(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


# -------------------------------
# G-Loss helpers
# -------------------------------
def _paper_extra_scales(scaling_factor: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """
    Recommended SG_n combinations reported in the paper.

    scale 2 -> G1 + SG2/4
    scale 3 -> G1 + SG2/4 + SG3/9
    scale 4 -> G1 + SG1? no, paper uses G1 + SG2/4 + SG4/16
    """
    if scaling_factor == 2:
        return (2,), (1.0 / 4.0,)
    if scaling_factor == 3:
        return (2, 3), (1.0 / 4.0, 1.0 / 9.0)
    if scaling_factor == 4:
        return (2, 4), (1.0 / 4.0, 1.0 / 16.0)
    raise ValueError(
        "Paper-recommended G-Loss presets are defined for scaling_factor in {2, 3, 4}. "
        "For other factors, pass extra_scales and scale_weights explicitly."
    )


def _split_downsample(x: Tensor, factor: int) -> Tensor:
    h, w = x.shape[-2:]
    if h % factor != 0 or w % factor != 0:
        raise ValueError(
            f"For splitting, spatial size {(h, w)} must be divisible by factor={factor}."
        )
    return F.pixel_unshuffle(x, downscale_factor=factor)


def _downsample(x: Tensor, factor: int, mode: str) -> Tensor:
    if factor < 1:
        raise ValueError("Downsampling factors must be >= 1.")
    if factor == 1:
        return x

    if mode == "avg":
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)
    if mode == "max":
        return F.max_pool2d(x, kernel_size=factor, stride=factor)
    if mode == "split":
        return _split_downsample(x, factor)
    raise ValueError("downsample_mode must be one of {'avg', 'max', 'split'}.")


def _gradient_valid_mask(h: int, w: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    masks = []
    for dx, dy in _OFFSETS:
        mask = torch.zeros((h, w), device=device, dtype=dtype)
        row_start = max(0, -dx)
        row_end = h - max(0, dx)
        col_start = max(0, -dy)
        col_end = w - max(0, dy)
        mask[row_start:row_end, col_start:col_end] = 1.0
        masks.append(mask)
    return torch.stack(masks, dim=0).unsqueeze(0).unsqueeze(0)


def _gradient_feature_maps(x: Tensor, kernels: Tensor) -> tuple[Tensor, Tensor]:
    n, c, h, w = x.shape
    weight = kernels.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    feats = F.conv2d(x, weight, bias=None, stride=1, padding=1, groups=c)
    feats = feats.view(n, c, 8, h, w)

    valid_mask = _gradient_valid_mask(h, w, device=x.device, dtype=x.dtype)
    feats = feats * valid_mask
    return feats, valid_mask


def _masked_l1_loss(diff: Tensor, mask: Tensor, reduction: str) -> Tensor:
    weighted = diff * mask
    values = weighted.flatten(1).sum(dim=1) / mask.expand(diff.size(0), diff.size(1), -1, -1, -1).flatten(1).sum(dim=1).clamp_min(1.0)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")


def _gradient_term(input: Tensor, target: Tensor, kernels: Tensor, reduction: str) -> Tensor:
    input_grad, mask = _gradient_feature_maps(input, kernels)
    target_grad, _ = _gradient_feature_maps(target, kernels)
    return _masked_l1_loss((input_grad - target_grad).abs(), mask, reduction)


# -------------------------------
# SSIM / MS-SSIM helpers
# -------------------------------
def _gaussian_1d(window_size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_2d(window_size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    g1 = _gaussian_1d(window_size, sigma, device=device, dtype=dtype)
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    return g2


def _default_window_size_for_sigma(sigma: float) -> int:
    # Covers ~ +/- 3 sigma and keeps the kernel odd.
    size = int(math.ceil(6.0 * float(sigma))) + 1
    if size % 2 == 0:
        size += 1
    return max(size, 3)


def _effective_window_size(requested_window_size: int, height: int, width: int) -> int:
    max_size = min(2 * height - 1, 2 * width - 1)
    if max_size < 3:
        raise ValueError(f"Input spatial size {(height, width)} is too small for SSIM/MS-SSIM.")
    size = min(int(requested_window_size), int(max_size))
    if size % 2 == 0:
        size -= 1
    return max(size, 3)


def _apply_gaussian_filter(x: Tensor, kernel_2d: Tensor) -> Tensor:
    c = x.size(1)
    kh, kw = kernel_2d.shape
    pad_h = kh // 2
    pad_w = kw // 2

    if x.size(-2) <= pad_h or x.size(-1) <= pad_w:
        raise ValueError(
            f"Input spatial size {tuple(x.shape[-2:])} is too small for gaussian padding {(pad_h, pad_w)}."
        )

    weight = kernel_2d.view(1, 1, kh, kw).to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    return F.conv2d(x, weight, groups=c)


def _ssim_components(
    input: Tensor,
    target: Tensor,
    *,
    data_range: float,
    window_size: int,
    sigma: float,
    k1: float,
    k2: float,
) -> tuple[Tensor, Tensor]:
    window_size = _effective_window_size(int(window_size), int(input.size(-2)), int(input.size(-1)))
    kernel = _gaussian_2d(window_size, sigma, device=input.device, dtype=input.dtype)

    mu_x = _apply_gaussian_filter(input, kernel)
    mu_y = _apply_gaussian_filter(target, kernel)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = _apply_gaussian_filter(input * input, kernel) - mu_x_sq
    sigma_y_sq = _apply_gaussian_filter(target * target, kernel) - mu_y_sq
    sigma_xy = _apply_gaussian_filter(input * target, kernel) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Numerical guard.
    sigma_x_sq = sigma_x_sq.clamp_min(0.0)
    sigma_y_sq = sigma_y_sq.clamp_min(0.0)

    luminance = (2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)
    cs = (2.0 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = luminance * cs
    return ssim_map, cs


def _ssim_per_image(
    input: Tensor,
    target: Tensor,
    *,
    data_range: float,
    window_size: int,
    sigma: float,
    k1: float,
    k2: float,
) -> tuple[Tensor, Tensor]:
    ssim_map, cs_map = _ssim_components(
        input,
        target,
        data_range=data_range,
        window_size=window_size,
        sigma=sigma,
        k1=k1,
        k2=k2,
    )
    ssim_vals = ssim_map.flatten(2).mean(dim=2).mean(dim=1)
    cs_vals = cs_map.flatten(2).mean(dim=2).mean(dim=1)
    return ssim_vals, cs_vals


# -------------------------------
# Functional losses
# -------------------------------
def g_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    scaling_factor: int = 4,
    downsample_mode: str = "split",
    extra_scales: Sequence[int] | None = None,
    scale_weights: Sequence[float] | None = None,
    include_pixel_loss: bool = True,
    include_g1: bool = True,
    pixel_weight: float = 1.0,
    g1_weight: float = 1.0,
    kernels: Tensor | None = None,
) -> Tensor:
    if kernels is None:
        kernels = _GRADIENT_KERNELS

    if target.size() != input.size():
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This may lead to unintended broadcasting. Please ensure they have the same size.",
            stacklevel=2,
        )

    reduction = _validate_reduction(_legacy_get_reduction(size_average, reduce, reduction))

    input, target = torch.broadcast_tensors(input, target)
    input, _ = _as_4d(input)
    target, _ = _as_4d(target)

    if extra_scales is None:
        extra_scales, default_scale_weights = _paper_extra_scales(int(scaling_factor))
    else:
        extra_scales = tuple(int(s) for s in extra_scales)
        default_scale_weights = tuple(1.0 / float(s * s) for s in extra_scales)

    if scale_weights is None:
        scale_weights = default_scale_weights
    else:
        scale_weights = tuple(float(w) for w in scale_weights)

    if len(extra_scales) != len(scale_weights):
        raise ValueError("extra_scales and scale_weights must have the same length.")

    total = input.new_zeros(()) if reduction != "none" else input.new_zeros((input.size(0),))

    if include_pixel_loss:
        pixel_loss = F.l1_loss(input, target, reduction="none").flatten(1).mean(dim=1)
        total = total + float(pixel_weight) * _reduce_batch_values(pixel_loss, reduction)

    if include_g1:
        total = total + float(g1_weight) * _gradient_term(input, target, kernels, reduction)

    for factor, weight in zip(extra_scales, scale_weights):
        ds_input = _downsample(input, factor, downsample_mode)
        ds_target = _downsample(target, factor, downsample_mode)
        total = total + float(weight) * _gradient_term(ds_input, ds_target, kernels, reduction)

    return total


def ssim_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    data_range: float = _DEFAULT_DATA_RANGE,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Tensor:
    if target.size() != input.size():
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This may lead to unintended broadcasting. Please ensure they have the same size.",
            stacklevel=2,
        )

    reduction = _validate_reduction(_legacy_get_reduction(size_average, reduce, reduction))
    input, target = torch.broadcast_tensors(input, target)
    input, _ = _as_4d(input)
    target, _ = _as_4d(target)

    ssim_vals, _ = _ssim_per_image(
        input,
        target,
        data_range=float(data_range),
        window_size=int(window_size),
        sigma=float(sigma),
        k1=float(k1),
        k2=float(k2),
    )
    losses = 1.0 - ssim_vals
    return _reduce_batch_values(losses, reduction)


def ms_ssim_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    data_range: float = _DEFAULT_DATA_RANGE,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    levels: int = _DEFAULT_MS_LEVELS,
) -> Tensor:
    if target.size() != input.size():
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This may lead to unintended broadcasting. Please ensure they have the same size.",
            stacklevel=2,
        )

    reduction = _validate_reduction(_legacy_get_reduction(size_average, reduce, reduction))
    input, target = torch.broadcast_tensors(input, target)
    input, _ = _as_4d(input)
    target, _ = _as_4d(target)

    if levels < 1:
        raise ValueError("levels must be >= 1")

    current_x = input
    current_y = target
    cs_terms: list[Tensor] = []
    last_ssim: Tensor | None = None

    for level in range(int(levels)):
        ssim_vals, cs_vals = _ssim_per_image(
            current_x,
            current_y,
            data_range=float(data_range),
            window_size=int(window_size),
            sigma=float(sigma),
            k1=float(k1),
            k2=float(k2),
        )
        last_ssim = ssim_vals

        if level < int(levels) - 1:
            cs_terms.append(cs_vals)
            if min(current_x.shape[-2:]) < 2 or min(current_y.shape[-2:]) < 2:
                break
            current_x = F.avg_pool2d(current_x, kernel_size=2, stride=2, ceil_mode=False)
            current_y = F.avg_pool2d(current_y, kernel_size=2, stride=2, ceil_mode=False)

    assert last_ssim is not None

    if cs_terms:
        ms_ssim_vals = torch.prod(torch.stack(cs_terms, dim=0), dim=0) * last_ssim
    else:
        ms_ssim_vals = last_ssim

    losses = 1.0 - ms_ssim_vals
    return _reduce_batch_values(losses, reduction)


def ms_ssim_l1_loss(
    input: Tensor,
    target: Tensor,
    size_average: bool | None = None,
    reduce: bool | None = None,
    reduction: str = "mean",
    data_range: float = _DEFAULT_DATA_RANGE,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    levels: int = _DEFAULT_MS_LEVELS,
    alpha: float = _DEFAULT_MIX_ALPHA,
    l1_sigma: float = 1.5,
    l1_window_size: int | None = None,
) -> Tensor:
    reduction = _validate_reduction(_legacy_get_reduction(size_average, reduce, reduction))
    input, target = torch.broadcast_tensors(input, target)
    input, _ = _as_4d(input)
    target, _ = _as_4d(target)

    ms_component = ms_ssim_loss(
        input,
        target,
        reduction="none",
        data_range=float(data_range),
        window_size=int(window_size),
        sigma=float(sigma),
        k1=float(k1),
        k2=float(k2),
        levels=int(levels),
    )

    if l1_window_size is None:
        l1_window_size = _default_window_size_for_sigma(float(l1_sigma))
    kernel = _gaussian_2d(int(l1_window_size), float(l1_sigma), device=input.device, dtype=input.dtype)
    l1_map = _apply_gaussian_filter((input - target).abs(), kernel)
    l1_component = l1_map.flatten(1).mean(dim=1) / float(data_range)

    mixed = float(alpha) * ms_component + (1.0 - float(alpha)) * l1_component
    return _reduce_batch_values(mixed, reduction)


# -------------------------------
# Module wrappers
# -------------------------------
class GLoss(nn.Module):
    """
    nn.Module wrapper for G-Loss.

    Defaults are chosen so that, with scaling_factor=4, this matches the paper's
    recommended split-based loss composition: PL + G1 + SG2/4 + SG4/16.
    """

    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        scaling_factor: int = 4,
        downsample_mode: str = "split",
        extra_scales: Sequence[int] | None = None,
        scale_weights: Sequence[float] | None = None,
        include_pixel_loss: bool = True,
        include_g1: bool = True,
        pixel_weight: float = 1.0,
        g1_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.scaling_factor = int(scaling_factor)
        self.downsample_mode = downsample_mode
        self.extra_scales = None if extra_scales is None else tuple(int(s) for s in extra_scales)
        self.scale_weights = None if scale_weights is None else tuple(float(w) for w in scale_weights)
        self.include_pixel_loss = bool(include_pixel_loss)
        self.include_g1 = bool(include_g1)
        self.pixel_weight = float(pixel_weight)
        self.g1_weight = float(g1_weight)

        self.register_buffer("gradient_kernels", _GRADIENT_KERNELS.clone())

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return g_loss(
            input=input,
            target=target,
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction,
            scaling_factor=self.scaling_factor,
            downsample_mode=self.downsample_mode,
            extra_scales=self.extra_scales,
            scale_weights=self.scale_weights,
            include_pixel_loss=self.include_pixel_loss,
            include_g1=self.include_g1,
            pixel_weight=self.pixel_weight,
            g1_weight=self.g1_weight,
            kernels=self.gradient_kernels,
        )

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction!r}, scaling_factor={self.scaling_factor}, "
            f"downsample_mode={self.downsample_mode!r}, extra_scales={self.extra_scales}, "
            f"scale_weights={self.scale_weights}, include_pixel_loss={self.include_pixel_loss}, "
            f"include_g1={self.include_g1}, pixel_weight={self.pixel_weight}, g1_weight={self.g1_weight}"
        )


class SSIMLoss(nn.Module):
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        data_range: float = _DEFAULT_DATA_RANGE,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return ssim_loss(
            input=input,
            target=target,
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction,
            data_range=self.data_range,
            window_size=self.window_size,
            sigma=self.sigma,
            k1=self.k1,
            k2=self.k2,
        )

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction!r}, data_range={self.data_range}, window_size={self.window_size}, "
            f"sigma={self.sigma}, k1={self.k1}, k2={self.k2}"
        )


class MSSSIMLoss(nn.Module):
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        data_range: float = _DEFAULT_DATA_RANGE,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        levels: int = _DEFAULT_MS_LEVELS,
    ) -> None:
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.levels = int(levels)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return ms_ssim_loss(
            input=input,
            target=target,
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction,
            data_range=self.data_range,
            window_size=self.window_size,
            sigma=self.sigma,
            k1=self.k1,
            k2=self.k2,
            levels=self.levels,
        )

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction!r}, data_range={self.data_range}, window_size={self.window_size}, "
            f"sigma={self.sigma}, k1={self.k1}, k2={self.k2}, levels={self.levels}"
        )


class MSSSIML1Loss(nn.Module):
    def __init__(
        self,
        size_average: bool | None = None,
        reduce: bool | None = None,
        reduction: str = "mean",
        data_range: float = _DEFAULT_DATA_RANGE,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        levels: int = _DEFAULT_MS_LEVELS,
        alpha: float = _DEFAULT_MIX_ALPHA,
        l1_sigma: float = 1.5,
        l1_window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.levels = int(levels)
        self.alpha = float(alpha)
        self.l1_sigma = float(l1_sigma)
        self.l1_window_size = None if l1_window_size is None else int(l1_window_size)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return ms_ssim_l1_loss(
            input=input,
            target=target,
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction,
            data_range=self.data_range,
            window_size=self.window_size,
            sigma=self.sigma,
            k1=self.k1,
            k2=self.k2,
            levels=self.levels,
            alpha=self.alpha,
            l1_sigma=self.l1_sigma,
            l1_window_size=self.l1_window_size,
        )

    def extra_repr(self) -> str:
        return (
            f"reduction={self.reduction!r}, data_range={self.data_range}, window_size={self.window_size}, "
            f"sigma={self.sigma}, k1={self.k1}, k2={self.k2}, levels={self.levels}, alpha={self.alpha}, "
            f"l1_sigma={self.l1_sigma}, l1_window_size={self.l1_window_size}"
        )


# -------------------------------
# Factory for train_srresnet.py
# -------------------------------
def build_loss_function(
    loss_name: str,
    *,
    scaling_factor: int = 4,
    data_range: float = _DEFAULT_DATA_RANGE,
    huber_delta: float = 1.0,
) -> nn.Module:
    name = _normalize_loss_name(loss_name)

    if name == "MSE":
        return nn.MSELoss()
    if name == "L1":
        return nn.L1Loss()
    if name in {"HUBER", "HUBERLOSS", "SMOOTHL1"}:
        return nn.HuberLoss(delta=float(huber_delta))

    if name in {"GLOSS"}:
        return GLoss(
            scaling_factor=scaling_factor,
            reduction="mean",
            downsample_mode="split",
        )

    if name in {"SSIM", "SSIMLOSS"}:
        return SSIMLoss(
            reduction="mean",
            data_range=data_range,
        )

    if name in {"MSSSIM", "MSSSIMLOSS"}:
        return MSSSIMLoss(
            reduction="mean",
            data_range=data_range,
        )

    if name in {"MSSSIML1", "MIX", "MSSSIMPLUSL1"}:
        return MSSSIML1Loss(
            reduction="mean",
            data_range=data_range,
        )

    raise ValueError(f"Unsupported loss function: {loss_name}")


__all__ = [
    "GLoss",
    "SSIMLoss",
    "MSSSIMLoss",
    "MSSSIML1Loss",
    "g_loss",
    "ssim_loss",
    "ms_ssim_loss",
    "ms_ssim_l1_loss",
    "build_loss_function",
]
