# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Vendored ConvNeXtV2 building blocks.

Adapted from the official Meta implementation
(https://github.com/facebookresearch/ConvNeXt-V2, MIT-licensed) so the
backbone can be used inside Ultralytics without pulling ``timm`` or
``MinkowskiEngine`` as runtime dependencies.

Only the dense (non-sparse) building blocks needed by ``ConvNeXtV2Backbone``
are kept here. ``DropPath`` and ``trunc_normal_`` are reimplemented inline
to avoid the ``timm`` dependency.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("LayerNorm2d", "GRN", "ConvNeXtV2Block", "ConvNeXtV2Stages", "CONVNEXTV2_VARIANTS")


# ----------------------------------------------------------------------------
# Lightweight reimplementations of timm helpers
# ----------------------------------------------------------------------------
def _trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Fill ``tensor`` with a truncated normal distribution (|x| <= 2*std)."""
    with torch.no_grad():
        tensor.normal_(mean, std)
        # Resample values that fall outside [-2*std, 2*std] until they are inside.
        for _ in range(10):
            mask = (tensor - mean).abs() > 2 * std
            if not bool(mask.any()):
                break
            tensor[mask] = torch.empty_like(tensor[mask]).normal_(mean, std)
        tensor.clamp_(min=mean - 2 * std, max=mean + 2 * std)
    return tensor


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    if keep > 0.0:
        mask.div_(keep)
    return x * mask


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


# ----------------------------------------------------------------------------
# Norm / GRN
# ----------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """LayerNorm supporting both ``channels_last`` (NHWC) and ``channels_first`` (NCHW).

    ``_affine_active`` lets ``ConvNeXtV2Backbone.fuse()`` neutralize the affine pair
    (weights are folded into the next Conv/Linear) while keeping the parameters in
    ``state_dict``. When ``False``, ``F.layer_norm`` is called with ``weight=None,
    bias=None`` so the ONNX exporter emits a bare LayerNormalization node that
    TensorRT can fuse without trailing Mul/Add.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(f"data_format must be channels_last or channels_first, got {data_format!r}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
        self._affine_active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self._affine_active else None
        bias = self.bias if self._affine_active else None
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        # channels_first: route through F.layer_norm on the last dim so the ONNX
        # exporter emits a single LayerNormalization node (opset>=17), which
        # TensorRT fuses natively. Mathematically identical to the manual form.
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        return x.permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXtV2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equivalent to torch.norm(x, p=2, dim=(1, 2), keepdim=True) but exports
        # to a small, TRT-friendly subgraph (Mul -> ReduceSum -> Sqrt) instead of
        # the deprecated `aten::norm` op.
        Gx = (x * x).sum(dim=(1, 2), keepdim=True).sqrt()
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ----------------------------------------------------------------------------
# Block
# ----------------------------------------------------------------------------
class ConvNeXtV2Block(nn.Module):
    """ConvNeXtV2 residual block (dwconv 7x7 -> LN -> MLP with GRN)."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)  # channels_last (NHWC) variant
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return identity + self.drop_path(x)


# ----------------------------------------------------------------------------
# Stages container (matches official keying: downsample_layers.* / stages.*)
# ----------------------------------------------------------------------------
class ConvNeXtV2Stages(nn.Module):
    """Container that holds ``downsample_layers`` + ``stages`` exactly as the official
    ``ConvNeXtV2`` model does, so official Meta checkpoints can be loaded directly
    (after dropping the classifier ``norm`` / ``head`` keys).
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must be 4-tuples (4 stages, strides 4/8/16/32).")
        self.depths: List[int] = list(depths)
        self.dims: List[int] = list(dims)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            self.stages.append(
                nn.Sequential(*[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            )
            cur += depths[i]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run all 4 stages, returning the post-stage feature maps (strides 4, 8, 16, 32)."""
        feats: List[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feats.append(x)
        return feats


# ----------------------------------------------------------------------------
# Variant table (depths, dims) - mirrors the official factory functions.
# ----------------------------------------------------------------------------
CONVNEXTV2_VARIANTS = {
    "atto":  {"depths": (2, 2, 6, 2),  "dims": (40, 80, 160, 320)},
    "femto": {"depths": (2, 2, 6, 2),  "dims": (48, 96, 192, 384)},
    "pico":  {"depths": (2, 2, 6, 2),  "dims": (64, 128, 256, 512)},
    "nano":  {"depths": (2, 2, 8, 2),  "dims": (80, 160, 320, 640)},
    "tiny":  {"depths": (3, 3, 9, 3),  "dims": (96, 192, 384, 768)},
    "base":  {"depths": (3, 3, 27, 3), "dims": (128, 256, 512, 1024)},
    "large": {"depths": (3, 3, 27, 3), "dims": (192, 384, 768, 1536)},
    "huge":  {"depths": (3, 3, 27, 3), "dims": (352, 704, 1408, 2816)},
}
