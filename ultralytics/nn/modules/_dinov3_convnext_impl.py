# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# Vendored from facebookresearch/dinov3 (dinov3/models/convnext.py) and trimmed
# for use as a YOLO feature extractor. The DINO classification / SSL plumbing
# was removed; only the ConvNeXt feature pyramid that the YOLO adapter
# (DINOv3ConvNeXt) consumes is exposed. Kept here to avoid depending on the
# dinov3 source tree in production.
#
# ---------------------------------------------------------------------------
# Differences vs. the upstream dinov3/models/convnext.py
# ---------------------------------------------------------------------------
# Removed (not on the YOLO forward path; would only add latency / dead code):
#   * ConvNeXt.forward, forward_features, forward_features_list:
#       upstream returns a dict with x_norm_clstoken / x_norm_patchtokens /
#       x_storage_tokens / x_prenorm built via global avg pool + flatten +
#       cat([CLS, patches]) + final LayerNorm. YOLO needs none of that --
#       the adapter consumes downsample_layers + stages directly and feeds
#       NCHW feature maps to the YOLO neck.
#   * ConvNeXt.get_intermediate_layers / _get_intermediate_layers:
#       upstream contains an F.interpolate(..., antialias=True) that resizes
#       feature maps to a ViT-style patch grid. For YOLO this destroys the
#       native /4,/8,/16,/32 strides and is also a slow op on TensorRT/ORT.
#   * Attributes head, n_blocks, n_storage_tokens, chunked_blocks,
#     embed_dim (single), input_pad_size, patch_size:
#       SSL/loader bookkeeping that no consumer in this repo reads.
#   * The `masks` argument and iBOT masking branch:
#       only meaningful during DINOv3 self-supervised pretraining.
#
# Kept verbatim (do NOT change -- required for strict=True checkpoint load
# of official DINOv3 weights such as
# dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth):
#   * Block, LayerNorm, DropPath, drop_path:
#       define the parameter layout encoded in the checkpoint state_dict.
#   * ConvNeXt.downsample_layers, ConvNeXt.stages:
#       the actual backbone weights.
#   * ConvNeXt.norm (nn.LayerNorm on dims[-1]) AND
#     ConvNeXt.norms = ModuleList([Id, Id, Id, self.norm]):
#       both must remain. PyTorch's state_dict() does NOT deduplicate the
#       aliased parameter -- the official checkpoint contains BOTH
#       `norm.{weight,bias}` and `norms.3.{weight,bias}` keys, so dropping
#       either attribute would break strict=True loading. They are
#       intentionally not invoked on the YOLO forward path (the adapter
#       takes only downsample_layers + stages).
#   * ConvNeXt.init_weights / _init_weights:
#       used by the adapter when pretrained=False is requested.
#   * convnext_sizes, get_convnext_arch:
#       the variant registry consumed by the adapter.
#
# Trainer integration note:
#   ultralytics/engine/trainer.py imports the custom `LayerNorm` class from
#   this file and adds it to its `bn` tuple so that its weights/biases land
#   in the no-decay parameter group (alongside torch.nn norm layers and the
#   ConvNeXt LayerScale `gamma`). Renaming or removing `LayerNorm` here
#   would silently move those params into the L2-decayed group.
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
from functools import partial
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn


logger = logging.getLogger("dinov3")


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init_value = layer_scale_init_value
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def init_weights(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt backbone trimmed for use as a YOLO feature extractor.

    Only ``downsample_layers`` and ``stages`` participate in forward. The final
    ``norm`` LayerNorm and the ``norms`` ModuleList are kept solely so that
    official DINOv3 checkpoints load with ``strict=True``; they are not invoked
    on the YOLO forward path (the :class:`DINOv3ConvNeXt` adapter consumes
    ``downsample_layers`` and ``stages`` directly).

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (List[int]): Blocks per stage. Default: [3, 3, 9, 3]
        dims (List[int]): Feature dimension per stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): LayerScale init value. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        self.downsample_layers = nn.ModuleList()  # stem + 3 intermediate downsamplers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Kept for checkpoint key compatibility (norm.weight, norm.bias). Not on forward path.
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.norms = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.norms.append(self.norm)

        self.embed_dims = dims  # per-stage dimensions, used by external feature consumers

    def init_weights(self):
        self.apply(self._init_weights)
        for stage_id, stage in enumerate(self.stages):
            for block_id, block in enumerate(stage):
                if block.gamma is not None:
                    nn.init.constant_(self.stages[stage_id][block_id].gamma, block.layer_scale_init_value)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )
