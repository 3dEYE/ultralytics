# Ultralytics ?? AGPL-3.0 License - https://ultralytics.com/license
"""DINOv3 ConvNeXt backbone adapter for Ultralytics YOLO.

Uses a vendored copy of the ConvNeXt implementation (see `_dinov3_convnext_impl.py`)
so it runs on production machines without the upstream DINOv3 source tree.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ._dinov3_convnext_impl import ConvNeXt, convnext_sizes

__all__ = ("DINOv3ConvNeXt",)


_DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"

# Hashes come from dinov3/hub/backbones.py (facebookresearch/dinov3).
_WEIGHT_HASHES = {
    "tiny":  "21b726bb",
    "small": "296db49d",
    "base":  "801f2ba9",
    "large": "61fa432d",
}
_KNOWN_TAGS = {"LVD1689M", "SAT493M"}


def _official_weights_url(variant: str, tag: str) -> str:
    arch = f"dinov3_convnext_{variant}"
    filename = f"{arch}_pretrain_{tag.lower()}-{_WEIGHT_HASHES[variant]}.pth"
    return f"{_DINOV3_BASE_URL}/{arch}/{filename}"


def _load_state_dict(path_or_url: str) -> dict:
    """Load a state_dict from HTTP(S) URL or local filesystem path."""
    if path_or_url.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu")
    return torch.load(path_or_url, map_location="cpu")


def _build_convnext(variant: str, pretrained: bool, weights: Optional[str]) -> ConvNeXt:
    cfg = convnext_sizes[variant]
    model = ConvNeXt(in_chans=3, depths=cfg["depths"], dims=cfg["dims"])
    if not pretrained:
        model.init_weights()
        return model
    if weights in _KNOWN_TAGS:
        url = _official_weights_url(variant, weights)
    elif weights is None:
        url = _official_weights_url(variant, "LVD1689M")
    else:
        url = weights
    state_dict = _load_state_dict(url)
    model.load_state_dict(state_dict, strict=True)
    return model


class DINOv3ConvNeXt(nn.Module):
    """Wraps a DINOv3 ConvNeXt and yields a tuple of feature maps.

    YAML args: [variant, pretrained, freeze, weights, imagenet_norm, out_indices]
      variant     : "tiny" | "small" | "base" | "large"
      pretrained  : bool
      freeze      : bool - also sets a marker the Ultralytics trainer respects
      weights     : "LVD1689M" | "SAT493M" | local path | URL
      imagenet_norm : bool
      out_indices : tuple/list of ConvNeXt stage indices to expose.
                    Stage 0 -> P2/4, 1 -> P3/8, 2 -> P4/16, 3 -> P5/32.
                    Default (1, 2, 3) keeps backwards compatibility.
    """

    # Class-level defaults so checkpoints pickled before these attributes
    # existed still load and run correctly.
    out_indices = (1, 2, 3)
    imagenet_norm = True

    def __init__(
        self,
        variant: str = "small",
        pretrained: bool = False,
        freeze: bool = True,
        weights: Optional[str] = "LVD1689M",
        imagenet_norm: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        assert variant in convnext_sizes, f"unknown variant: {variant}"
        self.variant = variant
        self.imagenet_norm = imagenet_norm
        self.out_indices = tuple(out_indices)
        assert all(0 <= i < 4 for i in self.out_indices), f"out_indices must be in [0, 3], got {self.out_indices}"

        model = _build_convnext(variant, pretrained=pretrained, weights=weights)
        self.downsample_layers = model.downsample_layers
        self.stages = model.stages

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self._ultralytics_keep_frozen = True

        dims = convnext_sizes[variant]["dims"]
        self._out_channels: List[int] = [dims[i] for i in self.out_indices]

        if imagenet_norm:
            self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor):
        if self.imagenet_norm:
            x = (x - self._mean) / self._std
        feats = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                feats.append(x)
        return tuple(feats)
