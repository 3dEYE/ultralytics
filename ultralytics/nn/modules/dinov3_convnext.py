# Ultralytics ?? AGPL-3.0 License - https://ultralytics.com/license
"""DINOv3 ConvNeXt backbone adapter for Ultralytics YOLO.

Uses a vendored copy of the ConvNeXt implementation (see `_dinov3_convnext_impl.py`)
so it runs on production machines without the upstream DINOv3 source tree.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from ._dinov3_convnext_impl import ConvNeXt

__all__ = ("DINOv3ConvNeXt",)


def _infer_arch_from_state_dict(state_dict: dict) -> Tuple[List[int], List[int]]:
    """Recover ``(depths, dims)`` of a 4-stage ConvNeXt from an official DINOv3 checkpoint.

    ``dims`` come from the stem / downsample conv shapes, ``depths`` from the
    highest block index present per stage. Avoids hard-coding a variant table.
    """
    # dims[0] = stem out_channels: downsample_layers.0.0.weight has shape (dims[0], in, 4, 4)
    dims = [int(state_dict["downsample_layers.0.0.weight"].shape[0])]
    for i in range(1, 4):
        # downsample_layers.{i}.1.weight has shape (dims[i], dims[i-1], 2, 2)
        dims.append(int(state_dict[f"downsample_layers.{i}.1.weight"].shape[0]))

    depths = [0, 0, 0, 0]
    for k in state_dict:
        if k.startswith("stages.") and k.endswith(".dwconv.weight"):
            _, stage_idx, block_idx, _, _ = k.split(".")
            si, bi = int(stage_idx), int(block_idx)
            if depths[si] < bi + 1:
                depths[si] = bi + 1
    if any(d == 0 for d in depths):
        raise ValueError(f"Could not infer ConvNeXt depths from checkpoint, got {depths}")
    return depths, dims


def _build_convnext(weights: str) -> ConvNeXt:
    """Build a ConvNeXt whose architecture matches the given local checkpoint and load it."""
    state_dict = torch.load(weights, map_location="cpu")
    depths, dims = _infer_arch_from_state_dict(state_dict)
    model = ConvNeXt(in_chans=3, depths=depths, dims=dims)
    model.load_state_dict(state_dict, strict=True)
    return model


class DINOv3ConvNeXt(nn.Module):
    """Wraps a DINOv3 ConvNeXt and yields a tuple of feature maps.

    YAML args: [weights, freeze, imagenet_norm, out_indices]
      weights     : local filesystem path to an official DINOv3 ConvNeXt
                    ``*.pth`` checkpoint (e.g.
                    ``dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth``).
                    The variant (tiny/small/base/large) is inferred from the
                    checkpoint -- no architecture argument is needed.
      freeze      : bool - also sets a marker the Ultralytics trainer respects
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
        weights: str,
        freeze: bool = True,
        imagenet_norm: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        self.imagenet_norm = imagenet_norm
        self.out_indices = tuple(out_indices)
        assert all(0 <= i < 4 for i in self.out_indices), f"out_indices must be in [0, 3], got {self.out_indices}"

        model = _build_convnext(weights)
        self.downsample_layers = model.downsample_layers
        self.stages = model.stages

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self._ultralytics_keep_frozen = True

        # Architecture-defining channel sizes recovered from the checkpoint.
        dims = list(model.embed_dims)
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
