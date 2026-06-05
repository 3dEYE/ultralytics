# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DINOv3 ViT + Spatial Tuning Adapter backbone for Ultralytics YOLO.

This module turns single-scale DINOv3 ViT features (stride 16) into a
multi-scale pyramid (P3/8, P4/16, P5/32) suitable for detection heads.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("DINOv3ViTSTA",)


class DINOv3ViTSTA(nn.Module):
    """DINOv3 ViT backbone with a lightweight Spatial Tuning Adapter (STA).

    YAML args:
      arch            : "vits16" | "vitb16" | "vitl16" | "vit7b16"
      pretrained      : bool
      freeze          : bool
      weights         : None | local path | URL | Weights enum string (handled by dinov3 hub)
      out_channels    : [P3, P4, P5] output channels (default: [192, 384, 768])
      detail_channels : channels for detail branch (default: 96)
      imagenet_norm   : apply ImageNet normalization before ViT + detail path
    """

    _BUILDERS: Dict[str, Tuple[Callable[..., nn.Module], int]] = {}

    def __init__(
        self,
        arch: str = "vitb16",
        pretrained: bool = True,
        freeze: bool = True,
        weights: Optional[str] = None,
        interaction_indexes: Optional[Sequence[int]] = None,
        out_channels: Sequence[int] = (192, 384, 768),
        detail_channels: int = 96,
        imagenet_norm: bool = True,
    ) -> None:
        super().__init__()

        if not DINOv3ViTSTA._BUILDERS:
            from dinov3.hub.backbones import dinov3_vit7b16, dinov3_vitb16, dinov3_vitl16, dinov3_vits16

            DINOv3ViTSTA._BUILDERS = {
                "vits16": (dinov3_vits16, 384),
                "vitb16": (dinov3_vitb16, 768),
                "vitl16": (dinov3_vitl16, 1024),
                "vit7b16": (dinov3_vit7b16, 4096),
            }

        if arch not in self._BUILDERS:
            raise ValueError(f"Unknown DINOv3 ViT arch '{arch}'. Known: {sorted(self._BUILDERS)}")

        if len(out_channels) != 3:
            raise ValueError(f"out_channels must contain 3 values [P3, P4, P5], got: {out_channels}")

        self.arch = arch
        self.freeze = bool(freeze)
        self.imagenet_norm = bool(imagenet_norm)

        builder, embed_dim = self._BUILDERS[arch]
        builder_kwargs = dict(pretrained=bool(pretrained))
        if weights is not None:
            builder_kwargs["weights"] = weights
        self.vit = builder(**builder_kwargs)
        self.interaction_indexes = list(interaction_indexes) if interaction_indexes is not None else None

        self.p3_channels = int(out_channels[0])
        self.p4_channels = int(out_channels[1])
        self.p5_channels = int(out_channels[2])
        self._out_channels: List[int] = [self.p3_channels, self.p4_channels, self.p5_channels]

        self._mean: torch.Tensor
        self._std: torch.Tensor
        if self.imagenet_norm:
            self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        dch = int(detail_channels)
        # Detail path (DEIMv2-style SPM-lite): 1/8, 1/16, 1/32 maps.
        self.detail_stem = nn.Sequential(
            nn.Conv2d(3, dch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.detail_conv2 = nn.Sequential(
            nn.Conv2d(dch, dch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dch * 2),
        )
        self.detail_conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(dch * 2, dch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dch * 4),
        )
        self.detail_conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(dch * 4, dch * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dch * 4),
        )

        # Per-scale projection after semantic+detail concatenation.
        self.fuse_p3_conv = nn.Conv2d(embed_dim + dch * 2, self.p3_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fuse_p4_conv = nn.Conv2d(embed_dim + dch * 4, self.p4_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fuse_p5_conv = nn.Conv2d(embed_dim + dch * 4, self.p5_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fuse_p3_bn = nn.BatchNorm2d(self.p3_channels)
        self.fuse_p4_bn = nn.BatchNorm2d(self.p4_channels)
        self.fuse_p5_bn = nn.BatchNorm2d(self.p5_channels)

        if self.freeze:
            self.set_frozen(True)

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def _pick_vit_layers(self) -> List[int]:
        if self.interaction_indexes is not None:
            if len(self.interaction_indexes) == 0:
                raise ValueError("interaction_indexes must not be empty")
            return [int(i) for i in self.interaction_indexes]
        n_blocks = int(getattr(self.vit, "n_blocks", len(getattr(self.vit, "blocks", []))))
        if n_blocks <= 0:
            raise RuntimeError("DINOv3 ViT model has no transformer blocks")
        base = max(0, n_blocks - 3)
        return [base, min(base + 1, n_blocks - 1), n_blocks - 1]

    def _forward_vit_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        layer_ids = self._pick_vit_layers()
        outs = self.vit.get_intermediate_layers(x, n=layer_ids, reshape=True, return_class_token=False)
        if len(outs) == 0:
            raise RuntimeError("Expected at least one intermediate ViT map")
        if len(outs) == 1:
            outs = (outs[0], outs[0], outs[0])
        elif len(outs) >= 3:
            outs = (outs[0], outs[1], outs[2])
        else:
            outs = (outs[0], outs[0], outs[-1])
        return outs

    @staticmethod
    def _resize_semantic_for_scales(sem_feats: Tuple[torch.Tensor, ...], h16: int, w16: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # DEIMv2 pattern: first selected layer -> P3(1/8), second -> P4(1/16), third -> P5(1/32).
        p3_h, p3_w = h16 * 2, w16 * 2
        p4_h, p4_w = h16, w16
        p5_h, p5_w = max(1, h16 // 2), max(1, w16 // 2)
        p3_sem = torch.nn.functional.interpolate(sem_feats[0], size=(p3_h, p3_w), mode="bilinear", align_corners=False)
        p4_sem = torch.nn.functional.interpolate(sem_feats[1], size=(p4_h, p4_w), mode="bilinear", align_corners=False)
        p5_sem = torch.nn.functional.interpolate(sem_feats[2], size=(p5_h, p5_w), mode="bilinear", align_corners=False)
        return p3_sem, p4_sem, p5_sem

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.imagenet_norm:
            x_in = (x - self._mean) / self._std
        else:
            x_in = x

        h16 = max(1, x_in.shape[2] // 16)
        w16 = max(1, x_in.shape[3] // 16)

        # Fine-grained detail path.
        d = self.detail_stem(x_in)
        p3_detail = self.detail_conv2(d)
        p4_detail = self.detail_conv3(p3_detail)
        p5_detail = self.detail_conv4(p4_detail)

        # Strong semantic path from selected ViT blocks.
        grad_enabled = not self.freeze and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad_enabled):
            sem_feats = self._forward_vit_maps(x_in)

        p3_sem, p4_sem, p5_sem = self._resize_semantic_for_scales(sem_feats, h16, w16)

        p3 = self.fuse_p3_bn(self.fuse_p3_conv(torch.cat((p3_sem, p3_detail), dim=1)))
        p4 = self.fuse_p4_bn(self.fuse_p4_conv(torch.cat((p4_sem, p4_detail), dim=1)))
        p5 = self.fuse_p5_bn(self.fuse_p5_conv(torch.cat((p5_sem, p5_detail), dim=1)))

        return [p3, p4, p5]

    def set_frozen(self, frozen: bool = True) -> "DINOv3ViTSTA":
        self.freeze = bool(frozen)
        for p in self.vit.parameters():
            p.requires_grad_(not self.freeze)
        if self.freeze:
            self.vit._ultralytics_keep_frozen = True
        elif hasattr(self.vit, "_ultralytics_keep_frozen"):
            delattr(self.vit, "_ultralytics_keep_frozen")
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vit.eval()
        return self

    def fuse(self) -> "DINOv3ViTSTA":
        return self
