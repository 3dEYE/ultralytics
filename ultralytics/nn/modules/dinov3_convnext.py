# Ultralytics ?? AGPL-3.0 License - https://ultralytics.com/license
"""DINOv3 ConvNeXt backbone adapter for Ultralytics YOLO.

Uses a vendored copy of the ConvNeXt implementation (see `_dinov3_convnext_impl.py`)
so it runs on production machines without the upstream DINOv3 source tree.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ._dinov3_convnext_impl import Block, ConvNeXt, LayerNorm, convnext_sizes

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
    """Wraps a DINOv3 ConvNeXt and yields (P3, P4, P5) tuple.

    YAML args: [variant, pretrained, freeze, weights]
      variant    : "tiny" | "small" | "base" | "large"
      pretrained : bool
      freeze     : bool - also sets a marker the Ultralytics trainer respects
      weights    : "LVD1689M" | "SAT493M" | local path | URL
    """

    out_indices: Tuple[int, int, int] = (1, 2, 3)  # strides 8, 16, 32

    def __init__(
        self,
        variant: str = "small",
        pretrained: bool = False,
        freeze: bool = True,
        weights: Optional[str] = "LVD1689M",
        imagenet_norm: bool = True,
    ) -> None:
        super().__init__()
        assert variant in convnext_sizes, f"unknown variant: {variant}"
        self.variant = variant
        self.imagenet_norm = imagenet_norm

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

        # Persistent fused-state marker (see ``fuse``). Stored as a buffer so it
        # round-trips through ``state_dict`` and we can re-apply the runtime
        # adjustments after ``load_state_dict``.
        self.register_buffer("_fused", torch.zeros((), dtype=torch.bool), persistent=True)

        # Re-apply fused-state side effects after ``load_state_dict`` (incl. wrappers
        # that recurse via ``_load_from_state_dict``). Public API in torch>=2.0,
        # private in torch>=1.10; we accept either.
        _register_post_hook = getattr(self, "register_load_state_dict_post_hook", None) or getattr(
            self, "_register_load_state_dict_post_hook", None
        )
        if _register_post_hook is not None:
            _register_post_hook(lambda module, _ic: module._sync_fused_state())

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

    # ------------------------------------------------------------------ fuse
    @staticmethod
    def _take_affine(norm: nn.Module, target: nn.Module):
        """Return ``(weight, bias)`` of an active LayerNorm-like ``norm`` ready to fold into ``target``.

        Returns ``None`` if there is nothing to fold (already neutralized, missing
        params, or both are exactly identity ``(1, 0)``).
        """
        if not getattr(norm, "_affine_active", True):
            return None
        weight = getattr(norm, "weight", None)
        bias = getattr(norm, "bias", None)
        if weight is None or bias is None:
            return None
        ref = target.weight
        weight = weight.detach().to(device=ref.device, dtype=ref.dtype)
        bias = bias.detach().to(device=ref.device, dtype=ref.dtype)
        if torch.all(weight == 1) and torch.all(bias == 0):
            return None
        return weight, bias

    @staticmethod
    def _neutralize_layernorm_affine(norm: nn.Module) -> None:
        """Set LN affine to identity ``(1, 0)`` and disable its application."""
        with torch.no_grad():
            if getattr(norm, "weight", None) is not None:
                norm.weight.fill_(1)
                norm.weight.requires_grad_(False)
            if getattr(norm, "bias", None) is not None:
                norm.bias.zero_()
                norm.bias.requires_grad_(False)
        norm._affine_active = False

    @staticmethod
    @torch.no_grad()
    def _fuse_layernorm_affine_into_linear(norm: nn.Module, linear: nn.Linear) -> None:
        """Fold LayerNorm affine ``(γ, β)`` into the following ``Linear``."""
        affine = DINOv3ConvNeXt._take_affine(norm, linear)
        if affine is not None:
            weight, bias = affine
            fused_bias = linear.weight.detach().matmul(bias)
            if linear.bias is None:
                linear.bias = nn.Parameter(fused_bias, requires_grad=False)
            else:
                linear.bias.add_(fused_bias)
            linear.weight.mul_(weight.reshape(1, -1))
            linear.weight.requires_grad_(False)
            linear.bias.requires_grad_(False)
        DINOv3ConvNeXt._neutralize_layernorm_affine(norm)

    @staticmethod
    @torch.no_grad()
    def _fuse_layernorm_affine_into_conv(norm: nn.Module, conv: nn.Conv2d) -> None:
        """Fold LayerNorm affine ``(γ, β)`` into the following dense ``Conv2d``.

        ``RuntimeError`` (not ``assert``) is intentional: ``python -O`` strips
        asserts and would otherwise let a miscompiled grouped-conv silently
        produce wrong weights.
        """
        if conv.groups != 1:
            raise RuntimeError(
                f"LayerNorm affine fusion into grouped conv is not implemented "
                f"(got groups={conv.groups}); refusing to corrupt weights."
            )
        affine = DINOv3ConvNeXt._take_affine(norm, conv)
        if affine is not None:
            weight, bias = affine
            conv_w = conv.weight.detach()
            fused_bias = (conv_w * bias.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3))
            if conv.bias is None:
                conv.bias = nn.Parameter(fused_bias, requires_grad=False)
            else:
                conv.bias.add_(fused_bias)
            conv.weight.mul_(weight.reshape(1, -1, 1, 1))
            conv.weight.requires_grad_(False)
            conv.bias.requires_grad_(False)
        DINOv3ConvNeXt._neutralize_layernorm_affine(norm)

    @staticmethod
    @torch.no_grad()
    def _fuse_layer_scale_into_pwconv2(block: "Block") -> None:
        """Fold per-channel layer-scale ``gamma`` into the preceding ``pwconv2``.

        For ``y = gamma ⊙ (W · x + b)`` the equivalent gamma-free form is
        ``W' = gamma ⊙ W`` (per-output-row scale) and ``b' = gamma ⊙ b``. After
        folding, ``gamma`` is set to ones (kept in state_dict for strict loads)
        and ``_layer_scale_active`` is disabled.
        """
        gamma = getattr(block, "gamma", None)
        if gamma is None or not getattr(block, "_layer_scale_active", True):
            return
        pwconv2 = block.pwconv2
        g = gamma.detach().to(device=pwconv2.weight.device, dtype=pwconv2.weight.dtype)
        if not torch.all(g == 1):
            pwconv2.weight.mul_(g.reshape(-1, 1))
            if pwconv2.bias is None:
                pwconv2.bias = nn.Parameter(
                    torch.zeros(pwconv2.out_features, device=pwconv2.weight.device, dtype=pwconv2.weight.dtype),
                    requires_grad=False,
                )
            else:
                pwconv2.bias.mul_(g)
            pwconv2.weight.requires_grad_(False)
            pwconv2.bias.requires_grad_(False)
            with torch.no_grad():
                gamma.fill_(1)
        gamma.requires_grad_(False)
        block._layer_scale_active = False

    @torch.no_grad()
    def _fuse_imagenet_norm_into_stem(self) -> None:
        """Fold input ImageNet ``(x - mean) / std`` into the first stem ``Conv2d``."""
        if not self.imagenet_norm or not hasattr(self, "_mean") or not hasattr(self, "_std"):
            return
        stem = self.downsample_layers[0]
        conv = stem[0] if isinstance(stem, nn.Sequential) and len(stem) else None
        if not isinstance(conv, nn.Conv2d) or conv.in_channels != self._mean.numel():
            return
        if any(p != 0 for p in conv.padding):
            return  # implicit zero-pad would break the equivalence; skip silently

        mean = self._mean.reshape(-1).to(device=conv.weight.device, dtype=conv.weight.dtype)
        std = self._std.reshape(-1).to(device=conv.weight.device, dtype=conv.weight.dtype)
        weight = conv.weight.detach().clone()
        if conv.bias is None:
            conv.bias = nn.Parameter(
                torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype),
                requires_grad=False,
            )
        conv.weight.copy_(weight / std.reshape(1, -1, 1, 1))
        conv.bias.sub_((weight * (mean / std).reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3)))
        conv.weight.requires_grad_(False)
        conv.bias.requires_grad_(False)
        self.imagenet_norm = False

    def _ensure_fuse_compat_state(self) -> None:
        """Create fuse-related runtime state missing from old pickled models.

        PyTorch full-module checkpoints bypass ``__init__`` during ``torch.load``.
        Models saved before the fuse machinery existed therefore lack the
        persistent ``_fused`` buffer and the per-module runtime flags. Treat them
        as unfused and install the missing attributes lazily before
        inference-time auto-fuse touches them.
        """
        if "_fused" not in self._buffers:
            device = next(self.parameters(), torch.empty(0)).device
            self.register_buffer("_fused", torch.zeros((), dtype=torch.bool, device=device), persistent=True)
        for m in self.modules():
            if isinstance(m, LayerNorm) and not hasattr(m, "_affine_active"):
                m._affine_active = True
            if isinstance(m, Block) and not hasattr(m, "_layer_scale_active"):
                m._layer_scale_active = True

    def _sync_fused_state(self) -> None:
        """Re-apply fused-state side effects after ``load_state_dict``.

        When a fused checkpoint is loaded into a freshly built instance, the
        persistent ``_fused`` buffer comes back as ``True``. We mirror the runtime
        flags so the eager forward matches what produced the checkpoint. The stem
        LayerNorm is intentionally kept active because its output feeds the first
        block's residual identity branch and is not folded by ``fuse()``.
        """
        self._ensure_fuse_compat_state()
        if not bool(self._fused):
            return
        self.imagenet_norm = False
        for m in self.modules():
            if isinstance(m, LayerNorm):
                m._affine_active = False
            elif isinstance(m, Block):
                m._layer_scale_active = False
        stem = self.downsample_layers[0]
        if isinstance(stem, nn.Sequential) and len(stem) > 1 and isinstance(stem[1], LayerNorm):
            stem[1]._affine_active = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Inject ``_fused=False`` for legacy checkpoints saved before this buffer existed.

        Absence of ``_fused`` unambiguously means the checkpoint predates the
        fused-state machinery, i.e. weights are unfused. We must therefore inject
        ``False`` (not ``self._fused``) -- otherwise loading a legacy checkpoint
        into an instance on which ``fuse()`` was already called would leave the
        ``_fused=True`` flag on top of unfused weights and silently mismatch the
        runtime flags with the actual parameters.
        """
        self._ensure_fuse_compat_state()
        key = prefix + "_fused"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self._fused)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @torch.no_grad()
    def fuse(self) -> "DINOv3ConvNeXt":
        """Fuse inference-only affine operations to simplify ONNX/TensorRT graphs.

        Folds in this order:

        * ImageNet ``(x - mean) / std`` → stem ``Conv2d`` weight/bias.
        * Downsample-stage ``LayerNorm(channels_first)`` (stages 1..3) →
          following dense ``Conv2d``.
        * Per-block ``LayerNorm(channels_last)`` affine → following ``pwconv1``
          ``Linear``; per-block layer-scale ``gamma`` → preceding ``pwconv2``
          ``Linear``.

        The stem ``LayerNorm`` (``downsample_layers[0][1]``) is intentionally
        preserved because it feeds the first block's residual identity branch.
        Idempotent: re-calling is a no-op.
        """
        self._ensure_fuse_compat_state()
        if bool(self._fused):
            return self
        self._fuse_imagenet_norm_into_stem()
        for downsample in self.downsample_layers[1:]:
            if isinstance(downsample, nn.Sequential) and len(downsample) >= 2:
                self._fuse_layernorm_affine_into_conv(downsample[0], downsample[1])
        for m in self.modules():
            if isinstance(m, Block):
                self._fuse_layernorm_affine_into_linear(m.norm, m.pwconv1)
                self._fuse_layer_scale_into_pwconv2(m)
        self._fused.fill_(True)
        return self
