"""Export the YOLO26Backbone block from trained YOLO27 weights.

Given a checkpoint trained from a YAML that uses ``YOLO26Backbone`` as its
first layer (e.g. ``yolo27.yaml`` for detection or ``yolo27-cls.yaml`` for
classification), this script extracts the backbone sub-module and saves it
as well as the remaining head, both as standalone artefacts:

* ``<out>.pt``        -- backbone state_dict + meta info, ready to be loaded
                          into a fresh ``YOLO26Backbone``.
* ``<out>.onnx``      -- standalone backbone ONNX graph (single image input,
                          single P5 output -- or ``[P3, P4, P5]`` if the
                          backbone was built with ``multi_scale=True``).
* ``<out>_head.onnx`` -- standalone head ONNX graph: takes the backbone's
                          feature map(s) as inputs and emits the model's
                          final outputs (raw detection tensors / class
                          logits, depending on the task).

Examples:
    # Detection checkpoint -> backbone + head artefacts
    python tools/export_yolo26_backbone.py \
        --weights runs/detect/train/weights/best.pt \
        --out backbones/yolo27n \
        --imgsz 640

    # Classification pretrain -> backbone + head artefacts
    python tools/export_yolo26_backbone.py \
        --weights runs/classify/train/weights/best.pt \
        --out backbones/yolo27n_cls \
        --imgsz 224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running this script directly without an editable install of ultralytics.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import YOLO26Backbone  # noqa: E402
from ultralytics.nn.modules.head import Classify, Detect, OBB, Pose, Segment  # noqa: E402

# Heads whose forward returns auxiliary tensors in eval mode and only the final
# exported tensor when ``self.export = True``.
_EXPORT_CLEAN_OUTPUT = (Detect, Pose, Segment, OBB, Classify)


def find_backbone(model: torch.nn.Module) -> tuple[int, YOLO26Backbone]:
    """Locate the YOLO26Backbone instance inside a parsed YOLO model."""
    # Parsed model exposes ``.model`` (nn.Sequential of layers from YAML).
    layers = getattr(model, "model", model)
    for i, m in enumerate(layers):
        if isinstance(m, YOLO26Backbone):
            return i, m
    raise RuntimeError(
        "No YOLO26Backbone module found in the checkpoint. "
        "Make sure the model was trained from a YAML using YOLO26Backbone "
        "(e.g. yolo27.yaml or yolo27-cls.yaml)."
    )


def backbone_meta(bb: YOLO26Backbone) -> dict:
    """Collect constructor-equivalent metadata so the backbone can be rebuilt."""
    build_args = getattr(bb, "_build_args", None)
    if not build_args:
        raise RuntimeError(
            "YOLO26Backbone instance is missing the `_build_args` attribute. "
            "This checkpoint was likely produced by an older revision of "
            "YOLO26Backbone that did not persist constructor metadata. "
            "Re-train (or at least re-instantiate and re-save) the model with "
            "the current code so that the backbone can be rebuilt from the .pt file."
        )
    meta = {"class": "YOLO26Backbone"}
    meta.update(build_args)
    meta.update(
        out_channels_p3=int(bb.out_channels_p3),
        out_channels_p4=int(bb.out_channels_p4),
        out_channels_p5=int(bb.out_channels_p5),
    )
    return meta


def export_pt(bb: YOLO26Backbone, path: Path) -> None:
    """Save backbone state_dict + meta to a plain PyTorch checkpoint."""
    payload = {
        "state_dict": bb.state_dict(),
        "meta": backbone_meta(bb),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"  - saved PT  : {path}  ({path.stat().st_size / 1e6:.2f} MB)")


def export_onnx(bb: YOLO26Backbone, path: Path, imgsz: int, opset: int, dynamic: bool) -> None:
    """Export the backbone to a standalone ONNX graph."""
    bb = bb.eval()
    dummy = torch.zeros(1, bb.stem0.conv.in_channels, imgsz, imgsz)

    # Output names depend on multi_scale flag
    output_names = ["p3", "p4", "p5"] if bb.multi_scale else ["p5"]

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "H", 3: "W"}}
        for name in output_names:
            # Use distinct symbolic names per output so consumers don't infer
            # equality constraints between feature maps of different strides.
            dynamic_axes[name] = {0: "batch", 2: f"H_{name}", 3: f"W_{name}"}

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        bb,
        dummy,
        str(path),
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,  # use legacy TorchScript exporter; dynamo path has opset-mismatch issues here
    )
    print(f"  - saved ONNX: {path}  ({path.stat().st_size / 1e6:.2f} MB, opset={opset})")


class HeadModule(torch.nn.Module):
    """Replays a parsed YOLO model from layer 1 onward, treating the backbone outputs as inputs.

    Mirrors ``BaseModel._predict_once`` but seeds the layer-output cache with the
    pre-computed backbone feature map(s) at index 0, so the wrapped graph contains
    everything *except* the backbone block.
    """

    def __init__(self, full_model: torch.nn.Module, backbone_idx: int, multi_scale: bool):
        super().__init__()
        layers = list(getattr(full_model, "model", full_model))
        if backbone_idx != 0:
            raise RuntimeError(f"Backbone must be at layer 0 for head export, got idx={backbone_idx}.")
        # Keep all layers except the backbone; preserve their .f / .i / save attrs.
        self.head_layers = torch.nn.ModuleList(layers[1:])
        self.save = set(getattr(full_model, "save", []))
        self.multi_scale = multi_scale
        # Force export-capable heads into export mode so the ONNX graph emits only
        # the final prediction tensor instead of also leaking auxiliary tensors
        # used during training/loss computation.
        for m in self.head_layers:
            if isinstance(m, _EXPORT_CLEAN_OUTPUT):
                m.export = True

    def forward(self, *features):
        """Run the head graph given backbone features (P3, P4, P5) or (P5,)."""
        feats = list(features)
        # y[0] = backbone output (list for multi_scale, tensor otherwise)
        y: list = [feats if self.multi_scale else feats[0]]
        x = y[0]
        for m in self.head_layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            # Pad y so indices match the original model's layer numbering.
            while len(y) < m.i:
                y.append(None)
            y.append(x if m.i in self.save else None)
        return x


def _flatten(out) -> list[torch.Tensor]:
    """Flatten nested list/tuple/dict outputs into a flat list of tensors."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        flat = []
        for v in out.values():
            flat.extend(_flatten(v))
        return flat
    if isinstance(out, (list, tuple)):
        flat = []
        for v in out:
            flat.extend(_flatten(v))
        return flat
    return []


def export_head_onnx(
    full_model: torch.nn.Module,
    bb: YOLO26Backbone,
    backbone_idx: int,
    path: Path,
    imgsz: int,
    opset: int,
    dynamic: bool,
) -> None:
    """Export the head (everything after the backbone) to a standalone ONNX graph."""
    head = HeadModule(full_model, backbone_idx, multi_scale=bb.multi_scale).eval()

    # Build dummy backbone outputs by running the actual backbone once.
    bb_eval = bb.eval()
    with torch.no_grad():
        dummy_img = torch.zeros(1, bb.stem0.conv.in_channels, imgsz, imgsz)
        feats = bb_eval(dummy_img)
        feats = feats if isinstance(feats, (list, tuple)) else [feats]

    input_names = ["p3", "p4", "p5"] if bb.multi_scale else ["p5"]
    dummy_inputs = tuple(f.detach() for f in feats)

    # Discover output names by running the head once.
    with torch.no_grad():
        out = head(*dummy_inputs)
    flat_out = _flatten(out)
    output_names = [f"output{i}" if len(flat_out) > 1 else "output" for i in range(len(flat_out))]

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {}
        # Feature-map inputs: distinct H/W symbol per stride to avoid implied equalities.
        for name, t in zip(input_names, dummy_inputs):
            if t.ndim == 4:
                dynamic_axes[name] = {0: "batch", 2: f"H_{name}", 3: f"W_{name}"}
            else:
                dynamic_axes[name] = {0: "batch"}
        for name, t in zip(output_names, flat_out):
            if t.ndim == 4:
                dynamic_axes[name] = {0: "batch", 2: f"H_{name}", 3: f"W_{name}"}
            elif t.ndim == 3:
                # Detect-style (B, C, N) prediction: anchor count varies with input H/W.
                dynamic_axes[name] = {0: "batch", 2: "anchors"}
            elif t.ndim >= 2:
                dynamic_axes[name] = {0: "batch"}

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        head,
        dummy_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  - saved HEAD: {path}  ({path.stat().st_size / 1e6:.2f} MB, opset={opset})")
    print(f"      head inputs : {list(zip(input_names, [tuple(t.shape) for t in dummy_inputs]))}")
    print(f"      head outputs: {list(zip(output_names, [tuple(t.shape) for t in flat_out]))}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO26Backbone from a trained YOLO27 checkpoint.")
    p.add_argument("--weights", type=Path, required=True, help="Path to trained .pt checkpoint.")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output basename (without extension). Two files will be written: <out>.pt and <out>.onnx.",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Dummy input spatial size for ONNX export.")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    p.add_argument("--dynamic", action="store_true", help="Export ONNX with dynamic batch / spatial axes.")
    p.add_argument("--no-onnx", action="store_true", help="Skip backbone ONNX export.")
    p.add_argument("--no-pt", action="store_true", help="Skip PT export.")
    p.add_argument("--no-head", action="store_true", help="Skip head ONNX export.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    print(f"Loading checkpoint: {args.weights}")
    model = YOLO(str(args.weights)).model
    idx, bb = find_backbone(model)
    print(f"Found YOLO26Backbone at layer index {idx}")
    print(f"  meta: {backbone_meta(bb)}")

    out_pt = args.out.with_suffix(".pt")
    out_onnx = args.out.with_suffix(".onnx")
    out_head = args.out.with_name(args.out.name + "_head").with_suffix(".onnx")

    if not args.no_pt:
        export_pt(bb, out_pt)
    if not args.no_onnx:
        export_onnx(bb, out_onnx, imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic)
    if not args.no_head:
        export_head_onnx(model, bb, idx, out_head, imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic)

    print("Done.")


if __name__ == "__main__":
    main()
