#!/usr/bin/env python3
"""
export.py — Extract weights, biases, and requantization shifts from a trained
            QAT DSCNN checkpoint for use in the RTL testbench.

Usage (run from src/ml/models/dscnn/):
    python3 export.py [--ckpt dscnn-4-10.pt]

Outputs written to the same directory as the checkpoint:
    weights.hex  — all INT8 weights concatenated, one byte per line (2-digit hex)
    bias.hex     — all INT32 biases concatenated, one per line (8-digit hex)
    scales.txt   — per-layer requant shift values (paste into LAYER_CFGS in test)
    bias_DFFs.sv snippet printed to stdout for updating bias_dff/bias_DFFs.sv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dscnn import DSCNN


def load_model(ckpt_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg     = checkpoint["config"]["model"]
    preproc = checkpoint["config"]["preprocessing"]
    backend = checkpoint.get("qat_backend", "fbgemm")

    # "pow2" is our training label, not a PyTorch engine name.
    # The pow2 model uses per-tensor symmetric weights — qnnpack matches that.
    # fbgemm uses per-channel weights by default and would misload the state_dict.
    if backend == "pow2":
        backend = "qnnpack"

    model = DSCNN(
        n_classes=cfg["n_classes"],
        n_mels=preproc["n_mels"],
        first_conv_filters=cfg["first_conv"]["filters"],
        first_conv_kernel=tuple(cfg["first_conv"]["kernel_size"]),
        first_conv_stride=tuple(cfg["first_conv"]["stride"]),
        n_ds_blocks=cfg["ds_blocks"]["n_blocks"],
        ds_filters=cfg["ds_blocks"]["filters"],
        ds_kernel=tuple(cfg["ds_blocks"]["kernel_size"]),
        ds_stride=tuple(cfg["ds_blocks"]["stride"]),
    )
    torch.backends.quantized.engine = backend
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=HERE / "dscnn-4-13.pt")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ckpt_path.parent
    print(f"Loading checkpoint : {ckpt_path}")
    model, ckpt = load_model(ckpt_path)
    print(f"Labels             : {ckpt['labels']}")
    print(f"QAT backend        : {ckpt.get('qat_backend', 'fbgemm')}")
    print()

    # ── Collect all quantized Conv2d layers in order ──────────────────────────
    # true_input_scale: QuantStub scale for layer 0, previous layer's output_scale thereafter.
    # output_scale: module.scale (the activation scale of THIS layer's output).
    # Requant shift = round(-log2(true_input_scale * mean_wscale / output_scale)).
    quant_input_scale = float(model.quant.scale)

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Conv2d):
            w_int8 = module.weight().int_repr().numpy().flatten().astype(np.int8)
            # Support both per-channel and per-tensor weight quantization
            try:
                w_scale = module.weight().q_per_channel_scales().numpy()
            except RuntimeError:
                w_scale = np.array([float(module.weight().q_scale())])
            true_input_scale = quant_input_scale if len(layers) == 0 else layers[-1]["output_scale"]
            output_scale     = float(module.scale)
            layers.append({
                "name":              name,
                "shape":             tuple(module.weight().int_repr().numpy().shape),
                "offset":            sum(len(l["weights"]) for l in layers),
                "weights":           w_int8,
                "w_scale":           w_scale,
                "true_input_scale":  true_input_scale,
                "output_scale":      output_scale,
            })
            print(f"  {name:35s}  shape={str(layers[-1]['shape']):20s}  "
                  f"offset={layers[-1]['offset']:5d}  n={len(w_int8)}")

    # ── weights.hex ───────────────────────────────────────────────────────────
    all_weights = np.concatenate([l["weights"] for l in layers])
    weights_path = out_dir / "weights.hex"
    with open(weights_path, "w") as f:
        for v in all_weights:
            f.write(f"{int(v) & 0xFF:02x}\n")
    print(f"\nweights.hex  : {weights_path}  ({len(all_weights)} INT8 values)")

    # ── scales.txt (requant shifts for LAYER_CFGS) ────────────────────────────
    # shift = round(-log2(true_input_scale * mean_wscale / output_scale))
    scales_path = out_dir / "scales.txt"
    print(f"\n{'layer':<35s}  {'mean_w_scale':>14s}  {'in_scale':>10s}  {'out_scale':>10s}  {'shift':>6s}")
    print("-" * 83)
    with open(scales_path, "w") as f:
        f.write(f"{'layer':<35s}  {'mean_w_scale':>14s}  {'in_scale':>10s}  {'out_scale':>10s}  {'shift':>6s}\n")
        f.write("-" * 83 + "\n")
        for l in layers:
            mean_wscale   = float(np.mean(l["w_scale"]))
            requant_scale = l["true_input_scale"] * mean_wscale / l["output_scale"]
            shift = round(-math.log2(requant_scale))
            shift = max(0, min(31, shift))
            line = (f"{l['name']:<35s}  {mean_wscale:>14.8f}  "
                    f"{l['true_input_scale']:>10.6f}  {l['output_scale']:>10.6f}  {shift:>6d}")
            print(line)
            f.write(line + "\n")
    print(f"\nscales.txt   : {scales_path}")

    # ── bias.hex (quantized INT32) ────────────────────────────────────────────
    # bias_scale = true_input_scale * mean_wscale  (units of the accumulator before shift)
    bias_layers = []
    for l in layers:
        name   = l["name"]
        module = dict(model.named_modules())[name]
        if module.bias() is not None:
            b_float    = module.bias().detach().numpy()
            mean_wscale = float(np.mean(l["w_scale"]))
            bias_scale  = l["true_input_scale"] * mean_wscale
            b_int32     = np.round(b_float / bias_scale).astype(np.int32)
            bias_layers.append({
                "name":   name,
                "offset": sum(len(bl["bias"]) for bl in bias_layers),
                "bias":   b_int32,
            })

    all_biases = np.concatenate([l["bias"] for l in bias_layers])
    bias_path = out_dir / "bias.hex"
    with open(bias_path, "w") as f:
        for v in all_biases:
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")
    print(f"bias.hex     : {bias_path}  ({len(all_biases)} INT32 values)")

    # ── QuantStub input scale (for generate_spect.py / spect_buffer_ctrl) ─────
    input_scale = float(model.quant.scale)
    spect_shift = round(-math.log2(input_scale))
    spect_shift = max(0, min(15, spect_shift))
    print(f"\nQuantStub scale   : {input_scale:.8f}  → SPECT_SHIFT={spect_shift}")
    print("  (Update SPECT_SHIFT in spect_buffer_ctrl.sv if needed)")

    # ── bias_DFFs.sv case statement snippet ───────────────────────────────────
    print("\n" + "="*70)
    print("Paste the following into bias_dff/bias_DFFs.sv case statement:")
    print("="*70)
    offset = 0
    for l in bias_layers:
        print(f"\n    // {l['name']} (bias_off={l['offset']}, {len(l['bias'])} channels)")
        for i, val in enumerate(l["bias"]):
            u = int(val) & 0xFFFFFFFF
            print(f"    8'd{offset + i}: data = 32'sh{u:08X};")
        offset += len(l["bias"])
    print(f"\n    // Total entries: {offset}")

    # ── Summary for updating LAYER_CFGS in test_kws_top.py ───────────────────
    print("\n" + "="*70)
    print("Update LAYER_CFGS shifts in test_kws_top.py (field index 11):")
    print("="*70)
    for i, l in enumerate(layers):
        mean_wscale = float(np.mean(l["w_scale"]))
        shift = round(-math.log2(mean_wscale))
        shift = max(0, min(31, shift))
        print(f"  layer {i:2d}  {l['name']:35s}  shift={shift}")


if __name__ == "__main__":
    main()
