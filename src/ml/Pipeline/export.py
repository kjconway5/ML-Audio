#!/usr/bin/env python3
"""
export.py — Extract weights, biases, and requantization shifts from a trained
            QAT DSCNN checkpoint for use in the RTL testbench.

Usage (run from src/ml/Pipeline/):
    python3 export.py [--ckpt ../models/dscnn-32mac-v1/dscnn-32mac-v1.pt]

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

HERE      = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "models" / "dscnn"
sys.path.insert(0, str(MODEL_DIR))
from dscnn import DSCNN


def compute_mult_shift(scale: float) -> tuple:
    """
    Decompose a float requantization scale into (mult, shift) for the RTL
    multiply-shift requantizer.

    RTL operation:
        product[63:0] = acc * mult          (32-bit signed × 32-bit unsigned)
        result        = product[63:32] >>> shift

    Effective scale: mult * 2^-(shift + 32)

    mult is chosen to fall in [2^31, 2^32) for maximum 32-bit precision.
    shift (exponent n) is clamped to [0, 31].
    """
    if scale <= 0:
        raise ValueError(f"requant scale must be positive, got {scale}")
    raw_exp = -math.log2(scale)
    shift   = max(0, min(31, math.ceil(raw_exp) - 1))
    mult    = round(scale * (1 << (shift + 32)))
    mult    = min(max(mult, 0), (1 << 32) - 1)   # clamp to 32-bit unsigned
    return mult, shift


def _int8_sym_qconfig():
    """Static-PTQ version of the INT8 symmetric qconfig used by get_int8_sym_qat_qconfig()."""
    obs = torch.quantization.observer.MinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    return torch.quantization.QConfig(activation=obs, weight=obs)


def load_model(ckpt_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg     = checkpoint["config"]["model"]
    preproc = checkpoint["config"]["preprocessing"]
    backend = checkpoint.get("qat_backend", "fbgemm")

    # Map training-time labels to PyTorch engine names.
    # qnnpack_int8sym: INT8 symmetric activations — reconstruct with matching qconfig.
    # pow2/qnnpack: per-tensor symmetric weights, quint8 activations.
    # fbgemm: per-channel weights — do NOT use (mean_wscale approximation causes errors).
    int8_sym = (backend == "qnnpack_int8sym")
    if backend in ("pow2", "qnnpack_int8sym"):
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
    model.qconfig = _int8_sym_qconfig() if int8_sym else torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=MODEL_DIR.parent / "dscnn-32requant-v11" / "dscnn-32requant-v11.pt")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not ckpt_path.exists():
        alt = HERE.parent / "models" / ckpt_path
        if alt.exists():
            ckpt_path = alt
        else:
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

    # ── scales.txt (requant mult + shift for LAYER_CFGS) ─────────────────────
    # Multiply-shift: effective scale = mult * 2^-(shift+32)
    # mult falls in [2^31, 2^32) for maximum 32-bit precision.
    # shift (last column) is kept last so update_rtl.py's load_shifts() still works.
    scales_path = out_dir / "scales.txt"
    hdr = f"{'layer':<35s}  {'mean_w_scale':>14s}  {'in_scale':>10s}  {'out_scale':>10s}  {'mult':>10s}  {'shift':>6s}"
    sep = "-" * 95
    print(f"\n{hdr}")
    print(sep)
    with open(scales_path, "w") as f:
        f.write(hdr + "\n")
        f.write(sep + "\n")
        for l in layers:
            mean_wscale   = float(np.mean(l["w_scale"]))
            requant_scale = l["true_input_scale"] * mean_wscale / l["output_scale"]
            mult, shift   = compute_mult_shift(requant_scale)
            line = (f"{l['name']:<35s}  {mean_wscale:>14.8f}  "
                    f"{l['true_input_scale']:>10.6f}  {l['output_scale']:>10.6f}  "
                    f"{mult:>10d}  {shift:>6d}")
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
    # The mel features are Q6.10 fixed-point (divide by 2^Q_FRAC to get float log2).
    # RTL applies: int8 = Q6.10_uint16 >> SPECT_SHIFT
    # Match condition: 2^SPECT_SHIFT ≈ 2^Q_FRAC / (1/scale) = 2^Q_FRAC * scale
    # → SPECT_SHIFT = round(Q_FRAC + log2(scale)) = Q_FRAC - round(-log2(scale))
    Q_FRAC_LOG = 10  # must match golden_model.py Q_FRAC
    input_scale = float(model.quant.scale)
    spect_shift = Q_FRAC_LOG - round(-math.log2(input_scale))
    spect_shift = max(0, min(15, spect_shift))
    print(f"\nQuantStub scale   : {input_scale:.8f}  → SPECT_SHIFT={spect_shift}")
    print("  (Update SPECT_SHIFT in spect_buffer_ctrl.sv if needed)")

    # ── bias_DFFs.sv case statement snippet ───────────────────────────────────
    total_biases = len(all_biases)
    addr_bits = max(8, total_biases - 1).bit_length()   # 8 for ≤255, 9 for ≤511, etc.
    print("\n" + "="*70)
    print("Paste the following into bias_dff/bias_DFFs.sv case statement:")
    print("="*70)
    offset = 0
    for l in bias_layers:
        print(f"\n    // {l['name']} (bias_off={l['offset']}, {len(l['bias'])} channels)")
        for i, val in enumerate(l["bias"]):
            u = int(val) & 0xFFFFFFFF
            print(f"    {addr_bits}'d{offset + i}: data = 32'sh{u:08X};")
        offset += len(l["bias"])
    print(f"\n    // Total entries: {offset}  (addr_bits={addr_bits})")

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
