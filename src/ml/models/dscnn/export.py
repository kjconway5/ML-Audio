import torch
import numpy as np
from dscnn import DSCNN

checkpoint = torch.load("dscnn7.pt", map_location="cpu")

cfg = checkpoint["config"]["model"]
model = DSCNN(
    n_classes=cfg["n_classes"],
    n_mels=checkpoint["config"]["preprocessing"]["n_mels"],
    first_conv_filters=cfg["first_conv"]["filters"],
    first_conv_kernel=tuple(cfg["first_conv"]["kernel_size"]),
    first_conv_stride=tuple(cfg["first_conv"]["stride"]),
    n_ds_blocks=cfg["ds_blocks"]["n_blocks"],
    ds_filters=cfg["ds_blocks"]["filters"],
    ds_kernel=tuple(cfg["ds_blocks"]["kernel_size"]),
    ds_stride=tuple(cfg["ds_blocks"]["stride"]),
)

model.eval()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ── Extract weights in exact layer order ──────────────────────────────────────
# Order must match what your layer_controller.v will expect in weight SRAM
layers = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.quantized.Conv2d):
        w = module.weight().int_repr().numpy().flatten().astype(np.int8)
        
        # Also grab per-channel quantization scale for requantization in RTL
        scale = module.weight().q_per_channel_scales().numpy()
        zero  = module.weight().q_per_channel_zero_points().numpy()
        
        layers.append({
            "name":   name,
            "shape":  module.weight().int_repr().numpy().shape,
            "offset": sum(len(l["weights"]) for l in layers),
            "weights": w,
            "scale":  scale,
            "zero":   zero,
        })
        print(f"{name:35s}  shape={module.weight().int_repr().numpy().shape}  "
              f"offset={layers[-1]['offset']:5d}  n_vals={len(w)}")

# ── Write weights.hex ─────────────────────────────────────────────────────────
all_weights = np.concatenate([l["weights"] for l in layers])
with open("weights.hex", "w") as f:
    for val in all_weights:
        f.write(f"{val & 0xFF:02x}\n")
print(f"\nweights.hex written — {len(all_weights)} total INT8 values")

# ── Write scales.txt for RTL requantization shifts ────────────────────────────
# The RTL requant module needs a right-shift value per layer.
# shift = round(-log2(scale)) — approximates the quantization scale as a power of 2
with open("scales.txt", "w") as f:
    f.write(f"{'layer':<35s}  {'mean_scale':>12s}  {'shift':>6s}\n")
    f.write("-" * 60 + "\n")
    for l in layers:
        mean_scale = float(np.mean(l["scale"]))
        import math
        shift = round(-math.log2(mean_scale))
        shift = max(0, min(31, shift))  # clamp to valid range
        f.write(f"{l['name']:<35s}  {mean_scale:>12.6f}  {shift:>6d}\n")
        print(f"{l['name']:35s}  scale={mean_scale:.6f}  shift={shift}")

# ── Write layer config for controller ROM ────────────────────────────────────
print("\n--- Layer config for controller.v ROM ---")
for l in layers:
    print(f"  name={l['name']:35s}  shape={str(l['shape']):20s}  "
          f"offset={l['offset']:5d}  n_vals={len(l['weights'])}")

print("\n--- Bias extraction ---")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.quantized.Conv2d):
        if module.bias() is not None:
            b = module.bias().detach().numpy()
            np.save(f"bias_{name}.npy", b)
            print(f"{name}: {b.shape}  dtype={b.dtype}")
        else:
            print(f"{name}: no bias")

print("\n--- Bias extraction (quantized to INT32) ---")

bias_layers = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.quantized.Conv2d):
        if module.bias() is not None:
            # Get the float bias
            b_float = module.bias().detach().numpy()
            
            # Get scales needed for bias quantization
            # Input activation scale (tracked by QAT observer)
            # Weight scale (per-channel, take mean for single scale approx)
            w_scale = module.weight().q_per_channel_scales().numpy().mean()
            
            # Bias scale = input_scale × weight_scale
            # Input scale is stored in the module's scale attribute
            input_scale = float(module.scale)
            bias_scale = input_scale * w_scale
            
            # Quantize bias to INT32
            b_int32 = np.round(b_float / bias_scale).astype(np.int32)
            
            np.save(f"bias_{name}.npy", b_int32)
            
            bias_layers.append({
                "name":   name,
                "offset": sum(len(l["bias"]) for l in bias_layers),
                "bias":   b_int32,
            })
            
            print(f"{name:35s}  shape={b_int32.shape}  "
                  f"offset={bias_layers[-1]['offset']:4d}  "
                  f"scale={bias_scale:.6f}")
        else:
            print(f"{name:35s}  no bias")

# Write bias.hex — INT32 values, each written as 8 hex chars (4 bytes)
all_biases = np.concatenate([l["bias"] for l in bias_layers])
with open("bias.hex", "w") as f:
    for val in all_biases:
        # Write as unsigned 32-bit hex, 8 characters wide
        f.write(f"{val & 0xFFFFFFFF:08x}\n")

print(f"\nbias.hex written — {len(all_biases)} total INT32 values")
print(f"Total bias storage: {len(all_biases) * 4} bytes")

print("\n--- Bias config for controller.v ROM ---")
for l in bias_layers:
    print(f"  name={l['name']:35s}  "
          f"offset={l['offset']:4d}  n_vals={len(l['bias'])}")

print("\n--- bias_sram.sv case statement (paste into module) ---")
offset = 0
for l in bias_layers:
    print(f"\n            // {l['name']} (offset {l['offset']}, {len(l['bias'])} values)")
    for i, val in enumerate(l["bias"]):
        unsigned = int(val) & 0xFFFFFFFF
        print(f"            8'd{offset + i}: data = 32'sh{unsigned:08X};")
    offset += len(l["bias"])
print(f"\n            // Total entries: {offset}")