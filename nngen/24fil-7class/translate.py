import torch
import numpy as np
import nngen as ng
from audiocnn import DSCNN

file = "24fil-7class"

# --- Load checkpoint + build model ---
ckpt = torch.load(f"{file}.pt", map_location="cpu")

model = DSCNN(
    n_classes=len(ckpt["labels"]), 
    n_mels=ckpt["config"]["preprocessing"]["n_mels"], 
).eval()

sd = ckpt["model_state_dict"]

# remove quant metadata keys that don't exist in the float model
sd = {k: v for k, v in sd.items()
      if not (k.endswith(".scale") or k.endswith(".zero_point") or k.startswith("quant."))}

# dequantize int8 weights -> float weights so PyTorch can load them
for k, v in list(sd.items()):
    if torch.is_tensor(v) and v.is_quantized:
        sd[k] = v.dequantize()

# your checkpoint doesn't have BN params, so strict must be False
missing, unexpected = model.load_state_dict(sd, strict=False)

print("Loaded ✅")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()
print("Loaded ✅")

# --- PyTorch -> ONNX ---
onnx_filename = f"{file}.onnx"

# NCHW: (N, C, T, F)
dummy_input = torch.randn(1, 1, 49, 40)

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    input_names=["act"],
    output_names=["out"],
    opset_version=11,
    do_constant_folding=True,
    dynamo=False,      # ✅ force legacy exporter
)

print(f"Wrote ONNX: {onnx_filename}")

# --- ONNX -> NNgen ---
act_dtype = ng.fixed16_8
weight_dtype = ng.fixed16_8
scale_dtype = ng.fixed16_8
bias_dtype = ng.fixed32_16

(outputs, placeholders, variables, constants, operators) = ng.from_onnx(
    onnx_filename,
    default_placeholder_dtype=act_dtype,
    default_variable_dtype=weight_dtype,
    default_constant_dtype=weight_dtype,
    default_operator_dtype=act_dtype,
    default_scale_dtype=scale_dtype,
    default_bias_dtype=bias_dtype,
    disable_fusion=False,
)

print("Imported ONNX -> NNgen ✅")
print("placeholders:", list(placeholders.keys()))
print("outputs:", list(outputs.keys()))

output_layer = list(outputs.values())[0]

# ✅ Quantize graph (this is the key “int/fixed-point” step)
ng.quantize([output_layer], {'act': 8})

# --- Verilog ---
axi_datawidth = 32
rtl = ng.to_verilog(
    [output_layer],
    file,
    silent=False,
    config={'maxi_datawidth': axi_datawidth},
)

with open(f"{file}.sv", "w") as f:
    f.write(rtl)

print("Verilog generated:", f"{file}.sv")

# --- Params (you usually want this) ---
param_filename = f"{file}_params.npy"
chunk_size = 64
param_data = ng.export_ndarray([output_layer], chunk_size)
np.save(param_filename, param_data)
print("Params saved:", param_filename)