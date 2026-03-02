import torch
import torch.nn as nn
import numpy as np
import nngen as ng
from audiocnn import DSCNN

class NHWCWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_nhwc):
        # NHWC -> NCHW
        # different way to lay things out in a way nngen expects 
        # N = batch
        # H = height (time frames for audio)
        # W = width  (mel bins)
        # C = channels
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        return self.model(x)

# Build model
# build the base of the model from audiocnn.py
# then convert NHWC to NCHW for nngen compatibility with above wrapper

base = DSCNN(n_classes=3, n_mels=40).eval()
model = NHWCWrapper(base).eval()

# --- PyTorch -> ONNX ---

# file we're writing to
onnx_filename = "mlaudio_dscnn.onnx"

# NHWC: (N, T, F, C)
# random inputs for now until we know what they should be 
dummy_input = torch.randn(1, 49, 40, 1)

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    input_names=["act"],
    output_names=["out"],
    opset_version=11,
    do_constant_folding=True,
    dynamo=False,
)

print(f"Wrote ONNX: {onnx_filename}")

# --- ONNX -> NNgen ---

# onnx data types to nngen specific data types for an easier port
# this is all documented and pulled almost directly from nngen readme
act_dtype = ng.fixed16_8
weight_dtype = ng.fixed16_8
scale_dtype = ng.fixed16_8
bias_dtype = ng.fixed32_16

(outputs, placeholders, variables, constants, operators) = ng.from_onnx(
    onnx_filename,
    value_dtypes={},
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

# --- Verilog ---
# defining certain parameters for verilog generation
# NNGen to actual RTL file
axi_datawidth = 32
output_layer = list(outputs.values())[0]

rtl = ng.to_verilog(
    [output_layer],
    'mlaudio_dscnn',
    silent=False,
    config={'maxi_datawidth': axi_datawidth},
)

with open("mlaudio_dscnn.sv", "w") as f:
    f.write(rtl)

print("Verilog generated: mlaudio_dscnn.sv")

# --- Params ---
# param_filename = "mlaudio_dscnn_params.npy"
# chunk_size = 64
# param_data = ng.export_ndarray([output_layer], chunk_size)
# np.save(param_filename, param_data)
# print("Params saved:", param_filename)