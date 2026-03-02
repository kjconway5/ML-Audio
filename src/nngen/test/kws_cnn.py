import torch
import torch.nn as nn
import numpy as np
import nngen as ng

# ============================================================
# 1) Define a small KWS-style CNN (NO depthwise, NO Linear)
#    NNgen-compatible ops only
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.net(x)


class TinyKwsCNN(nn.Module):
    """
    Internal PyTorch layout: NCHW
    Logical input: (1, 49, 10, 1)  -> MFCC-style
    Output: (1, 12)
    """
    def __init__(self, n_classes=12):
        super().__init__()
        self.stem   = ConvBNReLU(1, 16)
        self.block1 = ConvBNReLU(16, 32)
        self.block2 = ConvBNReLU(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)           # (N, 64, 1, 1)
        x = self.classifier(x)     # (N, 12, 1, 1)
        x = torch.flatten(x, 1)    # (N, 12)
        return x


# ============================================================
# 2) NHWC wrapper (NNgen expects NHWC at the boundary)
# ============================================================

class NHWCWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_nhwc):
        # NHWC -> NCHW
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        return self.model(x)


model = NHWCWrapper(TinyKwsCNN(n_classes=12)).eval()

# ============================================================
# 3) Export to ONNX (legacy exporter = NNgen friendly)
# ============================================================

onnx_filename = "tiny_kws_cnn.onnx"

dummy_input = torch.randn(1, 49, 10, 1)  # NHWC

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    input_names=["act"],
    output_names=["out"],
    opset_version=11,
    do_constant_folding=True,
    dynamo=False   # IMPORTANT
)

print(f"Wrote ONNX: {onnx_filename}")

# ============================================================
# 4) Import ONNX into NNgen
# ============================================================

act_dtype    = ng.fixed16_8
weight_dtype = ng.fixed16_8
scale_dtype  = ng.fixed16_8
bias_dtype   = ng.fixed32_16

(outputs, placeholders, variables, constants, operators) = ng.from_onnx(
    onnx_filename,
    value_dtypes={},
    default_placeholder_dtype=act_dtype,
    default_variable_dtype=weight_dtype,
    default_constant_dtype=weight_dtype,
    default_operator_dtype=act_dtype,
    default_scale_dtype=scale_dtype,
    default_bias_dtype=bias_dtype,
    disable_fusion=False
)

print("Imported ONNX -> NNgen ✅")
print("placeholders:", list(placeholders.keys()))
print("outputs:", list(outputs.keys()))

# ============================================================
# 5) Generate Verilog RTL
# ============================================================

silent = False
axi_datawidth = 32

output_layer = list(outputs.values())[0]

rtl = ng.to_verilog(
    [output_layer],
    'tiny_kws_cnn',
    silent=silent,
    config={'maxi_datawidth': axi_datawidth}
)

with open("tiny_kws_cnn.v", "w") as f:
    f.write(rtl)

print("Verilog generated: tiny_kws_cnn.v")

# ============================================================
# 6) Export parameter memory image
# ============================================================

param_filename = "tiny_kws_cnn_params.npy"
chunk_size = 64

param_data = ng.export_ndarray([output_layer], chunk_size)
np.save(param_filename, param_data)

print("Params saved:", param_filename)

# ============================================================
# DONE
# ============================================================