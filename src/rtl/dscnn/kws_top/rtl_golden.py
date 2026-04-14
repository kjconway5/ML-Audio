"""
rtl_golden.py — Shared RTL arithmetic golden model for kws_top.

Provides:
  rtl_golden_predict(spect_int8, weights, biases) -> (class_idx, gap_scores)
  LAYER_CFGS_ARCH — architecture-fixed fields (everything except shift)
  load_layer_cfgs(scales_path) -> LAYER_CFGS list with shifts from scales.txt
  load_hex_file(path, signed, width) -> list of ints

This is the canonical integer-arithmetic reference that matches the RTL exactly.
Both test_kws_top.py and generate_spect.py import from here so they always agree.
"""

from pathlib import Path


# ── Architecture-fixed layer parameters ───────────────────────────────────────
# Format: (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, w_off, relu, ofmap_h, ofmap_w, bias_off)
# 'shift' (field 11 in full LAYER_CFGS) is omitted here and filled in by load_layer_cfgs().
LAYER_CFGS_ARCH = [
    #  layer  in  out  kH  kW  sh sw ph pw  dw   w_off  relu  oh  ow bias_off
    (   0,    1,  24, 10,  4,  2, 2, 4, 1,  0,     0,    1,  25, 20,   0),
    (   1,    1,  24,  3,  3,  1, 1, 1, 1,  1,   960,    1,  25, 20,  24),
    (   2,   24,  24,  1,  1,  1, 1, 0, 0,  0,  1176,    1,  25, 20,  48),
    (   3,    1,  24,  3,  3,  1, 1, 1, 1,  1,  1752,    1,  25, 20,  72),
    (   4,   24,  24,  1,  1,  1, 1, 0, 0,  0,  1968,    1,  25, 20,  96),
    (   5,    1,  24,  3,  3,  1, 1, 1, 1,  1,  2544,    1,  25, 20, 120),
    (   6,   24,  24,  1,  1,  1, 1, 0, 0,  0,  2760,    1,  25, 20, 144),
    (   7,    1,  24,  3,  3,  1, 1, 1, 1,  1,  3336,    1,  25, 20, 168),
    (   8,   24,  24,  1,  1,  1, 1, 0, 0,  0,  3552,    1,  25, 20, 192),
    (   9,   24,   7,  1,  1,  1, 1, 0, 0,  0,  4128,    0,  25, 20, 216),
]

LAYER_NAMES = [
    "first_conv",
    "ds0.depthwise",
    "ds0.pointwise",
    "ds1.depthwise",
    "ds1.pointwise",
    "ds2.depthwise",
    "ds2.pointwise",
    "ds3.depthwise",
    "ds3.pointwise",
    "classifier",
]


def load_shifts(scales_path: Path) -> list:
    """Parse shift values (last column) from scales.txt."""
    shifts = []
    with open(scales_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("layer") or s.startswith("-"):
                continue
            shifts.append(int(s.split()[-1]))
    return shifts


def load_layer_cfgs(scales_path: Path) -> list:
    """
    Build the full LAYER_CFGS list by merging LAYER_CFGS_ARCH with shift values
    from scales.txt. Returns tuples in the format expected by rtl_golden_predict():
      (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, w_off, shift, relu, ofmap_h, ofmap_w, bias_off)
    """
    shifts = load_shifts(scales_path)
    if len(shifts) != len(LAYER_CFGS_ARCH):
        raise ValueError(
            f"scales.txt has {len(shifts)} shifts but architecture has {len(LAYER_CFGS_ARCH)} layers"
        )
    cfgs = []
    for arch, shift in zip(LAYER_CFGS_ARCH, shifts):
        layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, w_off, relu, oh, ow, bias_off = arch
        cfgs.append((layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, w_off, shift, relu, oh, ow, bias_off))
    return cfgs


def load_hex_file(path, signed=False, width=8) -> list:
    """Load a hex file (one value per line). Returns list of ints."""
    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if signed and v >= (1 << (width - 1)):
                v -= (1 << width)
            values.append(v)
    return values


def rtl_golden_predict(spect_int8: list, weights: list, biases: list,
                       layer_cfgs: list = None) -> tuple:
    """
    Pure-Python RTL arithmetic golden model.

    Executes the same integer operations as kws_top RTL:
      - INT8 MAC accumulation into INT32
      - Add INT32 bias
      - Arithmetic right-shift by layer shift
      - Saturate to INT8 [-128, 127]
      - ReLU if relu=1
      - Global average pool (sum over spatial) on final layer

    Args:
        spect_int8  : 2000 INT8 spectrogram values (flat, frame-major)
        weights     : all INT8 weights concatenated
        biases      : all INT32 biases concatenated
        layer_cfgs  : LAYER_CFGS list; if None, uses the hardcoded values
                      (only valid if shifts were already set via update_rtl.py)

    Returns:
        (predicted_class_idx, gap_scores_list)
    """
    if layer_cfgs is None:
        # Fall back to module-level LAYER_CFGS if caller didn't supply one.
        # This requires that LAYER_CFGS is defined (imported from test_kws_top or set globally).
        raise ValueError("layer_cfgs must be provided — call load_layer_cfgs(scales_path) first")

    feat = list(spect_int8)

    for (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw,
         dw, w_off, shift, relu, ofmap_h, ofmap_w, bias_off) in layer_cfgs:

        if layer == 0:
            ifmap_h, ifmap_w = 50, 40
        else:
            prev = layer_cfgs[layer - 1]
            ifmap_h, ifmap_w = prev[13], prev[14]   # ofmap_h, ofmap_w of previous layer

        out = [0] * (out_ch * ofmap_h * ofmap_w)

        for oc in range(out_ch):
            bias = biases[bias_off + oc]
            for oh in range(ofmap_h):
                for ow_idx in range(ofmap_w):
                    acc = bias
                    if dw:
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * sh + kh - ph
                                iw = ow_idx * sw + kw - pw
                                if 0 <= ih < ifmap_h and 0 <= iw < ifmap_w:
                                    fv = (feat[ih * ifmap_w + iw] if layer == 0
                                          else feat[oc * ifmap_h * ifmap_w + ih * ifmap_w + iw])
                                    acc += fv * weights[w_off + oc * kH * kW + kh * kW + kw]
                    else:
                        for ic in range(in_ch):
                            for kh in range(kH):
                                for kw in range(kW):
                                    ih = oh * sh + kh - ph
                                    iw = ow_idx * sw + kw - pw
                                    if 0 <= ih < ifmap_h and 0 <= iw < ifmap_w:
                                        fv = (feat[ih * ifmap_w + iw] if layer == 0
                                              else feat[ic * ifmap_h * ifmap_w + ih * ifmap_w + iw])
                                        widx = (w_off + oc * in_ch * kH * kW
                                                + ic * kH * kW + kh * kW + kw)
                                        acc += fv * weights[widx]

                    shifted = acc >> shift if acc >= 0 else -((-acc) >> shift)
                    sat = max(-128, min(127, shifted))
                    out[oc * ofmap_h * ofmap_w + oh * ofmap_w + ow_idx] = max(0, sat) if relu else sat

        feat = out

    n_classes = layer_cfgs[-1][2]
    spatial   = layer_cfgs[-1][13] * layer_cfgs[-1][14]
    gap = [sum(feat[c * spatial:(c + 1) * spatial]) for c in range(n_classes)]
    return gap.index(max(gap)), gap
