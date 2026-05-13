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

# 16-filter model: 2352 weights total, 151 biases total
LAYER_CFGS_ARCH_16 = [
    #  layer  in  out  kH  kW  sh sw ph pw  dw   w_off  relu  oh  ow bias_off
    (   0,    1,  16, 10,  4,  2, 2, 4, 1,  0,     0,    1,  25, 20,   0),
    (   1,    1,  16,  3,  3,  1, 1, 1, 1,  1,   640,    1,  25, 20,  16),
    (   2,   16,  16,  1,  1,  1, 1, 0, 0,  0,   784,    1,  25, 20,  32),
    (   3,    1,  16,  3,  3,  1, 1, 1, 1,  1,  1040,    1,  25, 20,  48),
    (   4,   16,  16,  1,  1,  1, 1, 0, 0,  0,  1184,    1,  25, 20,  64),
    (   5,    1,  16,  3,  3,  1, 1, 1, 1,  1,  1440,    1,  25, 20,  80),
    (   6,   16,  16,  1,  1,  1, 1, 0, 0,  0,  1584,    1,  25, 20,  96),
    (   7,    1,  16,  3,  3,  1, 1, 1, 1,  1,  1840,    1,  25, 20, 112),
    (   8,   16,  16,  1,  1,  1, 1, 0, 0,  0,  1984,    1,  25, 20, 128),
    (   9,   16,   7,  1,  1,  1, 1, 0, 0,  0,  2240,    0,  25, 20, 144),
]

# 24-filter model (v1–v8): 4296 weights total, 223 biases total
LAYER_CFGS_ARCH_24 = [
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

# 32-filter model (v9+): 6752 weights total, 295 biases total
LAYER_CFGS_ARCH_32 = [
    #  layer  in  out  kH  kW  sh sw ph pw  dw   w_off  relu  oh  ow bias_off
    (   0,    1,  32, 10,  4,  2, 2, 4, 1,  0,     0,    1,  25, 20,   0),
    (   1,    1,  32,  3,  3,  1, 1, 1, 1,  1,  1280,    1,  25, 20,  32),
    (   2,   32,  32,  1,  1,  1, 1, 0, 0,  0,  1568,    1,  25, 20,  64),
    (   3,    1,  32,  3,  3,  1, 1, 1, 1,  1,  2592,    1,  25, 20,  96),
    (   4,   32,  32,  1,  1,  1, 1, 0, 0,  0,  2880,    1,  25, 20, 128),
    (   5,    1,  32,  3,  3,  1, 1, 1, 1,  1,  3904,    1,  25, 20, 160),
    (   6,   32,  32,  1,  1,  1, 1, 0, 0,  0,  4192,    1,  25, 20, 192),
    (   7,    1,  32,  3,  3,  1, 1, 1, 1,  1,  5216,    1,  25, 20, 224),
    (   8,   32,  32,  1,  1,  1, 1, 0, 0,  0,  5504,    1,  25, 20, 256),
    (   9,   32,   7,  1,  1,  1, 1, 0, 0,  0,  6528,    0,  25, 20, 288),
]

# Default (current RTL target); backward-compat alias
LAYER_CFGS_ARCH = LAYER_CFGS_ARCH_32

_ARCH_BY_FILTERS = {16: LAYER_CFGS_ARCH_16, 24: LAYER_CFGS_ARCH_24, 32: LAYER_CFGS_ARCH_32}

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


def load_mults(scales_path: Path) -> list:
    """
    Parse mult values (second-to-last column) from scales.txt.

    Returns a list of 32-bit unsigned ints, one per layer.
    Falls back to (2^32 - 1) per layer if scales.txt uses the old format
    (no mult column), which reproduces the legacy power-of-2 shift behaviour
    to within 1 LSB.
    """
    mults = []
    with open(scales_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("layer") or s.startswith("-"):
                continue
            parts = s.split()
            if len(parts) < 6:          # old format: layer mw_scale in_scale out_scale shift
                return None
            try:
                mults.append(int(parts[-2]))
            except ValueError:
                return None
    return mults


def load_layer_cfgs(scales_path: Path, n_filters: int = 32) -> list:
    """
    Build the full LAYER_CFGS list by merging the architecture table for n_filters
    with per-layer (mult, shift) values from scales.txt.

    Returns tuples in the format expected by rtl_golden_predict():
      (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, w_off,
       mult, shift, relu, ofmap_h, ofmap_w, bias_off)

    mult and shift encode the multiply-shift requantization:
        effective_scale = mult * 2^-(shift + 32)

    If scales.txt was produced by an older export.py (no mult column), mult
    defaults to (2^32 - 1) which reproduces the legacy power-of-2 shift
    behaviour to within 1 LSB.  Re-run export.py to get accurate mult values.

    Args:
        scales_path : path to scales.txt produced by export.py
        n_filters   : number of DS-block filters (16, 24, or 32)
    """
    arch = _ARCH_BY_FILTERS.get(n_filters)
    if arch is None:
        raise ValueError(f"No LAYER_CFGS_ARCH defined for n_filters={n_filters}; "
                         f"supported: {sorted(_ARCH_BY_FILTERS)}")
    shifts = load_shifts(scales_path)
    mults  = load_mults(scales_path)
    if mults is None:
        print("WARNING: scales.txt has no mult column (old export.py format).")
        print("         Using mult=0xFFFFFFFF, which approximates the legacy shift.")
        print("         Re-run export.py to get accurate multiply-shift values.")
        mults = [(1 << 32) - 1] * len(shifts)
    if len(shifts) != len(arch) or len(mults) != len(arch):
        raise ValueError(
            f"scales.txt has {len(shifts)} rows but architecture has {len(arch)} layers"
        )
    cfgs = []
    for a, mult, shift in zip(arch, mults, shifts):
        layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, w_off, relu, oh, ow, bias_off = a
        cfgs.append((layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw, dw, w_off,
                     mult, shift, relu, oh, ow, bias_off))
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
         dw, w_off, mult, shift, relu, ofmap_h, ofmap_w, bias_off) in layer_cfgs:

        if layer == 0:
            ifmap_h, ifmap_w = 50, 40
        else:
            prev = layer_cfgs[layer - 1]
            ifmap_h, ifmap_w = prev[14], prev[15]   # ofmap_h, ofmap_w of previous layer

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

                    # ── Legacy power-of-2 shift (commented out — kept for reference) ──
                    # shifted = acc >> shift if acc >= 0 else -((-acc) >> shift)

                    # ── Multiply-shift (matches new requant.sv) ──────────────────────
                    # Mirrors: product[63:0] = acc * mult; result = product[63:32] >>> shift
                    product  = acc * mult                         # exact Python int
                    raw      = (product >> 32) & 0xFFFFFFFF       # upper 32 bits, unsigned
                    upper32  = raw if raw < (1 << 31) else raw - (1 << 32)  # to signed
                    shifted  = upper32 >> shift                   # arithmetic right shift
                    sat = max(-128, min(127, shifted))
                    out[oc * ofmap_h * ofmap_w + oh * ofmap_w + ow_idx] = max(0, sat) if relu else sat

        feat = out

    n_classes = layer_cfgs[-1][2]
    spatial   = layer_cfgs[-1][14] * layer_cfgs[-1][15]
    gap = [sum(feat[c * spatial:(c + 1) * spatial]) for c in range(n_classes)]
    return gap.index(max(gap)), gap
