# test_kws_top.py
# End-to-end cocotb testbench for kws_top.sv


import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, First, Timer

CLK_PERIOD_NS = 100

# ── FSM debug maps ────────────────────────────────────────────────────────────
FSM_STATES = {
    0:  "IDLE",
    1:  "LOAD_LAYER",
    2:  "CLEAR_ACC",
    3:  "FETCH",
    4:  "COMPUTE",
    5:  "DRAIN",
    6:  "WRITE_OFMAP",
    7:  "NEXT_PIXEL",
    8:  "NEXT_LAYER",
    9:  "GLOBAL_POOL",
    10: "OUTPUT",
}

LAYER_NAMES = [
    "first_conv      ",   
    "ds0.depthwise   ",   
    "ds0.pointwise   ",   
    "ds1.depthwise   ",   
    "ds1.pointwise   ",   
    "ds2.depthwise   ",   
    "ds2.pointwise   ",   
    "ds3.depthwise   ",   
    "ds3.pointwise   ",   
    "classifier      ",   
]


async def monitor_fsm_progress(dut, log_every_cycles=50_000):
    elapsed = 0
    while True:
        await ClockCycles(dut.clk, log_every_cycles)
        elapsed += log_every_cycles
        try:
            state = int(dut.inst_ctrl.state.value)
            layer = int(dut.inst_ctrl.layer.value)
            oh    = int(dut.inst_ctrl.oh.value)
            ow    = int(dut.inst_ctrl.ow.value)
            oc    = int(dut.inst_ctrl.oc.value)
            kh    = int(dut.inst_ctrl.kh.value)
            kw    = int(dut.inst_ctrl.kw.value)
            ic    = int(dut.inst_ctrl.ic.value)
        except Exception as e:
            dut._log.warning(f"[monitor] could not read FSM signals: {e}")
            continue

        state_name = FSM_STATES.get(state, f"?{state}")
        layer_name = LAYER_NAMES[layer] if layer < len(LAYER_NAMES) else f"L{layer}"
        dut._log.info(
            f"[+{elapsed:>8,} cyc]  state={state_name:<12}  "
            f"L{layer}={layer_name}  "
            f"oh={oh:3d} ow={ow:3d} oc={oc:3d}  "
            f"kh={kh:2d} kw={kw:2d} ic={ic:3d}"
        )

LAYER_CFGS = [
    (  0,     1,  24, 10,  4,  2, 2, 4, 1, 0,     0,   10,   1, 25, 20,   0),
    (  1,     1,  24,  3,  3,  1, 1, 1, 1, 1,   960,    5,   1, 25, 20,  24),
    (  2,    24,  24,  1,  1,  1, 1, 0, 0, 0,  1176,    7,   1, 25, 20,  48),
    (  3,     1,  24,  3,  3,  1, 1, 1, 1, 1,  1752,    5,   1, 25, 20,  72),
    (  4,    24,  24,  1,  1,  1, 1, 0, 0, 0,  1968,    7,   1, 25, 20,  96),
    (  5,     1,  24,  3,  3,  1, 1, 1, 1, 1,  2544,    7,   1, 25, 20, 120),
    (  6,    24,  24,  1,  1,  1, 1, 0, 0, 0,  2760,    7,   1, 25, 20, 144),
    (  7,     1,  24,  3,  3,  1, 1, 1, 1, 1,  3336,    7,   1, 25, 20, 168),
    (  8,    24,  24,  1,  1,  1, 1, 0, 0, 0,  3552,    6,   1, 25, 20, 192),
    (  9,    24,   7,  1,  1,  1, 1, 0, 0, 0,  4128,    4,   0, 25, 20, 216),
]

def rtl_golden_predict(spect_int8, weights, biases):
    feat = list(spect_int8)   

    for (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw,
         dw, w_off, shift, relu, ofmap_h, ofmap_w, bias_off) in LAYER_CFGS:

        if layer == 0:
            ifmap_h, ifmap_w = 50, 40
        else:
            _, _, _, _, _, _, _, _, _, _, _, _, _, prev_h, prev_w, _ = LAYER_CFGS[layer - 1]
            ifmap_h, ifmap_w = prev_h, prev_w

        out = [0] * (out_ch * ofmap_h * ofmap_w)

        for oc in range(out_ch):
            bias = biases[bias_off + oc]
            for oh in range(ofmap_h):
                for ow in range(ofmap_w):
                    acc = bias
                    if dw:
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * sh + kh - ph
                                iw = ow * sw + kw - pw
                                if 0 <= ih < ifmap_h and 0 <= iw < ifmap_w:
                                    if layer == 0:
                                        fv = feat[ih * ifmap_w + iw]
                                    else:
                                        fv = feat[oc * ifmap_h * ifmap_w + ih * ifmap_w + iw]
                                    acc += fv * weights[w_off + oc * kH * kW + kh * kW + kw]
                    else:
                        for ic in range(in_ch):
                            for kh in range(kH):
                                for kw in range(kW):
                                    ih = oh * sh + kh - ph
                                    iw = ow * sw + kw - pw
                                    if 0 <= ih < ifmap_h and 0 <= iw < ifmap_w:
                                        if layer == 0:
                                            fv = feat[ih * ifmap_w + iw]
                                        else:
                                            fv = feat[ic * ifmap_h * ifmap_w + ih * ifmap_w + iw]
                                        widx = (w_off + oc * in_ch * kH * kW
                                                + ic * kH * kW + kh * kW + kw)
                                        acc += fv * weights[widx]

                    shifted = acc >> shift if acc >= 0 else -((-acc) >> shift)
                    sat = max(-128, min(127, shifted))
                    out[oc * ofmap_h * ofmap_w + oh * ofmap_w + ow] = max(0, sat) if relu else sat

        feat = out

    n_classes = LAYER_CFGS[-1][2]   # out_ch of last layer = 7
    spatial   = LAYER_CFGS[-1][13] * LAYER_CFGS[-1][14]  # ofmap_h * ofmap_w = 500
    gap = [sum(feat[c * spatial:(c + 1) * spatial]) for c in range(n_classes)]
    return gap.index(max(gap)), gap

def _load_class_names(test_dir):
    path = os.path.join(test_dir, "class_names.txt")
    if os.path.exists(path):
        names = [l.strip() for l in open(path).readlines() if l.strip()]
        if names:
            return names
    return ["no", "off", "on", "silence", "unknown", "wow", "yes"]

CLASS_NAMES = _load_class_names(os.path.dirname(os.path.abspath(__file__)))


def load_hex_file(path, signed=False, width=8):
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


def get_test_dir():
    return os.path.dirname(os.path.abspath(__file__))



async def reset_dut(dut):
    dut.reset.value        = 1
    dut.start.value        = 0
    dut.cfg_we.value       = 0
    dut.cfg_addr.value     = 0
    dut.cfg_wdata.value    = 0
    dut.spect_done.value   = 0
    dut.spect_write_sel.value = 0
    dut.sp_a_we.value      = 0
    dut.sp_a_waddr.value   = 0
    dut.sp_a_wdata.value   = 0
    dut.sp_b_we.value      = 0
    dut.sp_b_waddr.value   = 0
    dut.sp_b_wdata.value   = 0
    dut.w_we.value         = 0
    dut.w_waddr.value      = 0
    dut.w_wdata.value      = 0

    await ClockCycles(dut.clk, 10)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 5)


async def load_weight_sram(dut, weights):
    dut._log.info(f"Loading weight SRAM ({len(weights)} values)...")
    for addr, val in enumerate(weights):
        await FallingEdge(dut.clk)
        dut.w_we.value    = 1
        dut.w_waddr.value = addr
        dut.w_wdata.value = val & 0xFF
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.w_we.value = 0
    await RisingEdge(dut.clk)
    dut._log.info("Weight SRAM loaded.")


async def load_spectrogram(dut, spect_int8):
    dut._log.info(f"Loading spectrogram SRAM bank A ({len(spect_int8)} values)...")
    for addr, val in enumerate(spect_int8):
        await FallingEdge(dut.clk)
        dut.sp_a_we.value    = 1
        dut.sp_a_waddr.value = addr
        dut.sp_a_wdata.value = val & 0xFF
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.sp_a_we.value = 0
    await RisingEdge(dut.clk)
    dut._log.info("Spectrogram SRAM loaded.")


async def signal_spect_done(dut):
    await FallingEdge(dut.clk)
    dut.spect_done.value      = 1
    dut.spect_write_sel.value = 0   # bank A
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.spect_done.value = 0
    await RisingEdge(dut.clk)


async def write_cfg(dut, addr, data):
    await FallingEdge(dut.clk)
    dut.cfg_we.value    = 1
    dut.cfg_addr.value  = addr
    dut.cfg_wdata.value = data & 0xFF
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.cfg_we.value = 0
    await RisingEdge(dut.clk)


async def program_layers(dut):
    dut._log.info("Programming layer configs...")
    for (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw,
         dw, w_off, shift, relu, ofmap_h, ofmap_w, bias_off) in LAYER_CFGS:

        base = layer << 4
        await write_cfg(dut, base | 0x0, in_ch)
        await write_cfg(dut, base | 0x1, out_ch)
        await write_cfg(dut, base | 0x2, kH)
        await write_cfg(dut, base | 0x3, kW)
        await write_cfg(dut, base | 0x4, sh)
        await write_cfg(dut, base | 0x5, sw)
        await write_cfg(dut, base | 0x6, ph)
        await write_cfg(dut, base | 0x7, pw)
        await write_cfg(dut, base | 0x8, dw)
        await write_cfg(dut, base | 0x9, w_off & 0xFF)           # w_off[7:0]
        await write_cfg(dut, base | 0xA, (w_off >> 8) & 0x1F)    # w_off[12:8]
        await write_cfg(dut, base | 0xB, shift)
        await write_cfg(dut, base | 0xC, relu)
        await write_cfg(dut, base | 0xD, ofmap_h)
        await write_cfg(dut, base | 0xE, ofmap_w)
        await write_cfg(dut, base | 0xF, bias_off)

    await write_cfg(dut, 0xFF, 1)
    dut._log.info("Layer configs programmed, cfg_load_done set.")



@cocotb.test()
async def test_kws_inference(dut):
    test_dir = get_test_dir()

    weights_path = os.path.join(test_dir, "..", "..", "..", "ml", "models", "dscnn", "weights.hex")
    bias_path    = os.path.join(test_dir, "..", "..", "..", "ml", "models", "dscnn", "bias.hex")
    spect_path   = os.path.join(test_dir, "spectrogram.hex")

    for p in [weights_path, bias_path, spect_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing test vector: {p}\n"
                f"Run: cd src/ml/models/dscnn && python3 export.py\n"
                f"     cd src/rtl/dscnn/kws_top && make gen-spect"
            )

    weights        = load_hex_file(weights_path, signed=True,  width=8)
    biases_raw     = load_hex_file(bias_path,    signed=True,  width=32)
    spect_int8     = load_hex_file(spect_path,   signed=True,  width=8)

    assert len(weights)    == 4296, f"Expected 4296 weights, got {len(weights)}"
    assert len(spect_int8) == 2000, f"Expected 2000 spectrogram values, got {len(spect_int8)}"

    # Ground truth: the label of the WAV file we actually fed in.
    # This is the real test — did the chip correctly identify the spoken word?
    gt_path = os.path.join(test_dir, "ground_truth_class.txt")
    gn_path = os.path.join(test_dir, "ground_truth_name.txt")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            "Missing ground_truth_class.txt — re-run generate_spect.py to regenerate test vectors"
        )
    expected_class = int(open(gt_path).read().strip())
    expected_name  = open(gn_path).read().strip() if os.path.exists(gn_path) else \
                     (CLASS_NAMES[expected_class] if expected_class < len(CLASS_NAMES) else f"class{expected_class}")

    # RTL arithmetic golden — logged for debug only, not used for pass/fail.
    # If this disagrees with ground truth, the power-of-2 quantization is lossy
    # on this sample (model quality issue, not RTL correctness issue).
    rtl_arith_class, rtl_arith_gap = rtl_golden_predict(spect_int8, weights, biases_raw)
    rtl_arith_name  = CLASS_NAMES[rtl_arith_class] if rtl_arith_class < len(CLASS_NAMES) else f"class{rtl_arith_class}"

    sorted_gap  = sorted(enumerate(rtl_arith_gap), key=lambda x: x[1], reverse=True)
    gap_margin  = sorted_gap[0][1] - sorted_gap[1][1]
    gap_summary = "  ".join(f"{CLASS_NAMES[c]}:{v}" for c, v in sorted_gap)

    dut._log.info(f"Ground truth   : {expected_class} ({expected_name})")
    dut._log.info(f"RTL arithmetic : {rtl_arith_class} ({rtl_arith_name})" +
                  ("" if rtl_arith_class == expected_class else "  ← quantization differs from ground truth"))
    dut._log.info(f"GAP scores     : {gap_summary}")
    dut._log.info(f"Margin (1st-2nd): {gap_margin}")
    dut._log.info(f"Spectrogram    : {len(spect_int8)} INT8 values, "
                  f"range [{min(spect_int8)}, {max(spect_int8)}]")

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())

    # reset 
    await reset_dut(dut)
    # load weight SRAM
    await load_weight_sram(dut, weights)
    # load spect bank A 
    await load_spectrogram(dut, spect_int8)

    # ── Phase 4: Signal spectrogram ready ────────────────────────────────────
    await signal_spect_done(dut)

    # ── Phase 5: Program layer configs ───────────────────────────────────────
    await program_layers(dut)

    # ── Phase 6: Assert start ────────────────────────────────────────────────
    await FallingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.start.value = 0
    dut._log.info("Inference started — FSM monitor will log every 50K cycles")

    # ── Phase 7: Launch FSM progress monitor ─────────────────────────────────
    # Logs state/layer/oh/ow/oc every 50K cycles so we can see exact progress.
    # Cocotb kills this coroutine automatically when the test ends.
    cocotb.start_soon(monitor_fsm_progress(dut, log_every_cycles=50_000))

    # ── Phase 8: Wait for done ────────────────────────────────────────────────
    # Sleep until done rises (or timeout). No per-cycle Python wake-ups.
    TIMEOUT_NS = 12_000_000 * CLK_PERIOD_NS
    result = await First(RisingEdge(dut.done), Timer(TIMEOUT_NS, units="ns"))

    if isinstance(result, Timer):
        # Log final FSM state before raising so the user can see where it stopped
        try:
            state = FSM_STATES.get(int(dut.inst_ctrl.state.value), "?")
            layer = int(dut.inst_ctrl.layer.value)
            oh    = int(dut.inst_ctrl.oh.value)
            ow    = int(dut.inst_ctrl.ow.value)
            oc    = int(dut.inst_ctrl.oc.value)
            dut._log.error(
                f"Timeout! Last FSM state: {state}  "
                f"L{layer}={LAYER_NAMES[layer] if layer < len(LAYER_NAMES) else '?'}  "
                f"oh={oh} ow={ow} oc={oc}"
            )
        except Exception:
            pass
        raise AssertionError(
            f"Timeout: done not asserted within {TIMEOUT_NS // CLK_PERIOD_NS:,} cycles"
        )

    dut._log.info("Done pulse received")

    # ── Phase 9: Check class_out ──────────────────────────────────────────────
    # done and class_out are set in the same OUTPUT state cycle
    rtl_class = int(dut.class_out.value)
    rtl_name  = CLASS_NAMES[rtl_class] if rtl_class < len(CLASS_NAMES) else f"unknown({rtl_class})"

    dut._log.info(f"RTL  class_out : {rtl_class} ({rtl_name})")
    dut._log.info(f"Ground truth   : {expected_class} ({expected_name})")
    dut._log.info(f"RTL arithmetic : {rtl_arith_class} ({rtl_arith_name})")
    dut._log.info(f"GAP scores     : {gap_summary}")
    dut._log.info(f"Margin (1st-2nd): {gap_margin}")

    if rtl_class == expected_class:
        dut._log.info("PASS: RTL correctly identified the spoken keyword")
    else:
        # Distinguish between RTL bug and quantization/model quality issue
        if rtl_class == rtl_arith_class:
            raise AssertionError(
                f"FAIL: RTL predicted '{rtl_name}' but ground truth is '{expected_name}'. "
                f"RTL arithmetic is internally consistent — this is a model quality issue "
                f"(power-of-2 quantization degraded accuracy on this sample)."
            )
        else:
            raise AssertionError(
                f"FAIL: RTL predicted '{rtl_name}' but ground truth is '{expected_name}'. "
                f"RTL arithmetic golden predicts '{rtl_arith_name}' — likely an RTL bug."
            )
