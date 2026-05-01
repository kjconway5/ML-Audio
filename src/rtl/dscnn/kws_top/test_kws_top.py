# test_kws_top.py
# End-to-end cocotb testbench for kws_top.sv


import json
import os
import sys
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, First, Timer

# Import shared RTL arithmetic golden model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtl_golden import rtl_golden_predict, load_layer_cfgs, load_hex_file

CLK_PERIOD_NS = 100

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

def _load_class_names(test_dir):
    path = os.path.join(test_dir, "class_names.txt")
    if os.path.exists(path):
        with open(path) as f:
            names = [l.strip() for l in f.readlines() if l.strip()]
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


async def program_layers(dut, layer_cfgs):
    dut._log.info("Programming layer configs...")
    for (layer, in_ch, out_ch, kH, kW, sh, sw, ph, pw,
         dw, w_off, mult, shift, relu, ofmap_h, ofmap_w, bias_off) in layer_cfgs:

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
        await write_cfg(dut, base | 0xB, shift)                   # exponent n
        # bit[0] = relu, bit[1] = bias_off[8] (high bit for 32-filter model where bias_off ≥ 256)
        await write_cfg(dut, base | 0xC, (relu & 0x1) | ((bias_off >> 7) & 0x2))
        await write_cfg(dut, base | 0xD, ofmap_h)
        await write_cfg(dut, base | 0xE, ofmap_w)
        await write_cfg(dut, base | 0xF, bias_off & 0xFF)

        # Multiply-shift multiplier: 32-bit mult sent as 4 bytes to addresses 0xA0+(layer*4)+byte
        # Address map: 0xA0=L0B0 ... 0xC7=L9B3  (see FSM.sv config decoder)
        mult_base = 0xA0 + layer * 4
        await write_cfg(dut, mult_base + 0, (mult >>  0) & 0xFF)  # mult[7:0]
        await write_cfg(dut, mult_base + 1, (mult >>  8) & 0xFF)  # mult[15:8]
        await write_cfg(dut, mult_base + 2, (mult >> 16) & 0xFF)  # mult[23:16]
        await write_cfg(dut, mult_base + 3, (mult >> 24) & 0xFF)  # mult[31:24]

    await write_cfg(dut, 0xFF, 1)
    dut._log.info("Layer configs programmed, cfg_load_done set.")



async def run_single_inference(dut, spect_int8, sample_idx, timeout_ns, layer_cfgs):
    """Load spectrogram, run one inference, return RTL class_out."""
    await load_spectrogram(dut, spect_int8)
    await signal_spect_done(dut)
    await program_layers(dut, layer_cfgs)

    await FallingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.start.value = 0

    result = await First(RisingEdge(dut.done), Timer(timeout_ns, units="ns"))

    if isinstance(result, Timer):
        try:
            state = FSM_STATES.get(int(dut.inst_ctrl.state.value), "?")
            layer = int(dut.inst_ctrl.layer.value)
            oh    = int(dut.inst_ctrl.oh.value)
            ow    = int(dut.inst_ctrl.ow.value)
            oc    = int(dut.inst_ctrl.oc.value)
            dut._log.error(
                f"[sample {sample_idx}] Timeout! Last FSM state: {state}  "
                f"L{layer}={LAYER_NAMES[layer] if layer < len(LAYER_NAMES) else '?'}  "
                f"oh={oh} ow={ow} oc={oc}"
            )
        except Exception:
            pass
        raise AssertionError(
            f"[sample {sample_idx}] Timeout: done not asserted within "
            f"{timeout_ns // CLK_PERIOD_NS:,} cycles"
        )

    return int(dut.class_out.value)


@cocotb.test()
async def test_kws_inference(dut):
    test_dir = get_test_dir()

    MODEL_DIR     = os.path.join(test_dir, "..", "..", "..", "ml", "models", "dscnn-32full-v1")
    weights_path  = os.path.join(MODEL_DIR, "weights.hex")
    bias_path     = os.path.join(MODEL_DIR, "bias.hex")
    scales_path   = os.path.join(MODEL_DIR, "scales.txt")
    manifest_path = os.path.join(test_dir, "spectrograms", "test_vectors.json")

    for p in [weights_path, bias_path, scales_path, manifest_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing file: {p}\n"
                f"Run: python3 src/ml/Pipeline/export.py\n"
                f"     python3 src/ml/Pipeline/update_rtl.py\n"
                f"     cd src/rtl/dscnn/kws_top && python3 generate_spect.py --keyword <kw>"
            )

    from pathlib import Path
    layer_cfgs = load_layer_cfgs(Path(scales_path), n_filters=32)

    with open(manifest_path) as f:
        manifest = json.load(f)

    weights    = load_hex_file(weights_path, signed=True, width=8)
    biases_raw = load_hex_file(bias_path,   signed=True, width=32)
    assert len(weights) == 6752, f"Expected 6752 weights, got {len(weights)}"

    keyword     = manifest["keyword"]
    class_names = manifest["class_names"]
    samples     = manifest["samples"]

    dut._log.info("=" * 72)
    dut._log.info(f"GOLDEN ARITHMETIC PREVIEW — keyword='{keyword}'  ({len(samples)} samples)")
    dut._log.info("=" * 72)

    sample_data = []  # pre-computed data reused in Phase 2
    for s in samples:
        hex_path = os.path.join(test_dir, s["hex_file"])
        spect_int8 = load_hex_file(hex_path, signed=True, width=8)
        assert len(spect_int8) == 2000, \
            f"Sample {s['index']}: expected 2000 values, got {len(spect_int8)}"

        arith_class, arith_gap = rtl_golden_predict(spect_int8, weights, biases_raw, layer_cfgs)
        arith_name = class_names[arith_class] if arith_class < len(class_names) else f"cls{arith_class}"

        sorted_gap = sorted(enumerate(arith_gap), key=lambda x: x[1], reverse=True)
        margin     = sorted_gap[0][1] - sorted_gap[1][1]
        gap_str    = "  ".join(f"{class_names[c]}:{v}" for c, v in sorted_gap)
        match      = "OK" if arith_name == keyword else "MISS"

        dut._log.info(
            f"[{s['index']}] {s['wav'].split('/')[-1]:<30}  "
            f"arith={arith_name:<8}  gt={keyword:<8}  {match}  margin={margin}"
        )
        dut._log.info(f"    GAP: {gap_str}")

        sample_data.append({
            "spect_int8":   spect_int8,
            "arith_class":  arith_class,
            "arith_name":   arith_name,
            "gt_class":     s["ground_truth_class"],
            "gt_name":      keyword,
            "gap_str":      gap_str,
            "margin":       margin,
        })

    dut._log.info("=" * 72)
    dut._log.info("STARTING RTL SIMULATION")
    dut._log.info("=" * 72)

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    await reset_dut(dut)
    await load_weight_sram(dut, weights)

    TIMEOUT_NS = 12_000_000 * CLK_PERIOD_NS
    results = []

    for i, sd in enumerate(sample_data):
        dut._log.info(f"--- RTL inference {i+1}/{len(sample_data)}: {samples[i]['wav'].split('/')[-1]} ---")

        # Reset FSM state between inferences (weights stay loaded)
        if i > 0:
            await reset_dut(dut)

        cocotb.start_soon(monitor_fsm_progress(dut, log_every_cycles=50_000))

        rtl_class = await run_single_inference(dut, sd["spect_int8"], i, TIMEOUT_NS, layer_cfgs)
        rtl_name  = class_names[rtl_class] if rtl_class < len(class_names) else f"cls{rtl_class}"

        passed = (rtl_class == sd["gt_class"])
        status = "PASS" if passed else "FAIL"
        consistent = (rtl_class == sd["arith_class"])

        dut._log.info(
            f"[{i}] RTL={rtl_name:<8}  GT={sd['gt_name']:<8}  "
            f"arith={sd['arith_name']:<8}  margin={sd['margin']}  {status}"
        )
        if not passed:
            if consistent:
                dut._log.warning(
                    f"[{i}] RTL matches arithmetic golden ({rtl_name}) — model quality issue"
                )
            else:
                dut._log.error(
                    f"[{i}] RTL ({rtl_name}) disagrees with arithmetic golden "
                    f"({sd['arith_name']}) — possible RTL bug"
                )

        results.append({"passed": passed, "rtl_name": rtl_name, "consistent": consistent})

    n_pass = sum(1 for r in results if r["passed"])
    dut._log.info("=" * 72)
    dut._log.info(f"FINAL RESULT: {n_pass}/{len(results)} samples correctly classified as '{keyword}'")
    for i, r in enumerate(results):
        mark = "PASS" if r["passed"] else "FAIL"
        dut._log.info(f"  [{i}] {mark}  RTL={r['rtl_name']}")
    dut._log.info("=" * 72)

    results_path = os.path.join(test_dir, "kws_results.txt")
    with open(results_path, "w") as f:
        f.write(f"keyword: {keyword}   {n_pass}/{len(results)} passed\n")
        f.write("-" * 40 + "\n")
        for i, (r, s) in enumerate(zip(results, samples)):
            mark = "PASS" if r["passed"] else "FAIL"
            wav  = s["wav"].split("/")[-1]
            note = ""
            if not r["passed"]:
                note = "  [RTL bug]" if not r["consistent"] else "  [model miss]"
            f.write(f"[{i}] {mark}  {wav:<30}  RTL={r['rtl_name']}{note}\n")

    if n_pass < len(results):
        raise AssertionError(
            f"{len(results) - n_pass}/{len(results)} sample(s) misclassified. "
            f"See per-sample logs above."
        )
