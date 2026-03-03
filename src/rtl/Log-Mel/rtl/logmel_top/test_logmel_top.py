import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, ReadOnly
import numpy as np

# Parameters matching logmel_top defaults
IW = 18
SHIFT = 6
N_MELS = 40
N_BINS = 129
MAX_COEFFS = 16
POWER_W = 31
WEIGHT_W = 16
ACCUM_W = 54
LOG_OUT_W = 16
OUT_W = 16
LUT_FRAC = 6
Q_FRAC = 12

CLK_PERIOD_NS = 10
MASK_IW = (1 << IW) - 1
MASK_POWER = (1 << POWER_W) - 1
MASK_ACCUM = (1 << ACCUM_W) - 1
MASK_OUT = (1 << OUT_W) - 1
MASK_LUT = (1 << LUT_FRAC) - 1


# ---------- Reference model helpers ----------

def load_hex(path):
    vals = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(int(s, 16))
    return vals


def to_signed(val, bits):
    if val >= (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def ref_power(re_list, im_list):
    """Match power_calc: (re^2 + im^2) >> SHIFT."""
    out = []
    for r, i in zip(re_list, im_list):
        rs = to_signed(r & MASK_IW, IW)
        is_ = to_signed(i & MASK_IW, IW)
        total = rs * rs + is_ * is_
        out.append((total >> SHIFT) & MASK_POWER)
    return out


def ref_filterbank(powers, coeffs, starts, ends):
    """Match mel_filterbank: accumulate power*weight for each mel bin."""
    accum = [0] * N_MELS
    for b in range(N_BINS):
        for m in range(N_MELS):
            if starts[m] <= b <= ends[m]:
                ci = b - starts[m]
                if ci < MAX_COEFFS:
                    w = coeffs[m * MAX_COEFFS + ci]
                    accum[m] += powers[b] * w
    return [a & MASK_ACCUM for a in accum]


def ref_log(energies, lut):
    """Match log_lut: floor(log2) integer + LUT fractional, Q4.12."""
    out = []
    for e in energies:
        if e == 0:
            out.append(0)
            continue
        log2_int = int(e).bit_length() - 1
        if log2_int >= LUT_FRAC:
            addr = (e >> (log2_int - LUT_FRAC)) & MASK_LUT
        else:
            addr = (e << (LUT_FRAC - log2_int)) & MASK_LUT
        val = lut[addr]
        result = (log2_int << Q_FRAC) + val
        out.append(result & MASK_OUT)
    return out


# ---------- DUT helpers ----------

async def reset_dut(dut, cycles=5):
    await RisingEdge(dut.clk)
    dut.reset.value = 1
    dut.fft_valid_il.value = 0
    dut.fft_sync_il.value = 0
    dut.re_il.value = 0
    dut.im_il.value = 0
    dut.cnn_ready_il.value = 1
    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


async def drive_frame(dut, re_vals, im_vals):
    """Drive one frame of 129 FFT bins with sync on the first bin."""
    for i in range(N_BINS):
        dut.re_il.value = int(re_vals[i]) & MASK_IW
        dut.im_il.value = int(im_vals[i]) & MASK_IW
        dut.fft_valid_il.value = 1
        dut.fft_sync_il.value = 1 if i == 0 else 0
        await RisingEdge(dut.clk)
    dut.fft_valid_il.value = 0
    dut.fft_sync_il.value = 0
    dut.re_il.value = 0
    dut.im_il.value = 0


async def collect_output(dut, timeout=500):
    """Collect N_MELS values from CNN output interface."""
    vals = []
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.cnn_valid_ol.value == 1:
            vals.append(int(dut.cnn_data_ol.value))
            if len(vals) == N_MELS:
                return vals
    raise AssertionError(f"Timeout: only got {len(vals)}/{N_MELS} outputs")


# ---------- Tests ----------

@cocotb.test()
async def test_logmel_flat(dut):
    """Flat input: constant re/im across all 129 bins."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    coeffs = load_hex("mel_coeffs.hex")
    starts = load_hex("mel_starts.hex")
    ends = load_hex("mel_ends.hex")
    lut = load_hex("log2_lut.hex")

    await reset_dut(dut)

    re_vals = [1000] * N_BINS
    im_vals = [500] * N_BINS

    exp_pow = ref_power(re_vals, im_vals)
    exp_mel = ref_filterbank(exp_pow, coeffs, starts, ends)
    exp_log = ref_log(exp_mel, lut)

    await drive_frame(dut, re_vals, im_vals)
    got = await collect_output(dut)

    for m in range(N_MELS):
        cocotb.log.info(f"  mel[{m:2d}] got={got[m]:5d} exp={exp_log[m]:5d}")

    deltas = [abs(got[m] - exp_log[m]) for m in range(N_MELS)]
    worst = max(range(N_MELS), key=lambda i: deltas[i])
    cocotb.log.info(f"Worst delta: mel[{worst}] = {deltas[worst]}")
    assert all(d <= 1 for d in deltas), f"FAIL at mel[{worst}] delta={deltas[worst]}"


@cocotb.test()
async def test_logmel_random(dut):
    """Random signed FFT input."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    coeffs = load_hex("mel_coeffs.hex")
    starts = load_hex("mel_starts.hex")
    ends = load_hex("mel_ends.hex")
    lut = load_hex("log2_lut.hex")

    await reset_dut(dut)

    rng = np.random.default_rng(seed=42)
    half = 1 << (IW - 1)
    re_signed = rng.integers(-half, half, size=N_BINS)
    im_signed = rng.integers(-half, half, size=N_BINS)
    re_vals = [int(x) & MASK_IW for x in re_signed]
    im_vals = [int(x) & MASK_IW for x in im_signed]

    exp_pow = ref_power(re_vals, im_vals)
    exp_mel = ref_filterbank(exp_pow, coeffs, starts, ends)
    exp_log = ref_log(exp_mel, lut)

    await drive_frame(dut, re_vals, im_vals)
    got = await collect_output(dut)

    deltas = [abs(got[m] - exp_log[m]) for m in range(N_MELS)]
    worst = max(range(N_MELS), key=lambda i: deltas[i])
    cocotb.log.info(f"Worst delta: mel[{worst}] = {deltas[worst]}")
    assert all(d <= 1 for d in deltas), f"FAIL at mel[{worst}] delta={deltas[worst]}"

    cocotb.log.info("ALL TESTS PASSED")
