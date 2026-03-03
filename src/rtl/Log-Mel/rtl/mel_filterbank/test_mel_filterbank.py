import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, ReadOnly
import numpy as np
import torchaudio.transforms as T

# Parameters
N_MELS, N_BINS, MAX_COEFFS = 40, 129, 16
POWER_W, WEIGHT_W, ACCUM_W = 31, 16, 54
SAMPLE_RATE, N_FFT = 16000, 256
F_MIN, F_MAX = 0.0, SAMPLE_RATE / 2.0

POWER_MAX  = (1 << POWER_W) - 1
WEIGHT_MAX = (1 << WEIGHT_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1
CLK_PERIOD_NS = 10 
TOLERANCE = 2**26

class My_MelFilterbank:
    def __init__(self):
        mel_t = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0)
        fb_fixed = np.round(mel_t.mel_scale.fb.numpy() * (2 ** 15)).astype(np.int64)
        self.fb_fixed = np.clip(fb_fixed, 0, WEIGHT_MAX)

    def compute(self, power_bins: np.ndarray) -> np.ndarray:
        p = (power_bins.astype(np.int64) & POWER_MAX)
        accum = p @ self.fb_fixed
        return (accum & ACCUM_MASK).astype(np.uint64)

async def reset_dut(dut, cycles: int = 5):
    await RisingEdge(dut.clk_i)  # sync to clock edge before driving reset
    dut.reset_i.value  = 1
    dut.valid_il.value = 0
    await ClockCycles(dut.clk_i, cycles)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 2)

async def drive_frame(dut, power_bins: np.ndarray):
    for p in power_bins:
        dut.power_il.value = int(p) & POWER_MAX
        dut.valid_il.value = 1
        await RisingEdge(dut.clk_i)
    dut.valid_il.value = 0
    dut.power_il.value = 0

async def wait_for_valid_ol(dut, timeout: int = N_MELS * (MAX_COEFFS + 2) + 50) -> int:
    #Wait for valid_ol to assert. Returns after the RisingEdge where valid_ol is seen
    for i in range(timeout):
        await RisingEdge(dut.clk_i)
        if dut.valid_ol.value == 1:
            return i
    raise AssertionError(f"Timeout: valid_ol did not assert within {timeout} cycles")

def snapshot_outputs(dut) -> np.ndarray:
    raw = int(dut.mel_ol.value)
    mask = (1 << ACCUM_W) - 1
    return np.array([(raw >> (m * ACCUM_W)) & mask for m in range(N_MELS)], dtype=np.uint64)

@cocotb.test()
async def test_mel_filterbank(dut):
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, unit="ns").start())
    ref = My_MelFilterbank()

    # Test 1 : Flat
    cocotb.log.info("### TEST 1: Flat ###")
    await reset_dut(dut)
    pb = np.full(N_BINS, 1 << 12, dtype=np.uint64)
    await drive_frame(dut, pb)
    await wait_for_valid_ol(dut)
    got = snapshot_outputs(dut)
    exp = ref.compute(pb)
    deltas = np.abs(got.astype(np.int64) - exp.astype(np.int64))
    worst = np.argmax(deltas)
    cocotb.log.info(f"Test 1 worst: mel[{worst}] got={got[worst]} exp={exp[worst]} delta={deltas[worst]}")
    cocotb.log.info(f"Test 1 first 5 got: {got[:5]}")
    cocotb.log.info(f"Test 1 first 5 exp: {exp[:5]}")
    cocotb.log.info(f"Test 1 nonzero got: {np.count_nonzero(got)}, nonzero exp: {np.count_nonzero(exp)}")
    assert np.all(deltas <= TOLERANCE), f"Test 1 FAIL: max delta={deltas[worst]} at mel[{worst}]"
    await ClockCycles(dut.clk_i, 1)

    # --- Test 2: Reset + Random ---
    cocotb.log.info("### TEST 2: Random & Reset")
    cocotb.log.info(f"  before RisingEdge, sim_time={cocotb.utils.get_sim_time('ns')}ns")
    await RisingEdge(dut.clk_i)
    cocotb.log.info(f"  after RisingEdge, sim_time={cocotb.utils.get_sim_time('ns')}ns")
    cocotb.log.info(f"  setting reset_i=1")
    dut.reset_i.value = 1
    dut.valid_il.value = 0
    cocotb.log.info(f"  before ClockCycles(1)")
    await ClockCycles(dut.clk_i, 1)
    cocotb.log.info(f"  after ClockCycles(1), sim_time={cocotb.utils.get_sim_time('ns')}ns")
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 2)
    cocotb.log.info(f"  reset complete, sim_time={cocotb.utils.get_sim_time('ns')}ns")

    rng = np.random.default_rng(seed=42)
    pb = rng.integers(0, 1 << 16, size=N_BINS, dtype=np.uint64)
    await drive_frame(dut, pb)
    await wait_for_valid_ol(dut)
    assert np.all(np.abs(snapshot_outputs(dut).astype(np.int64) - ref.compute(pb).astype(np.int64)) <= TOLERANCE)

    cocotb.log.info("ALL TESTS PASSED")