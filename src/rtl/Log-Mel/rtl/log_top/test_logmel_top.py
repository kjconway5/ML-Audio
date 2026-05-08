import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import torchaudio.transforms as T
from pathlib import Path

IW            = 18
SHIFT         = 6
N_MELS        = 40
N_BINS        = 129
POWER_W       = 31
WEIGHT_W      = 16
ACCUM_W       = 54
LOG_OUT_W     = 16
LUT_FRAC      = 6
Q_FRAC        = 10
CLK_PERIOD_NS = 10

SAMPLE_RATE = 16000
N_FFT       = 256
WIN_LENGTH  = 256
F_MIN       = 0.0
F_MAX       = SAMPLE_RATE / 2.0

IW_MASK    = (1 << IW)      - 1
POWER_MASK = (1 << POWER_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1
WEIGHT_MAX = (1 << WEIGHT_W) - 1

MAX_LOG_INT = (1 << (LOG_OUT_W - Q_FRAC)) - 1  # 63 for Q6.10

# Data directory
DATA_DIR = Path(__file__).resolve().parent / ".." / ".." / "data"

# log2 fractional LUT (Q6.10)
with open(DATA_DIR / "log2_lut.hex") as _f:
    LOG2_LUT = [int(line.strip(), 16) for line in _f if line.strip()]

LOG_TOLERANCE = 2


# ----------------------------------------------------------------
# Reference Model
# ----------------------------------------------------------------

class LogMelRef:

    def __init__(self, vad_threshold=0):
        self.vad_threshold = vad_threshold
        self._build_filterbank()

    def _build_filterbank(self):
        mel_t = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=128,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
        )
        fb_float = mel_t.mel_scale.fb.numpy()
        fb_fixed = np.round(fb_float * (2 ** 15)).astype(np.int64)
        self.fb  = np.clip(fb_fixed, 0, WEIGHT_MAX)

    def _power(self, re: np.ndarray, im: np.ndarray) -> np.ndarray:
        re_s = re.astype(np.int64)
        im_s = im.astype(np.int64)
        half = 1 << (IW - 1)
        re_s = np.where(re_s >= half, re_s - (1 << IW), re_s)
        im_s = np.where(im_s >= half, im_s - (1 << IW), im_s)
        real_sq  = (re_s ** 2).astype(np.uint64)
        imag_sq  = (im_s ** 2).astype(np.uint64)
        sum_full = real_sq + imag_sq
        return ((sum_full >> SHIFT) & POWER_MASK).astype(np.uint64)

    def _filterbank(self, power: np.ndarray) -> np.ndarray:
        p     = power.astype(np.int64)
        accum = p @ self.fb
        return (accum & ACCUM_MASK).astype(np.uint64)

    def _log_one(self, energy: int) -> int:
        if energy == 0:
            return 0
        log2_int = int(energy).bit_length() - 1
        if log2_int > MAX_LOG_INT:
            return (1 << LOG_OUT_W) - 1  # saturation
        mask = (1 << LUT_FRAC) - 1
        if log2_int >= LUT_FRAC:
            addr = (energy >> (log2_int - LUT_FRAC)) & mask
        else:
            addr = (energy << (LUT_FRAC - log2_int)) & mask
        result = (log2_int << Q_FRAC) + LOG2_LUT[addr]
        return result & ((1 << LOG_OUT_W) - 1)

    def is_voiced(self, power: np.ndarray) -> bool:
        energy = int(np.sum(power.astype(np.uint64)))
        energy = min(energy, (1 << 32) - 1)
        return energy > self.vad_threshold

    def compute(self, re: np.ndarray, im: np.ndarray):
        """Returns (log_mel_or_None, voiced_bool)."""
        pwr = self._power(re, im)
        energy = int(np.sum(pwr.astype(np.uint64)))
        energy = min(energy, (1 << 32) - 1)
        cocotb.log.info(f"  Frame spectral energy: {energy}")
        voiced = self.is_voiced(pwr)
        if not voiced:
            return None, False
        mel     = self._filterbank(pwr)
        log_mel = np.array([self._log_one(int(mel[m])) for m in range(N_MELS)],
                           dtype=np.uint64)
        return log_mel, True


# ----------------------------------------------------------------
# Flash Loading
# ----------------------------------------------------------------

async def flash_load_all(dut):
    """Load all SRAMs via flash ports."""

    # Idle all flash ports
    dut.flash_mel_coeff_we_i.value = 0
    dut.flash_mel_coeff_addr_i.value = 0
    dut.flash_mel_coeff_data_i.value = 0
    dut.flash_mel_index_we_i.value = 0
    dut.flash_mel_index_addr_i.value = 0
    dut.flash_mel_index_data_i.value = 0
    dut.flash_log_lut_we_i.value = 0
    dut.flash_log_lut_addr_i.value = 0
    dut.flash_log_lut_data_i.value = 0

    # Load sparse mel coefficients (16-bit)
    with open(DATA_DIR / "mel_coeffs_sparse.hex") as f:
        coeffs = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(coeffs)} sparse mel coeff entries...")
    dut.flash_mel_coeff_we_i.value = 1
    for addr, val in enumerate(coeffs):
        dut.flash_mel_coeff_addr_i.value = addr
        dut.flash_mel_coeff_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_coeff_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    # Load mel indices (8-bit)
    with open(DATA_DIR / "mel_indices.hex") as f:
        indices = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(indices)} mel index entries...")
    dut.flash_mel_index_we_i.value = 1
    for addr, val in enumerate(indices):
        dut.flash_mel_index_addr_i.value = addr
        dut.flash_mel_index_data_i.value = val & 0xFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_index_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    # Load log2 LUT (16-bit, Q6.10)
    with open(DATA_DIR / "log2_lut.hex") as f:
        lut = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(lut)} log LUT entries...")
    dut.flash_log_lut_we_i.value = 1
    for addr, val in enumerate(lut):
        dut.flash_log_lut_addr_i.value = addr
        dut.flash_log_lut_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_log_lut_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    cocotb.log.info("All SRAMs loaded.")


# ----------------------------------------------------------------
# Driver
# ----------------------------------------------------------------

class LogMelDriver:
    def __init__(self, dut):
        self.dut = dut

    async def reset(self, cycles: int = 5):
        dut = self.dut
        await RisingEdge(dut.clk_i)
        dut.reset_i.value        = 1
        dut.re_il.value          = 0
        dut.im_il.value          = 0
        dut.fft_valid_il.value   = 0
        dut.fft_sync_il.value    = 0
        dut.cnn_ready_il.value   = 0
        dut.vad_threshold_il.value = 0   # NEW — VAD disabled by default
        dut.flash_mel_coeff_we_i.value = 0
        dut.flash_mel_coeff_addr_i.value = 0
        dut.flash_mel_coeff_data_i.value = 0
        dut.flash_mel_index_we_i.value = 0
        dut.flash_mel_index_addr_i.value = 0
        dut.flash_mel_index_data_i.value = 0
        dut.flash_log_lut_we_i.value = 0
        dut.flash_log_lut_addr_i.value = 0
        dut.flash_log_lut_data_i.value = 0
        dut.test_mode_i.value = 0
        dut.test_lut_addr_i.value = 0
        dut.test_coeff_addr_i.value = 0
        dut.test_index_addr_i.value = 0
        await ClockCycles(dut.clk_i, cycles)
        dut.reset_i.value = 0
        await ClockCycles(dut.clk_i, 2)

    async def set_vad_threshold(self, threshold: int):
        self.dut.vad_threshold_il.value = threshold
        await RisingEdge(self.dut.clk_i)

    async def drive_frame(self, re: np.ndarray, im: np.ndarray):
        dut = self.dut
        dut.fft_sync_il.value  = 1
        dut.fft_valid_il.value = 0
        await RisingEdge(dut.clk_i)
        dut.fft_sync_il.value  = 0

        for i in range(N_BINS):
            dut.re_il.value        = int(re[i]) & IW_MASK
            dut.im_il.value        = int(im[i]) & IW_MASK
            dut.fft_valid_il.value = 1
            await RisingEdge(dut.clk_i)

        dut.fft_valid_il.value = 0
        dut.re_il.value        = 0
        dut.im_il.value        = 0


# ----------------------------------------------------------------
# Checker
# ----------------------------------------------------------------

class LogMelChecker:

    def __init__(self, dut):
        self.dut = dut

    async def collect_frame(self, pattern: list = None, timeout: int = 1500) -> list:
        if pattern is None:
            pattern = [1]
        dut = self.dut
        results = []
        for cycle in range(timeout):
            dut.cnn_ready_il.value = int(pattern[cycle % len(pattern)])
            await RisingEdge(dut.clk_i)
            if dut.cnn_valid_ol.value == 1 and dut.cnn_ready_il.value == 1:
                results.append(int(dut.cnn_data_ol.value))
            if len(results) == N_MELS:
                break
        dut.cnn_ready_il.value = 0
        return results

    def check(self, got: list, exp: np.ndarray, tag: str = "") -> None:
        assert len(got) == N_MELS, \
            f"{tag}: received {len(got)}/{N_MELS} CNN outputs — pipeline timeout?"
        got_a  = np.array(got, dtype=np.uint64)
        exp_a  = exp.astype(np.uint64)
        deltas = np.abs(got_a.astype(np.int64) - exp_a.astype(np.int64))
        worst  = int(np.argmax(deltas))
        cocotb.log.info(
            f"{tag} | worst mel[{worst}]: "
            f"got=0x{got[worst]:04x}  exp=0x{int(exp[worst]):04x}  "
            f"delta={deltas[worst]}  tolerance={LOG_TOLERANCE}"
        )
        assert np.all(deltas <= LOG_TOLERANCE), \
            f"{tag} FAIL: max delta={deltas[worst]} > {LOG_TOLERANCE} at mel[{worst}]"


# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, units="ns").start())
    ref     = LogMelRef()
    driver  = LogMelDriver(dut)
    checker = LogMelChecker(dut)

    await driver.reset()
    await flash_load_all(dut)
    await driver.reset()

    return ref, driver, checker


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

@cocotb.test()
async def test_zero_input(dut):
    """All-zero input."""
    ref, driver, checker = await setup(dut)

    re  = np.zeros(N_BINS, dtype=np.uint64)
    im  = np.zeros(N_BINS, dtype=np.uint64)
    exp, voiced = ref.compute(re, im)
    # threshold=0, zero input has zero energy, 0 > 0 is false so not voiced
    # but this matches old behavior where zero input produced zero output
    # set threshold to max to guarantee pass-through for this test
    if not voiced:
        # zero energy doesn't exceed threshold=0, so no output expected
        await driver.drive_frame(re, im)
        got = await checker.collect_frame(timeout=500)
        assert len(got) == 0, "Zero input should produce no output with threshold=0"
        cocotb.log.info("test_zero_input PASSED — zero energy correctly below threshold")
        return
    await driver.drive_frame(re, im)
    got = await checker.collect_frame()
    checker.check(got, exp, tag="test_zero_input")


@cocotb.test()
async def test_single_frame(dut):
    """Random FFT frame."""
    ref, driver, checker = await setup(dut)
    rng = np.random.default_rng(42)
    re  = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
    im  = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
    exp, voiced = ref.compute(re, im)
    assert voiced, "Random full-range input should exceed threshold=0"
    await driver.drive_frame(re, im)
    got = await checker.collect_frame()
    checker.check(got, exp, tag="test_single_frame")


@cocotb.test()
async def test_two_frames(dut):
    """Two consecutive frames."""
    ref, driver, checker = await setup(dut)
    rng = np.random.default_rng(7)
    for frame_idx in range(2):
        re  = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
        im  = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
        exp, voiced = ref.compute(re, im)
        assert voiced
        await driver.drive_frame(re, im)
        got = await checker.collect_frame()
        checker.check(got, exp, tag=f"test_two_frames[{frame_idx}]")
        await ClockCycles(dut.clk_i, 5)
        cocotb.log.info(f"test_two_frames frame {frame_idx} PASSED")

@cocotb.test()
async def test_vad_silence_suppressed(dut):
    """Near-zero input with high threshold — frame should be suppressed."""
    ref, driver, checker = await setup(dut)

    vad_thresh = 100_000
    ref.vad_threshold = vad_thresh
    await driver.set_vad_threshold(vad_thresh)

    re = np.ones(N_BINS, dtype=np.uint64) * 2
    im = np.ones(N_BINS, dtype=np.uint64) * 2
    exp, voiced = ref.compute(re, im)
    assert not voiced, "Low energy should not exceed threshold"

    await driver.drive_frame(re, im)
    got = await checker.collect_frame(timeout=500)
    assert len(got) == 0, \
        f"VAD should have suppressed frame but got {len(got)} outputs"

    cocotb.log.info("test_vad_silence_suppressed PASSED")


@cocotb.test()
async def test_vad_speech_passes(dut):
    """Full-range random input with threshold — should pass through."""
    ref, driver, checker = await setup(dut)

    vad_thresh = 100_000
    ref.vad_threshold = vad_thresh
    await driver.set_vad_threshold(vad_thresh)

    rng = np.random.default_rng(42)
    re  = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
    im  = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
    exp, voiced = ref.compute(re, im)
    assert voiced, "Full-range input should exceed threshold"

    await driver.drive_frame(re, im)
    got = await checker.collect_frame()
    checker.check(got, exp, tag="test_vad_speech_passes")

    cocotb.log.info("test_vad_speech_passes PASSED")


@cocotb.test()
async def test_vad_silence_then_speech(dut):
    """Two frames: silence (suppressed) then speech (passes)."""
    ref, driver, checker = await setup(dut)

    vad_thresh = 100_000
    ref.vad_threshold = vad_thresh
    await driver.set_vad_threshold(vad_thresh)

    # Frame 1: silence
    re_q = np.ones(N_BINS, dtype=np.uint64) * 2
    im_q = np.ones(N_BINS, dtype=np.uint64) * 2
    _, voiced = ref.compute(re_q, im_q)
    assert not voiced

    await driver.drive_frame(re_q, im_q)
    got = await checker.collect_frame(timeout=500)
    assert len(got) == 0, "Silent frame should produce no output"

    # Frame 2: speech
    rng = np.random.default_rng(99)
    re_l = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
    im_l = rng.integers(0, 1 << IW, size=N_BINS, dtype=np.uint64)
    exp, voiced = ref.compute(re_l, im_l)
    assert voiced

    await driver.drive_frame(re_l, im_l)
    got = await checker.collect_frame()
    checker.check(got, exp, tag="test_vad_silence_then_speech[speech]")

    cocotb.log.info("test_vad_silence_then_speech PASSED")


@cocotb.test()
async def test_vad_threshold_boundary(dut):
    """Test VAD with energy near the threshold boundary."""
    ref, driver, checker = await setup(dut)

    # Use small FFT values that won't saturate
    rng = np.random.default_rng(42)

    # Frame with low energy — should be suppressed
    re_low = rng.integers(0, 64, size=N_BINS, dtype=np.uint64)
    im_low = rng.integers(0, 64, size=N_BINS, dtype=np.uint64)
    pwr_low = ref._power(re_low, im_low)
    energy_low = min(int(np.sum(pwr_low.astype(np.uint64))), (1 << 32) - 1)

    # Frame with moderate energy — should pass
    re_mid = rng.integers(0, 4096, size=N_BINS, dtype=np.uint64)
    im_mid = rng.integers(0, 4096, size=N_BINS, dtype=np.uint64)
    pwr_mid = ref._power(re_mid, im_mid)
    energy_mid = min(int(np.sum(pwr_mid.astype(np.uint64))), (1 << 32) - 1)

    # Set threshold between the two
    threshold = (energy_low + energy_mid) // 2
    cocotb.log.info(f"Energy low: {energy_low}  mid: {energy_mid}  threshold: {threshold}")

    ref.vad_threshold = threshold
    await driver.set_vad_threshold(threshold)

    # Drive low energy frame — should be suppressed
    _, voiced = ref.compute(re_low, im_low)
    assert not voiced, f"Low energy {energy_low} should be below threshold {threshold}"
    await driver.drive_frame(re_low, im_low)
    got = await checker.collect_frame(timeout=500)
    assert len(got) == 0, f"Low energy frame should be suppressed, got {len(got)} outputs"
    cocotb.log.info("Low energy frame correctly suppressed")

    # Drive moderate energy frame — should pass
    exp, voiced = ref.compute(re_mid, im_mid)
    assert voiced, f"Mid energy {energy_mid} should exceed threshold {threshold}"
    await driver.drive_frame(re_mid, im_mid)
    got = await checker.collect_frame()
    checker.check(got, exp, tag="test_vad_threshold_boundary[mid]")
    cocotb.log.info("Moderate energy frame correctly passed")

    cocotb.log.info("test_vad_threshold_boundary PASSED")