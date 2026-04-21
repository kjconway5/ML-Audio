import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_W    = 16       # must match pipeline IW_STFFT
N_MELS      = 40
OUT_W       = 16
WIN_LEN     = 256      # FFT_SIZE (non-overlapping → HOP = WIN_LEN)
SAMPLE_RATE = 16_000
Q_FRAC      = 10

# CE_EVERY: clocks between valid input samples.
#
# WHY CE_EVERY must be >= 16:
#   At full rate (CE_EVERY=1) the R2FFT produces one frame every ~3750 clocks
#   but logmel_top takes ~7500 clocks to process each frame.  logmel therefore
#   drops every second FFT frame, halving the output count.
#   Setting CE_EVERY=20 makes the FFT frame period (20*256 + ~3494 = 8614 clocks)
#   comfortably longer than logmel's processing time (~7500 clocks), so every
#   FFT frame is consumed by logmel.
CE_EVERY    = 20

N_SAMPLES   = 7_500

# Drain: extra clocks after the last sample to let the pipeline flush.
# Last FFT sync fires ~ N_SAMPLES*CE_EVERY clocks in.
# logmel then needs ~7500 more clocks.  Use 20_000 for safety.
DRAIN       = 20_000

# Expected frame count.
#
# The exact count depends on R2FFT throughput and logmel latency.
# Empirically the pipeline produces ~23 frames from 7500 samples at CE_EVERY~20.
# If count is consistently ~half of FFT sync pulses, logmel is still the
# throughput bottleneck -- increase CE_EVERY until count == FFT sync count.
# EXPECTED_MAX is capped at N_FRAMES (spect_buffer size = 50).
EXPECTED_MIN = 5
EXPECTED_MAX = 50

DATA_DIR = Path(__file__).resolve().parent / ".." / "Log-Mel" / "data"


# ---------------------------------------------------------------------------
# Flash helpers
# ---------------------------------------------------------------------------

def _idle_flash(dut):
    dut.flash_mel_coeff_we_i.value   = 0
    dut.flash_mel_coeff_addr_i.value = 0
    dut.flash_mel_coeff_data_i.value = 0
    dut.flash_mel_index_we_i.value   = 0
    dut.flash_mel_index_addr_i.value = 0
    dut.flash_mel_index_data_i.value = 0
    dut.flash_log_lut_we_i.value     = 0
    dut.flash_log_lut_addr_i.value   = 0
    dut.flash_log_lut_data_i.value   = 0


async def flash_load_all(dut):
    _idle_flash(dut)

    with open(DATA_DIR / "mel_coeffs_sparse.hex") as f:
        coeffs = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d sparse mel coeff entries..." % len(coeffs))
    dut.flash_mel_coeff_we_i.value = 1
    for addr, val in enumerate(coeffs):
        dut.flash_mel_coeff_addr_i.value = addr
        dut.flash_mel_coeff_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_coeff_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    with open(DATA_DIR / "mel_indices.hex") as f:
        indices = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d mel index entries..." % len(indices))
    dut.flash_mel_index_we_i.value = 1
    for addr, val in enumerate(indices):
        dut.flash_mel_index_addr_i.value = addr
        dut.flash_mel_index_data_i.value = val & 0xFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_index_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    with open(DATA_DIR / "log2_lut.hex") as f:
        lut = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d log LUT entries..." % len(lut))
    dut.flash_log_lut_we_i.value = 1
    for addr, val in enumerate(lut):
        dut.flash_log_lut_addr_i.value = addr
        dut.flash_log_lut_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_log_lut_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    cocotb.log.info("All SRAMs loaded.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chirp(n):
    """Linear chirp 200 Hz to 7 kHz, scaled to 16-bit signed range."""
    dur   = n / SAMPLE_RATE
    t     = np.arange(n) / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * ((1 << (SAMPLE_W - 1)) - 1)).astype(np.int32)


async def do_reset(dut):
    dut.reset_i.value  = 1
    dut.data_i.value   = 0
    dut.valid_i.value  = 0
    _idle_flash(dut)
    await ClockCycles(dut.clk_i, 10)
    dut.reset_i.value  = 0
    await ClockCycles(dut.clk_i, 5)


async def drive_samples(dut, samples, ce_every=1):
    """
    Drive samples into the DUT.
      ce_every=1  -> one sample per clock (back-to-back, max rate)
      ce_every=N  -> one sample every N clocks (valid high for 1, idle for N-1)

    NOTE: keep ce_every >= 16 so that the FFT frame period is longer than
    logmel's processing time (~7500 clocks).  Otherwise logmel drops frames.
    """
    mask = (1 << SAMPLE_W) - 1
    for s in samples:
        dut.data_i.value  = int(s) & mask
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)
        for _ in range(ce_every - 1):
            dut.valid_i.value = 0
            dut.data_i.value  = 0
            await RisingEdge(dut.clk_i)
    dut.valid_i.value = 0
    dut.data_i.value  = 0


async def monitor_fft_sync(dut, duration_clks):
    """Count o_fft_sync pulses inside u_stfft over duration_clks cycles."""
    count = 0
    for _ in range(duration_clks):
        await RisingEdge(dut.clk_i)
        try:
            if int(dut.u_stfft.o_fft_sync.value):
                count += 1
        except (ValueError, AttributeError):
            pass
    return count


async def collect_frames(dut, timeout_clks):
    """
    Collect complete mel frames (N_MELS values each) from logmel output.
    A new frame is started when N_MELS values have accumulated in the current frame.
    """
    frames = []
    for _ in range(timeout_clks):
        await RisingEdge(dut.clk_i)
        try:
            valid = int(dut.u_logmel.cnn_valid_ol.value)
            ready = int(dut.u_logmel.cnn_ready_il.value)
            if valid and ready:
                v = int(dut.u_logmel.cnn_data_ol.value)
                if not frames or len(frames[-1]) == N_MELS:
                    frames.append([])
                frames[-1].append(v)
        except (ValueError, AttributeError):
            pass
    # Drop any incomplete trailing frame
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()
    return frames


# ---------------------------------------------------------------------------
# Test 1 — smoke: verify FFT syncs fire and logmel produces frames
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_frames(dut):
    """
    Feed a chirp at CE_EVERY clocks/sample and verify:
      - At least one FFT sync pulse (stfft is running)
      - At least one complete logmel frame
    """
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)

    samples = make_chirp(N_SAMPLES)
    timeout = N_SAMPLES * CE_EVERY + DRAIN

    drive_task = cocotb.start_soon(drive_samples(dut, samples, ce_every=CE_EVERY))
    sync_task  = cocotb.start_soon(monitor_fft_sync(dut, timeout))
    frames     = await collect_frames(dut, timeout)
    sync_count = await sync_task
    await drive_task

    n = len(frames)
    ideal_fft_frames = (N_SAMPLES - WIN_LEN) // WIN_LEN + 1
    cocotb.log.info(
        "FFT sync pulses : %d  (ideal non-overlapping frames for %d samples = %d)"
        % (sync_count, N_SAMPLES, ideal_fft_frames)
    )
    cocotb.log.info(
        "Logmel frames   : %d  (expect %d-%d after pipeline latency)"
        % (n, EXPECTED_MIN, EXPECTED_MAX)
    )

    assert sync_count > 0, "No FFT sync pulses seen — stfft not running"
    assert n > 0,          "No logmel frames produced — check sync/bin-counter timing"


# ---------------------------------------------------------------------------
# Test 2 — full pipeline: frame count, range, non-zero, save
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_pipeline(dut):
    """
    Full pipeline validation:
      1. Frame count in [EXPECTED_MIN, EXPECTED_MAX]
      2. Every frame has exactly N_MELS values in [0, 2^OUT_W)
      3. At least some outputs are non-zero
      4. Saves feature matrix to rtl_features.npy
    """
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)

    samples = make_chirp(N_SAMPLES)
    timeout = N_SAMPLES * CE_EVERY + DRAIN

    cocotb.start_soon(drive_samples(dut, samples, ce_every=CE_EVERY))
    frames = await collect_frames(dut, timeout)

    n = len(frames)
    cocotb.log.info(
        "%d samples (%d clk/sample) : %d frames  (expect %d to %d)"
        % (N_SAMPLES, CE_EVERY, n, EXPECTED_MIN, EXPECTED_MAX)
    )

    # 1. Frame count
    assert EXPECTED_MIN <= n <= EXPECTED_MAX, (
        "Frame count %d outside expected range [%d, %d].\n"
        "  If count ~ half expected : logmel is dropping frames because FFT\n"
        "    is faster than logmel.  Increase CE_EVERY (currently %d).\n"
        "    Rule of thumb: CE_EVERY >= 16 guarantees FFT period > logmel period.\n"
        "  If count is 0 : check fft_sync_rr timing in pipeline_top.sv.\n"
        "  If range itself is wrong : adjust EXPECTED_MIN/MAX in this file."
        % (n, EXPECTED_MIN, EXPECTED_MAX, CE_EVERY)
    )

    # 2. Shape and value range
    max_val = (1 << OUT_W) - 1
    for i, frame in enumerate(frames):
        assert len(frame) == N_MELS, \
            "frame %d: got %d mels, expected %d" % (i, len(frame), N_MELS)
        for j, v in enumerate(frame):
            assert 0 <= v <= max_val, \
                "frame[%d][%d] = %d out of [0, %d]" % (i, j, v, max_val)

    # 3. Non-zero check
    all_v = [v for frame in frames for v in frame]
    nz    = sum(1 for v in all_v if v > 0)
    cocotb.log.info(
        "Non-zero outputs: %d/%d (%.1f%%)" % (nz, len(all_v), 100 * nz / len(all_v))
    )
    assert nz > 0, "All outputs are zero — pipeline is not processing signal"

    # 4. Save feature matrix
    mat = np.stack(
        [np.array(frame, np.float32) / (1 << Q_FRAC) for frame in frames],
        axis=1
    )
    npy = os.path.join(os.path.dirname(__file__) or ".", "rtl_features.npy")
    np.save(npy, mat)
    cocotb.log.info(
        "PASS -- %dx%d feature matrix saved to %s" % (n, N_MELS, npy)
    )