import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_W    = 16       # IW_STFFT
N_MELS      = 40
OUT_W       = 16
WIN_LEN     = 256      # FFT_SIZE
HOP         = WIN_LEN // 2   # 50% overlap = 128 samples
SAMPLE_RATE = 16_000
Q_FRAC      = 10

# CE_EVERY: clocks between valid input samples.
#
# With 50% overlap (HOP=128), logmel receives a new frame every HOP*CE_EVERY
# clocks.  Logmel takes ~7500 clocks to process each frame.  Requirement:
#   HOP * CE_EVERY > 7500
#   128 * CE_EVERY > 7500
#   CE_EVERY > 58.6  -->  use CE_EVERY = 64
#
# NOTE: CE_EVERY >= 15 is also needed so each R2FFT instance finishes its
# computation+DMA before its next window's samples arrive.
CE_EVERY    = 64

N_SAMPLES   = 7_500

# Drain: extra clocks after the last sample to let the pipeline flush.
# Last sync at ~N_SAMPLES*CE_EVERY clocks; logmel adds ~7500 more.
DRAIN       = 30_000

# Expected frame count with 50% overlap:
#   ideal = (N_SAMPLES - WIN_LEN) // HOP + 1 = (7500-256)//128 + 1 = 57
#   Subtract startup frames (~2) -> expect ~55
EXPECTED_MIN = 40
EXPECTED_MAX = 60

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
    """200 Hz to 7 kHz linear chirp, 16-bit signed range."""
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
    Drive samples with CE_EVERY clock spacing.
    CE_EVERY=64 gives HOP*CE_EVERY = 8192 clocks between frames, which
    exceeds logmel's ~7500 clock processing time, so every frame is consumed.
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
    """Count o_fft_sync pulses (combines A and B syncs from new stfft)."""
    count = 0
    for _ in range(duration_clks):
        await RisingEdge(dut.clk_i)
        try:
            if int(dut.u_stfft.o_fft_sync.value):
                count += 1
        except (ValueError, AttributeError):
            pass
    return count


async def monitor_dma_bins(dut, duration_clks, bins=(10, 11, 12, 13, 14), max_frames=5):
    """
    Diagnostic: log raw R2FFT DMA output for selected bins.
    Run alongside collect_frames -- does NOT replace it.
    Stops logging after max_frames to avoid flooding the log.
    """
    frames_seen = 0
    for _ in range(duration_clks):
        await RisingEdge(dut.clk_i)
        if frames_seen >= max_frames:
            continue
        try:
            if int(dut.u_stfft.a_dmaact.value) == 1:
                addr = int(dut.u_stfft.a_dma_addr.value)
                if addr in bins:
                    re = int(dut.u_stfft.a_dmadr_real_w.value)
                    im = int(dut.u_stfft.a_dmadr_imag_w.value)
                    if re > 32767: re -= 65536
                    if im > 32767: im -= 65536
                    cocotb.log.info("DMA bin[%d] re=%d im=%d" % (addr, re, im))
                    if addr == max(bins):
                        frames_seen += 1
        except (AttributeError, ValueError):
            pass


# Probe: count how many times fft_valid=1 with large fft_re values
_fft_probe_max_re   = [0]  # max |re| seen while fft_valid=1
_fft_probe_count    = [0]  # total fft_valid=1 cycles seen
_fft_probe_nonzero  = [0]  # fft_valid=1 cycles where |re|>100


async def collect_frames(dut, timeout_clks):
    """Collect complete N_MELS-value frames from logmel output."""
    frames = []
    _fft_probe_max_re[0]  = 0
    _fft_probe_count[0]   = 0
    _fft_probe_nonzero[0] = 0
    for _ in range(timeout_clks):
        await RisingEdge(dut.clk_i)
        # Probe fft_re while fft_valid=1 (first 2 FFT frames only to avoid log flood)
        try:
            if int(dut.fft_valid.value) == 1:
                re = int(dut.fft_re.value)
                if re > 32767: re -= 65536
                _fft_probe_count[0] += 1
                if abs(re) > _fft_probe_max_re[0]:
                    _fft_probe_max_re[0] = abs(re)
                if abs(re) > 100:
                    _fft_probe_nonzero[0] += 1
        except (ValueError, AttributeError):
            pass
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
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()
    cocotb.log.info(
        "fft_re probe: total_valid=%d  max|re|=%d  cycles_with_|re|>100=%d"
        % (_fft_probe_count[0], _fft_probe_max_re[0], _fft_probe_nonzero[0])
    )
    return frames


# ---------------------------------------------------------------------------
# Test 1 — smoke: verify both FFT channels fire and logmel produces frames
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_frames(dut):
    """
    Feed chirp at CE_EVERY=64 clocks/sample and verify:
      - FFT sync pulses are seen (both A and B channels)
      - At least one logmel frame is produced
    """
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)

    samples = make_chirp(N_SAMPLES)
    timeout = N_SAMPLES * CE_EVERY + DRAIN

    ideal = (N_SAMPLES - WIN_LEN) // HOP + 1

    drive_task = cocotb.start_soon(drive_samples(dut, samples, ce_every=CE_EVERY))
    sync_task  = cocotb.start_soon(monitor_fft_sync(dut, timeout))
    dma_task   = cocotb.start_soon(monitor_dma_bins(dut, timeout))  # diagnostic only
    frames     = await collect_frames(dut, timeout)
    sync_count = await sync_task
    await drive_task
    await dma_task

    n = len(frames)
    cocotb.log.info(
        "FFT sync pulses : %d  (ideal 50pct-overlap frames for %d samples = %d)"
        % (sync_count, N_SAMPLES, ideal)
    )
    cocotb.log.info(
        "Logmel frames   : %d  (expect %d to %d)"
        % (n, EXPECTED_MIN, EXPECTED_MAX)
    )

    assert sync_count > 0, "No FFT sync pulses -- stfft A or B not running"
    assert n > 0,          "No logmel frames -- check sync/timing in pipeline_top"


# ---------------------------------------------------------------------------
# Test 2 — full pipeline validation
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_pipeline(dut):
    """
    Full pipeline test with 50% overlap stfft:
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
        "%d samples (%d clk/sample, HOP=%d) : %d frames  (expect %d to %d)"
        % (N_SAMPLES, CE_EVERY, HOP, n, EXPECTED_MIN, EXPECTED_MAX)
    )

    # 1. Frame count
    assert EXPECTED_MIN <= n <= EXPECTED_MAX, (
        "Frame count %d outside [%d, %d].\n"
        "  If ~half expected: logmel is dropping frames. Increase CE_EVERY (now %d).\n"
        "  Rule: HOP * CE_EVERY > logmel_time (~7500 clocks). Min CE_EVERY = %d.\n"
        "  If ~double expected: fft_sync timing wrong in pipeline_top (fft_sync_rr).\n"
        "  If 0: stfft A/B instances not running."
        % (n, EXPECTED_MIN, EXPECTED_MAX, CE_EVERY, (7500 // HOP) + 1)
    )

    # 2. Shape and range
    max_val = (1 << OUT_W) - 1
    for i, frame in enumerate(frames):
        assert len(frame) == N_MELS, \
            "frame %d: %d mels != %d" % (i, len(frame), N_MELS)
        for j, v in enumerate(frame):
            assert 0 <= v <= max_val, \
                "frame[%d][%d] = %d out of [0, %d]" % (i, j, v, max_val)

    # 3. Non-zero
    all_v = [v for frame in frames for v in frame]
    nz    = sum(1 for v in all_v if v > 0)
    cocotb.log.info(
        "Non-zero outputs: %d/%d (%.1f%%)" % (nz, len(all_v), 100*nz/len(all_v))
    )
    assert nz > 0, "All outputs zero -- pipeline not processing"

    # 4. Save
    mat = np.stack(
        [np.array(frame, np.float32) / (1 << Q_FRAC) for frame in frames],
        axis=1
    )
    npy = os.path.join(os.path.dirname(__file__) or ".", "rtl_features.npy")
    np.save(npy, mat)
    cocotb.log.info(
        "PASS -- %dx%d feature matrix saved to %s" % (n, N_MELS, npy)
    )