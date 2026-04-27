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
    dut.test_mode_i.value = 0
    dut.test_coeff_addr_i.value = 0
    dut.test_index_addr_i.value = 0
    dut.test_lut_addr_i.value = 0
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
    """Collect complete N_MELS-value frames from the pipeline output.

    Reads POST-bfpexp-compensation values via `dut.mel_compensated_o`, which is
    what the spect_buffer (and ultimately the CNN) actually consumes.  This is
    different from the earlier behavior which read pre-compensation values from
    `u_logmel.cnn_data_ol` and therefore could not agree with a BFP-aware
    golden model.

    Also records the `bfpexp_for_mel` latched in pipeline_top at each fft_sync,
    plus the pre-compensation logmel output — letting the comparison script
    isolate stage-by-stage residual error (FFT vs mel-filterbank-and-log) and
    verify the BFP model used by the golden.
    """
    frames = []
    pre_frames = []        # raw u_logmel.cnn_data_ol (pre-compensation)
    bfpexps = []           # one int8 per fft_sync_rr
    # Raw FFT bin dumps per frame: list of (re[129], im[129]) tuples.
    # Captured from fft_result_rr while fft_valid is asserted after each
    # fft_sync_rr.  This is the signal driving logmel's power_calc.
    fft_dumps = []
    cur_fft_re = []
    cur_fft_im = []
    # Absolute input-sample count (increments on every valid_i=1 cycle) at each
    # fft_sync_rr.  Exposes whether the R2FFT is actually seeing hop=128 or
    # whether its FSM is dropping samples during FFT+DMA processing.
    sample_counter = 0
    sync_sample_counts = []
    last_bfpexp = None
    for _ in range(timeout_clks):
        await RisingEdge(dut.clk_i)
        # Count every clock where the driver asserts valid_i.
        try:
            if int(dut.valid_i.value):
                sample_counter += 1
        except (ValueError, AttributeError):
            pass
        try:
            if int(dut.fft_sync_rr.value):
                # bfpexp_for_mel was declared `logic signed [7:0]` — reads as uint,
                # sign-extend manually.
                raw = int(dut.bfpexp_for_mel.value)
                if raw >= 0x80:
                    raw -= 0x100
                last_bfpexp = raw
                # Record the absolute sample count at this sync — this is the
                # number of input samples consumed by the pipeline so far.
                sync_sample_counts.append(sample_counter)
                # New frame — flush any in-progress bin collection.
                if cur_fft_re:
                    fft_dumps.append((cur_fft_re, cur_fft_im))
                cur_fft_re = []
                cur_fft_im = []
        except (ValueError, AttributeError):
            pass
        # Capture FFT bin while fft_valid is high (exactly N_BINS=129 cycles).
        #
        # IMPORTANT: there is a 3-cycle DMA-readout pipeline delay between
        # fft_valid going high and the ACTUAL bin 0 appearing on
        # fft_result_rr.  The path is:
        #   ram read → a_dmadr_real_r reg → a_result reg → o_fft_result reg
        #     → fft_result_r reg → fft_result_rr reg
        # That's ~7 cycles from a_sync to bin 0 in fft_result_rr, but
        # fft_valid goes high only ~4 cycles after a_sync (1 reg + 2 more
        # before bin_cnt_q loads), so the first 3 fft_valid=1 cycles show
        # STALE fft_result_rr data from the previous frame.  Skip them.
        try:
            if int(dut.fft_valid.value):
                re_u = int(dut.fft_re.value) & 0xFFFF
                im_u = int(dut.fft_im.value) & 0xFFFF
                # Two's-complement sign extension to int16.
                re_s = re_u - 0x10000 if re_u & 0x8000 else re_u
                im_s = im_u - 0x10000 if im_u & 0x8000 else im_u
                cur_fft_re.append(re_s)
                cur_fft_im.append(im_s)
        except (ValueError, AttributeError):
            pass
        try:
            valid = int(dut.mel_compensated_valid_o.value)
            ready = int(dut.u_logmel.cnn_ready_il.value)
            if valid and ready:
                v_post = int(dut.mel_compensated_o.value)
                v_pre  = int(dut.u_logmel.cnn_data_ol.value)
                if not frames or len(frames[-1]) == N_MELS:
                    frames.append([])
                    pre_frames.append([])
                    bfpexps.append(last_bfpexp)
                frames[-1].append(v_post)
                pre_frames[-1].append(v_pre)
        except (ValueError, AttributeError):
            pass
    # Flush any trailing FFT dump.
    if cur_fft_re:
        fft_dumps.append((cur_fft_re, cur_fft_im))
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()
        pre_frames.pop()
        bfpexps.pop()
    return frames, pre_frames, bfpexps, fft_dumps, sync_sample_counts


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
    frames, _pre, _bfp, _fft, _ssc = await collect_frames(dut, timeout)
    sync_count = await sync_task
    await drive_task

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
    frames, pre_frames, bfpexps, fft_dumps, sync_sample_counts = await collect_frames(dut, timeout)

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
    here = os.path.dirname(__file__) or "."
    npy = os.path.join(here, "rtl_features.npy")
    np.save(npy, mat)

    # Also save the pre-compensation logmel output (for isolating FFT-stage
    # error from mel-filterbank/log-stage error) and the bfpexp value the RTL
    # latched for each frame (for validating the golden's BFP model).
    pre_mat = np.stack(
        [np.array(frame, np.float32) / (1 << Q_FRAC) for frame in pre_frames],
        axis=1,
    )
    np.save(os.path.join(here, "rtl_features_precomp.npy"), pre_mat)
    np.save(
        os.path.join(here, "rtl_bfpexps.npy"),
        np.array([b if b is not None else -1 for b in bfpexps], dtype=np.int8),
    )

    # Raw FFT bin dumps — only keep frames with exactly N_BINS=129 entries,
    # and crop to the same count as the emitted logmel frames.  This lets the
    # golden's bit-accurate FFT be compared directly against the RTL's.
    n_bins_expected = 129
    clean_fft = [
        (re, im) for (re, im) in fft_dumps
        if len(re) == n_bins_expected and len(im) == n_bins_expected
    ]
    # The first STARTUP_LOSS FFT dumps may precede the first emitted logmel
    # frame — include them anyway and let the comparison script align.
    if clean_fft:
        re_mat = np.array([re for (re, _) in clean_fft], dtype=np.int32)  # (Nfft, 129)
        im_mat = np.array([im for (_, im) in clean_fft], dtype=np.int32)
        np.save(os.path.join(here, "rtl_fft_re.npy"), re_mat)
        np.save(os.path.join(here, "rtl_fft_im.npy"), im_mat)
        cocotb.log.info(
            "Dumped raw FFT bins: %d frames x %d bins",
            re_mat.shape[0], re_mat.shape[1],
        )

    # Absolute input-sample count at each fft_sync_rr pulse.  A correctly
    # hopping pipeline produces sync_sample_counts with a constant 128-sample
    # delta between consecutive entries (once past startup).  Drift here
    # directly exposes the R2FFT dropping samples during its busy state.
    np.save(
        os.path.join(here, "rtl_sync_sample_counts.npy"),
        np.array(sync_sample_counts, dtype=np.int32),
    )
    cocotb.log.info(
        "Sync sample counts (first 20): %s",
        sync_sample_counts[:20],
    )
    if len(sync_sample_counts) >= 2:
        deltas = np.diff(np.array(sync_sample_counts))
        cocotb.log.info(
            "Per-frame sample hops: min=%d max=%d mean=%.1f median=%.1f  (expect 128)",
            int(deltas.min()), int(deltas.max()),
            float(deltas.mean()), float(np.median(deltas)),
        )
    cocotb.log.info(
        "PASS -- %dx%d feature matrix saved to %s" % (n, N_MELS, npy)
    )
    cocotb.log.info(
        "bfpexps per frame (min=%d max=%d mean=%.1f): %s",
        min(b for b in bfpexps if b is not None),
        max(b for b in bfpexps if b is not None),
        np.mean([b for b in bfpexps if b is not None]),
        bfpexps,
    )