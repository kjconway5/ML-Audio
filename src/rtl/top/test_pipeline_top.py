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
# Two constraints set the floor for the new single-channel stfft:
#   1. logmel needs ~7500 clocks/frame:  HOP * CE_EVERY > 7500
#                                          -> CE_EVERY > 58.6
#   2. The new R2FFT compute is ~11k clocks/frame (vs the old dual-channel
#      design's parallel ~5.5k effective). The FFT must produce a frame
#      every HOP input samples or it falls behind:
#                                        HOP * CE_EVERY > ~11500
#                                          -> CE_EVERY > 89.8
#
# The FFT constraint dominates with the new design — at CE_EVERY=64 (which
# worked with the old dual-channel stfft) the single-channel FFT can't keep
# up, the ring-buffer readout stalls, and stfft's i_ready backpressures the
# producer to throttle the input rate. Frames stay correct (no corruption)
# but the test runs longer.
#
# Setting CE_EVERY=96 keeps the FFT comfortably ahead, so i_ready stays high
# and the run matches the old test's timing more closely.
CE_EVERY    = 96

N_SAMPLES   = 7_500

# Drain: extra clocks after the last sample for the pipeline to flush.
# Last input at ~N_SAMPLES*CE_EVERY clocks; the new stfft adds a 256-cycle
# readout, then R2FFT compute (~11k) + DMA (256) + logmel (~7500). 50k is
# enough at CE_EVERY=96; bump to 200k if you run at CE_EVERY=64 (so the
# i_ready-throttled producer has time to finish).
DRAIN       = 50_000

# Expected frame count with 50% overlap:
#   ideal = (N_SAMPLES - WIN_LEN) // HOP + 1 = (7500 - 256) // 128 + 1 = 57
#   Subtract a couple of startup-loss frames -> ~55
EXPECTED_MIN = 40
EXPECTED_MAX = 60

DATA_DIR = Path(__file__).resolve().parent / ".." / "Log-Mel" / "data"


# ---------------------------------------------------------------------------
# Flash helpers (unchanged)
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
    Drive samples with CE_EVERY clock spacing, honouring stfft's i_ready.

    The new single-channel stfft can backpressure (i_ready=0) under two
    conditions:
      1. A second frame trigger would queue before the first has started.
      2. The readout state machine is in progress AND the FFT is stalled
         waiting for one of its ping-pong RAMs to free up — accepting a
         new sample would overwrite an unread ring-buffer slot.

    At CE_EVERY=96 neither condition fires in practice. At CE_EVERY <= ~86
    the FFT can't keep up with 50% overlap and i_ready will throttle the
    producer; the loop below handles that gracefully (samples just get
    delayed, no data is lost).
    """
    mask = (1 << SAMPLE_W) - 1
    for s in samples:
        dut.data_i.value  = int(s) & mask
        dut.valid_i.value = 1
        # Hold valid_i high until we see i_ready high at a clock edge.
        await RisingEdge(dut.clk_i)
        try:
            while int(dut.u_stfft.i_ready.value) == 0:
                await RisingEdge(dut.clk_i)
        except (ValueError, AttributeError):
            pass  # i_ready not visible — fall back to blind drive
        # Idle for the rest of the pacing interval.
        for _ in range(ce_every - 1):
            dut.valid_i.value = 0
            dut.data_i.value  = 0
            await RisingEdge(dut.clk_i)
    dut.valid_i.value = 0
    dut.data_i.value  = 0


async def monitor_fft_sync(dut, duration_clks):
    """Count emitted FFT frames.

    The new stfft drives o_last for exactly one cycle per frame (with
    the final bin sample), so counting o_last gives a clean frame count.
    """
    count = 0
    for _ in range(duration_clks):
        await RisingEdge(dut.clk_i)
        try:
            if int(dut.u_stfft.o_last.value):
                count += 1
        except (ValueError, AttributeError):
            pass
    return count


async def collect_frames(dut, timeout_clks):
    """Collect logmel frames + diagnostic FFT bin dumps.

    Bin alignment is exact in the new pipeline_top: at every cycle where
    fft_valid=1, fft_result_rr holds the corresponding bin (no leading
    stale cycles). The previous "skip first 3 fft_valid cycles" workaround
    is gone.

    Returns (frames, pre_frames, bfpexps, fft_dumps, sync_sample_counts):
      frames               -- post-BFP mel_compensated_o values
      pre_frames           -- pre-compensation cnn_data_ol values
      bfpexps              -- one int8 per fft_sync_rr (latched bfpexp_for_mel)
      fft_dumps            -- list of (re[129], im[129]) per frame
      sync_sample_counts   -- absolute input-sample count at each fft_sync_rr
    """
    frames             = []
    pre_frames         = []
    bfpexps            = []
    fft_dumps          = []
    cur_fft_re         = []
    cur_fft_im         = []
    sample_counter     = 0
    sync_sample_counts = []
    last_bfpexp        = None

    for _ in range(timeout_clks):
        await RisingEdge(dut.clk_i)

        # Count every clock where the driver asserts valid_i.
        try:
            if int(dut.valid_i.value):
                sample_counter += 1
        except (ValueError, AttributeError):
            pass

        # Frame-start marker: fft_sync_rr fires at the cycle when
        # fft_result_rr holds bin 0 and fft_valid first goes high.
        # bfpexp_for_mel was latched one cycle earlier (on fft_sync_r)
        # so it's already stable here.
        try:
            if int(dut.fft_sync_rr.value):
                raw = int(dut.bfpexp_for_mel.value)
                if raw >= 0x80:
                    raw -= 0x100
                last_bfpexp = raw
                sync_sample_counts.append(sample_counter)
                if cur_fft_re:
                    fft_dumps.append((cur_fft_re, cur_fft_im))
                cur_fft_re = []
                cur_fft_im = []
        except (ValueError, AttributeError):
            pass

        # Per-bin capture — every fft_valid=1 cycle carries one valid bin.
        try:
            if int(dut.fft_valid.value):
                re_u = int(dut.fft_re.value) & 0xFFFF
                im_u = int(dut.fft_im.value) & 0xFFFF
                re_s = re_u - 0x10000 if re_u & 0x8000 else re_u
                im_s = im_u - 0x10000 if im_u & 0x8000 else im_u
                cur_fft_re.append(re_s)
                cur_fft_im.append(im_s)
        except (ValueError, AttributeError):
            pass

        # Logmel frame capture.
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

    if cur_fft_re:
        fft_dumps.append((cur_fft_re, cur_fft_im))
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()
        pre_frames.pop()
        bfpexps.pop()
    return frames, pre_frames, bfpexps, fft_dumps, sync_sample_counts


# ---------------------------------------------------------------------------
# Test 1 — smoke: stfft fires and logmel produces frames
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_frames(dut):
    """
    Feed chirp at CE_EVERY=64 clocks/sample and verify:
      - FFT frame syncs (o_last pulses) are seen
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
        "FFT frames (o_last pulses) : %d  (ideal 50pct-overlap frames for %d samples = %d)"
        % (sync_count, N_SAMPLES, ideal)
    )
    cocotb.log.info(
        "Logmel frames               : %d  (expect %d to %d)"
        % (n, EXPECTED_MIN, EXPECTED_MAX)
    )

    assert sync_count > 0, "No FFT frames emitted -- stfft not running"
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
      4. Saves feature matrix + diagnostics to *.npy
    """
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)

    samples = make_chirp(N_SAMPLES)
    timeout = N_SAMPLES * CE_EVERY + DRAIN

    cocotb.start_soon(drive_samples(dut, samples, ce_every=CE_EVERY))
    frames, pre_frames, bfpexps, fft_dumps, sync_sample_counts = \
        await collect_frames(dut, timeout)

    n = len(frames)
    cocotb.log.info(
        "%d samples (%d clk/sample, HOP=%d) : %d frames  (expect %d to %d)"
        % (N_SAMPLES, CE_EVERY, HOP, n, EXPECTED_MIN, EXPECTED_MAX)
    )

    # 1. Frame count
    assert EXPECTED_MIN <= n <= EXPECTED_MAX, (
        "Frame count %d outside [%d, %d].\n"
        "  ~half expected -> logmel is dropping frames. Increase CE_EVERY (now %d).\n"
        "  Rule: HOP * CE_EVERY > logmel_time (~7500). Min CE_EVERY = %d.\n"
        "  0 frames -> stfft not running or fft_sync_r never fires."
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
    here = os.path.dirname(__file__) or "."

    mat = np.stack(
        [np.array(frame, np.float32) / (1 << Q_FRAC) for frame in frames],
        axis=1
    )
    npy = os.path.join(here, "rtl_features.npy")
    np.save(npy, mat)

    pre_mat = np.stack(
        [np.array(frame, np.float32) / (1 << Q_FRAC) for frame in pre_frames],
        axis=1,
    )
    np.save(os.path.join(here, "rtl_features_precomp.npy"), pre_mat)

    np.save(
        os.path.join(here, "rtl_bfpexps.npy"),
        np.array([b if b is not None else -1 for b in bfpexps], dtype=np.int8),
    )

    # Raw FFT bin dumps — keep frames with exactly N_BINS=129 entries.
    n_bins_expected = 129
    clean_fft = [
        (re, im) for (re, im) in fft_dumps
        if len(re) == n_bins_expected and len(im) == n_bins_expected
    ]
    if clean_fft:
        re_mat = np.array([re for (re, _) in clean_fft], dtype=np.int32)
        im_mat = np.array([im for (_, im) in clean_fft], dtype=np.int32)
        np.save(os.path.join(here, "rtl_fft_re.npy"), re_mat)
        np.save(os.path.join(here, "rtl_fft_im.npy"), im_mat)
        cocotb.log.info(
            "Dumped raw FFT bins: %d frames x %d bins",
            re_mat.shape[0], re_mat.shape[1],
        )

    # Absolute input-sample count at each fft_sync_rr. With 50% overlap
    # the deltas should be constant at HOP=128 (once past the startup
    # warm-up fill of FFT_SIZE samples).
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
    if any(b is not None for b in bfpexps):
        valid_bfp = [b for b in bfpexps if b is not None]
        cocotb.log.info(
            "bfpexps per frame (min=%d max=%d mean=%.1f): %s",
            min(valid_bfp), max(valid_bfp), np.mean(valid_bfp), bfpexps,
        )