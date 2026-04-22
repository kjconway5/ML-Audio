"""GoldenLegacyExtractor — models the pre-FIFO, bug-infested stfft pipeline.

The "canonical" GoldenExtractor in golden_model.py matches the ideal
spec (hop=128, correctly aligned Hann, no sample drops) — and matches
the FIXED RTL (stfft_fixed.sv + pipeline_top_fixed.sv).

This legacy variant models the ORIGINAL stfft.sv + pipeline_top.sv so its
output matches what the pre-fix pipeline produced.  Four bugs modelled:

  Bug A — Sample drops during R2FFT FFT+DMA.
          After each emission, the R2FFT is in ST_RUN_FFT / ST_DONE for
          ~21 sample-times (at CE_EVERY=64), during which sact_istream
          samples are not written to its input RAM.  The next buffer
          therefore starts from sample index 256+21 (not 256), drifting
          the effective A stride to 277 samples (= 256+21), not 256.

  Bug B — Hann coefficient indexed by the GLOBAL sample_cnt.
          stfft.sv uses `a_coeff_idx = sample_cnt` and `b_coeff_idx =
          sample_cnt - HOP` (both mod FFT_SIZE).  After each drop cycle
          the Hann window is cyclically rotated by the drop amount; A
          and B use DIFFERENT indices (offset by HOP).

  Bug D — pipeline_top bin_cnt_q is loaded 3 cycles before bin 0 arrives
          at fft_result_rr, so the logmel's mel_filterbank stores 3
          STALE values into power_buf[0..2] before bin 0 shows up.
          Net effect: a cyclic roll of +3 on the FFT output that feeds
          the mel filterbank.  This is present in BOTH the legacy AND
          the fixed pipelines (they share pipeline_top timing code), but
          the shift is small enough in mel space that the fixed RTL
          still matches the canonical golden to ~0.7 log2 mean error.

  Bug C — bfpexp=0 on the FIRST emitted frame (observed but not modelled
          here).  Likely a reset-timing / bfp_bitWidthAcc init ordering
          issue; manifests as max|Δ| ≈ 29 log2 on frame 0.  Callers
          should discard the startup frame — consistent with the
          STARTUP_LOSS=2 convention the training pipeline already uses.

Calibration against the legacy RTL (7500-sample chirp, 55 emitted frames):
  - 54/54 peak bins match (excluding Bug-C startup frame 0).
  - 51/54 bfpexp match (the 3 mismatches are ±1 BFP edge cases where the
    RTL's per-stage streaming max-bw differs from the golden's global max
    by one bit).
  - Median |Δ| = 0.00 log2 on the post-comp features.
  - Mean |Δ| = 0.34 log2 (dominated by Bug-C startup frame).

Use this class to:
  - Validate legacy RTL pipeline_top against a matching reference.
  - Reproduce legacy RTL features for training corpora built before the
    RTL fix (so inference features on the fixed pipeline stay aligned
    with training features).

For FIXED RTL or ideal features, use GoldenExtractor (golden_model.py).
"""

from pathlib import Path
import numpy as np

# Re-use every constant, helper, and stage from the canonical golden.
from golden_model import (
    GoldenExtractor,
    _r2fft_emulate,
    N_FFT, N_MELS, N_BINS, HOP_LENGTH, WIN_LENGTH,
    SAMPLE_RATE, SAMPLE_W, SAMPLE_MAX, FFT_W, Q_FRAC, LOG_OUT_W,
)

# Default busy time measured empirically from the legacy RTL: A's sync
# stride in samples was observed to be 277 = 256 (buffer) + 21 (busy).
DEFAULT_BUSY_SAMPLES = 21

# The legacy cocotb test drives N_SAMPLES chirp samples + N_FFT=256 zeros
# of trailing padding.  Model the same to match frame count.
DEFAULT_POST_PAD = N_FFT

# Bug D (pipeline_top DMA-to-bin_cnt_q timing offset):
# In pipeline_top.sv, `bin_cnt_q` is loaded on `fft_sync_rr` but
# `fft_result_rr` holds bin 0 only 3-4 cycles LATER (ram read + a_dmadr_real_r
# + a_result + o_fft_result + 2 pipeline_top regs).  So for the first 3
# fft_valid=1 cycles, the logmel's mel_filterbank stores STALE fft_result_rr
# values into power_buf[0..2].  Net effect: the logmel reads FFT bin
# (k - 3) when it thinks it's reading bin k.
#
# This affects BOTH pipeline_top and pipeline_top_fixed (same timing code).
# Model it with a cyclic shift on the FFT output.
DMA_CAPTURE_SHIFT = 3


def _rtl_window_one(sample: int, hann_coef: int, IW: int = 16) -> int:
    """Exactly reproduce stfft.sv's windowing bit-select.

    sample is signed IW-bit, hann_coef is unsigned IW-bit (Q0.15).
    Product is (2*IW)-bit signed; the RTL takes bits [2*IW-2 : IW-1]
    (top IW bits, sign bit dropped), reinterpreted as signed IW-bit.
    """
    product = int(sample) * int(hann_coef)
    shifted = product >> (IW - 1)       # arithmetic right shift
    mask = (1 << IW) - 1
    wrapped = shifted & mask
    sign_bit = 1 << (IW - 1)
    return wrapped - (1 << IW) if wrapped >= sign_bit else wrapped


def _simulate_legacy_stfft(audio: np.ndarray, hann_coeffs: np.ndarray,
                           a_busy: int, b_busy: int):
    """Walk the sample stream sample-by-sample, emulating A and B channels.

    Returns a list of (channel, windowed_buffer_list[256]) tuples in the
    order the legacy RTL would emit them (interleaved A, B, A, B, ...).

    Critically, A and B have DIFFERENT Hann indexing:
        a_coeff_idx = sample_cnt
        b_coeff_idx = sample_cnt - HOP   (mod FFT_SIZE)
    so each sample produces TWO different windowed values, one per channel.
    """
    sample_cnt = 0
    a_buf, b_buf = [], []
    a_busy_left, b_busy_left = 0, 0
    b_armed = False
    emissions = []

    for x in audio:
        # Per-channel Hann indices (different between A and B — see stfft.sv).
        a_idx = sample_cnt
        b_idx = (sample_cnt - HOP_LENGTH) % N_FFT
        a_win = _rtl_window_one(int(x), int(hann_coeffs[a_idx]))
        b_win = _rtl_window_one(int(x), int(hann_coeffs[b_idx]))

        # Channel A: always captures when not busy.  On buffer full, emit
        # and enter busy for a_busy samples.
        if a_busy_left > 0:
            a_busy_left -= 1
        else:
            a_buf.append(a_win)
            if len(a_buf) == N_FFT:
                emissions.append(('A', a_buf))
                a_buf = []
                a_busy_left = a_busy

        # Channel B: armed on the sample AFTER sample_cnt == HOP.  The
        # sample at index HOP does NOT go into B (b_armed latches on that
        # cycle but b_s1_ce uses the PRE-edge value of b_armed = 0).
        if b_armed:
            if b_busy_left > 0:
                b_busy_left -= 1
            else:
                b_buf.append(b_win)
                if len(b_buf) == N_FFT:
                    emissions.append(('B', b_buf))
                    b_buf = []
                    b_busy_left = b_busy
        elif sample_cnt == HOP_LENGTH:
            b_armed = True

        sample_cnt = (sample_cnt + 1) % N_FFT

    return emissions


class GoldenLegacyExtractor:
    """Fixed-point replica of the LEGACY (pre-FIFO) RTL feature pipeline.

    Same stages as GoldenExtractor, but with a bug-faithful STFFT that
    drops samples during R2FFT busy and indexes Hann by a global counter.
    """

    def __init__(self,
                 bfp_compensate: bool = True,
                 a_busy_samples: int = DEFAULT_BUSY_SAMPLES,
                 b_busy_samples: int = DEFAULT_BUSY_SAMPLES,
                 post_pad: int = DEFAULT_POST_PAD):
        self.bfp_compensate = bfp_compensate
        self.a_busy = a_busy_samples
        self.b_busy = b_busy_samples
        self.post_pad = post_pad

        # The post-STFFT stages (power, mel, log, BFP comp) are unchanged
        # from the canonical golden.  Reuse its initialisation.
        self._canonical = GoldenExtractor(bfp_compensate=bfp_compensate)

    def extract_with_fft(self, audio: np.ndarray):
        """Return (features, fft_re, fft_im, bfpexps) — features is Q_FRAC
        uint16, the per-frame FFT bin arrays are int16, bfpexps int8.

        Useful for direct comparison with rtl_features_legacy.npy,
        rtl_fft_re_legacy.npy, etc."""
        samples = np.clip(np.asarray(audio, dtype=np.int32),
                          -SAMPLE_MAX - 1, SAMPLE_MAX)
        # Match the legacy cocotb test: N_FFT zeros of post-padding so the
        # RTL can finish its last in-flight frame.
        if self.post_pad > 0:
            samples = np.concatenate([samples, np.zeros(self.post_pad, dtype=np.int32)])

        emissions = _simulate_legacy_stfft(
            samples, self._canonical.win_coeffs, self.a_busy, self.b_busy,
        )

        n = len(emissions)
        feats = np.zeros((N_MELS, n), dtype=np.uint16)
        fft_re = np.zeros((n, N_BINS), dtype=np.int32)
        fft_im = np.zeros((n, N_BINS), dtype=np.int32)
        bfps   = np.zeros(n, dtype=np.int32)

        for f, (_ch, windowed) in enumerate(emissions):
            win_arr = np.asarray(windowed, dtype=np.int64)
            re_u, im_u, bfp = _r2fft_emulate(win_arr, FFT_W=FFT_W)

            # Unsigned-wrap → signed int16 for the bin dump.
            re_s = re_u.astype(np.int64)
            im_s = im_u.astype(np.int64)
            re_s = np.where(re_s >= 0x8000, re_s - 0x10000, re_s)
            im_s = np.where(im_s >= 0x8000, im_s - 0x10000, im_s)

            # Model Bug D: the cocotb capture and the logmel both see a
            # 3-bin CYCLIC ROLL of the FFT output due to pipeline_top's
            # bin_cnt_q firing 3 cycles before bin 0 is available at
            # fft_result_rr.  Rotate right by DMA_CAPTURE_SHIFT.
            if DMA_CAPTURE_SHIFT:
                k = DMA_CAPTURE_SHIFT
                re_s = np.roll(re_s, k)
                im_s = np.roll(im_s, k)
                re_u_shifted = np.roll(re_u, k)
                im_u_shifted = np.roll(im_u, k)
            else:
                re_u_shifted = re_u
                im_u_shifted = im_u

            fft_re[f, :] = re_s.astype(np.int32)
            fft_im[f, :] = im_s.astype(np.int32)
            bfps[f] = bfp

            # Rest of the pipeline: power → mel → log → BFP compensation.
            # Uses the shifted FFT output (matching what the RTL's logmel
            # actually sees on the shared fft_re/fft_im signals).
            pwr    = self._canonical._power(re_u_shifted, im_u_shifted)
            mel_e  = self._canonical._filterbank(pwr)
            log_pre = self._canonical._log_compress(mel_e)
            if self.bfp_compensate:
                feats[:, f] = self._canonical._bfp_compensate(log_pre, bfp)
            else:
                feats[:, f] = log_pre

        return feats, fft_re, fft_im, bfps

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """(N_MELS, n_frames) uint16 log-mel features matching legacy RTL."""
        feats, _re, _im, _bfp = self.extract_with_fft(audio)
        return feats

    def extract_float(self, audio: np.ndarray) -> np.ndarray:
        return self.extract(audio).astype(np.float32) / (1 << Q_FRAC)


if __name__ == "__main__":
    # Self-test: make sure extract runs without error and produces a sane
    # frame count matching the legacy RTL (55 frames for a 7500-sample chirp).
    dur_samples = 7500
    t = np.arange(dur_samples) / SAMPLE_RATE
    dur = dur_samples / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t ** 2)
    chirp = (np.sin(phase) * SAMPLE_MAX).astype(np.int32)

    ext = GoldenLegacyExtractor()
    feats = ext.extract(chirp)
    print(f"GoldenLegacyExtractor produced: {feats.shape} features")
    print(f"  (legacy RTL emitted 55 frames for the same chirp)")
    print(f"  max={feats.max()}, min={feats.min()} (expect large range)")
