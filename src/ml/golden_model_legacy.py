import numpy as np
import torch
import os
from pathlib import Path

# RTL parameters (must match logmel_top defaults)
SAMPLE_RATE = 16000
N_FFT       = 256
N_BINS      = N_FFT // 2 + 1   # 129
N_MELS      = 40
WIN_LENGTH  = 256
HOP_LENGTH  = N_FFT // 2       # 128

SAMPLE_W    = 16
FFT_W       = 16
SHIFT       = 6
POWER_W     = 2 * FFT_W - SHIFT + 1  # 27
WEIGHT_W    = 16                # Q0.15 mel coefficients
FRAC_BITS   = 15
ACCUM_W     = 54
MAX_COEFFS  = 16
LOG_OUT_W   = 16
LUT_FRAC    = 6
Q_FRAC      = 10

POWER_MASK = (1 << POWER_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1
LOG_MASK   = (1 << LOG_OUT_W) - 1
SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1   # 8191

WIN_COEFF_SCALE = 2047

# windowfn.v rounding parameters
_IW = SAMPLE_W
_TW = SAMPLE_W
_OW = SAMPLE_W
_AW = _IW + _TW  # 28

# Hex data paths
_HERE       = Path(__file__).resolve().parent
_DATA_DIR   = _HERE.parent / "rtl" / "Log-Mel" / "data"
_WIN_HEX    = _HERE.parent / "rtl" / "STFFT" / "ZipCPU" / "hanning.hex"


def _load_hex(path: Path) -> list[int]:
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# R2FFT emulation — bit-accurate replica of the RTL's per-stage BFP + 16-bit
# radix-2 butterfly.  Matches:
#   src/rtl/STFFT/R2FFT/hdl/radix2Butterfly.sv  (butterfly arithmetic)
#   src/rtl/STFFT/R2FFT/hdl/bfp_Shifter.sv       (operand normalization)
#   src/rtl/STFFT/R2FFT/hdl/bfp_bitWidthAcc.sv  (bfpexp accumulation)
#   src/rtl/STFFT/R2FFT/hdl/bfp_bitWidthDetector.sv  (stage bit-width detect)
# The overall control / addressing logic of R2FFT.sv is abstracted away — we
# just run the standard textbook DIF radix-2 FFT with the same underlying math.
# ═══════════════════════════════════════════════════════════════════════════════


def _bit_reverse_perm(x: np.ndarray, log2N: int) -> np.ndarray:
    """Return a copy of x with entries permuted by bit-reversal of index."""
    N = x.shape[0]
    idx = np.arange(N)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(log2N):
        rev |= ((idx >> b) & 1) << (log2N - 1 - b)
    return x[rev]


def _bit_width(v: int, DW: int = 16) -> int:
    """Compute the RTL's bfp_bitWidthDetector output for a single value.

    The RTL takes |value| (two's complement negate for negatives), then finds
    the highest set bit position + 1.  Range is [0, FFT_DW].
    """
    v = abs(int(v))
    if v == 0:
        return 0
    bw = v.bit_length()
    return min(bw, DW)


def _bfp_scale(bw: int, DW: int = 16) -> int:
    """RTL bfp_bitWidthAcc formula for per-stage exponent increment."""
    if bw == DW:
        return 1
    if bw == 0:
        return 0
    return bw - (DW - 2)


def _bfp_shifter(v: int, bw: int, DW: int = 16) -> int:
    """RTL bfp_Shifter: left-shift operand by (DW-1-bw) bits to normalize."""
    if bw == DW or bw == DW - 1 or bw == 0:
        return int(v)
    shift = (DW - 1) - bw
    return int(v) << shift


def _rtl_butterfly(a_re: int, a_im: int, b_re: int, b_im: int,
                   w_re: int, w_im: int, DW: int = 16):
    """Exact radix2Butterfly.sv arithmetic.

    Args:
      a_re, a_im, b_re, b_im: signed ints in [-2^(DW-1), 2^(DW-1)-1]
      w_re, w_im: twiddle components, signed ints in [-2^(DW-1), 2^(DW-1)]
                  (Q(DW-1), one extra bit so -1.0 is representable)
    Returns:
      (dst_a_re, dst_a_im, dst_b_re, dst_b_im), each signed int in DW bits.
    """
    # AddAvg / SubAvg: truncated (a op b) / 2.  Python >> floors for negatives,
    # which matches Verilog's {opa[DW-1], opa} +/- {opb[DW-1], opb}, then
    # taking bits [DW:1] (dropping bit 0).
    a_re, a_im = int(a_re), int(a_im)
    b_re, b_im = int(b_re), int(b_im)
    w_re, w_im = int(w_re), int(w_im)

    dst_a_re = (a_re + b_re) >> 1
    dst_a_im = (a_im + b_im) >> 1

    xbuf_re = (a_re - b_re) >> 1
    xbuf_im = (a_im - b_im) >> 1

    # Three-multiplication complex multiply:
    #   dst_b = ((xbuf_re + i*xbuf_im) * (w_re + i*w_im))
    #         = ((x_re + x_im)*w_re - (w_re + w_im)*x_im
    #            + i*((x_re + x_im)*w_re - (w_re - w_im)*x_re))
    xbuf_re_p_im = xbuf_re + xbuf_im
    tw_re_p_im   = w_re + w_im
    tw_re_m_im   = w_re - w_im

    tmp_a = xbuf_re_p_im * w_re
    tmp_r = tw_re_p_im   * xbuf_im
    tmp_i = tw_re_m_im   * xbuf_re

    # Rounding addend: 2^(DW-2) at the position of bit (DW-2) in the product.
    round_add = 1 << (DW - 2)
    yr = (tmp_a - tmp_r) + round_add
    yi = (tmp_a - tmp_i) + round_add

    # Arithmetic shift right by (DW-1) — Python >> on signed ints is floor.
    yr_shifted = yr >> (DW - 1)
    yi_shifted = yi >> (DW - 1)

    # LimitOperand: saturate to DW-bit signed.
    hi = (1 << (DW - 1)) - 1
    lo = -(1 << (DW - 1))
    yr_sat = max(lo, min(hi, yr_shifted))
    yi_sat = max(lo, min(hi, yi_shifted))

    return dst_a_re, dst_a_im, yr_sat, yi_sat


def _r2fft_emulate(x_real: np.ndarray, FFT_W: int = 16):
    """Emulate R2FFT.  Returns (re_u[N/2+1], im_u[N/2+1], bfpexp_signed_int8).

    Algorithm:
      1. Initialize complex buffer: X_re = samples (int), X_im = 0.
         Bit-width detection is done at input (matches the RTL's
         istreamBitWidthDetector running over the input stream).
      2. For each of log2(N) DIF stages:
           a. Apply bfp_Shifter to every sample (left-shift to normalize).
           b. For each butterfly pair at this stage, compute with exact RTL
              math and collect the output bit-width across all outputs.
           c. Accumulate bfpexp using the RTL rule.
      3. Bit-reverse permute the output to natural DFT order (standard DIF).
      4. Take the first N/2+1 bins and return as unsigned-DW representations.
    """
    DW = FFT_W
    N  = x_real.shape[0]
    log2N = int(np.log2(N)); assert 1 << log2N == N

    # Samples arrive as int16-signed; clip/check the DW-bit range.
    hi = (1 << (DW - 1)) - 1
    lo = -(1 << (DW - 1))
    X_re = [int(max(lo, min(hi, int(v)))) for v in x_real]
    X_im = [0] * N

    # Initial bit-width across the input stream.
    max_abs_in = max((abs(v) for v in X_re + X_im), default=0)
    bw = _bit_width(max_abs_in, DW)
    bfpexp = _bfp_scale(bw, DW)

    # Precompute twiddles for size-N DFT (shared across stages; indexing differs per stage).
    # Twiddle values in Q(DW-1) signed format — one extra bit for -1.0.
    tw_lim_hi = 1 << (DW - 1)        # =  2^(DW-1), matches +1.0 saturation
    tw_lim_lo = -tw_lim_hi           # = -2^(DW-1), matches -1.0
    tw_re = [0] * N
    tw_im = [0] * N
    for k in range(N):
        ang = -2.0 * np.pi * k / N
        tr = int(round(np.cos(ang) * (1 << (DW - 1))))
        ti = int(round(np.sin(ang) * (1 << (DW - 1))))
        tw_re[k] = max(tw_lim_lo, min(tw_lim_hi, tr))
        tw_im[k] = max(tw_lim_lo, min(tw_lim_hi, ti))

    # Run log2(N) DIF stages.  Group size halves each stage.
    group_size = N
    for s in range(log2N):
        half = group_size >> 1

        # BFP-normalize every sample based on the bit-width of the *previous*
        # stage's output (or of the input, for the first stage).
        if bw == 0 or bw == DW or bw == DW - 1:
            Xs_re, Xs_im = X_re, X_im
        else:
            shift = (DW - 1) - bw
            Xs_re = [v << shift for v in X_re]
            Xs_im = [v << shift for v in X_im]

        new_re = [0] * N
        new_im = [0] * N
        max_out = 0

        # Twiddle stride: for DIF stage s, the k-th butterfly of each group
        # uses twiddle W^(k * 2^s) in a size-N DFT.
        tw_stride = 1 << s

        for g in range(0, N, group_size):
            for k in range(half):
                tw_idx = k * tw_stride
                wr = tw_re[tw_idx]
                wi = tw_im[tw_idx]

                ai = g + k
                bi = g + k + half

                ya_re, ya_im, yb_re, yb_im = _rtl_butterfly(
                    Xs_re[ai], Xs_im[ai], Xs_re[bi], Xs_im[bi], wr, wi, DW
                )

                new_re[ai] = ya_re
                new_im[ai] = ya_im
                new_re[bi] = yb_re
                new_im[bi] = yb_im

                for v in (ya_re, ya_im, yb_re, yb_im):
                    av = -v if v < 0 else v
                    if av > max_out:
                        max_out = av

        X_re, X_im = new_re, new_im
        bw = _bit_width(max_out, DW)
        # Match RTL bfp_bitWidthAcc: the last stage's output-bw scale is NOT
        # accumulated into bfpexp, because `update` is gated by
        # !fftStageCountFull in R2FFT.sv.  The RTL only tracks the cumulative
        # shift applied up to and including the shifter at the start of this
        # stage (which used the previous stage's output bw).
        if s < log2N - 1:
            bfpexp += _bfp_scale(bw, DW)
        group_size = half

    # DIF on natural-order input puts results in bit-reversed order.  Permute.
    Xr = np.asarray(X_re, dtype=np.int64)
    Xi = np.asarray(X_im, dtype=np.int64)
    Xr_nat = _bit_reverse_perm(Xr, log2N)
    Xi_nat = _bit_reverse_perm(Xi, log2N)

    # Take the first N/2+1 bins and wrap to unsigned DW-bit.
    nbins = N // 2 + 1
    mask = (1 << DW) - 1
    re_u = (Xr_nat[:nbins] & mask).astype(np.uint64)
    im_u = (Xi_nat[:nbins] & mask).astype(np.uint64)
    return re_u, im_u, int(bfpexp)


class GoldenExtractor:
    """Fixed-point replica of the RTL feature pipeline.

    Models the RTL stages: window (Q0.15 Hann) → R2FFT with per-frame BFP
    (approximated; R2FFT actually uses per-stage BFP) → power (>>SHIFT) →
    sparse mel filterbank (Q0.15 weights, 54-bit accum) → log2 LUT → +2·bfpexp
    correction (pipeline_top's `mel_compensated` path).

    Accuracy: post-FFT stages are bit-identical to RTL.  The FFT stage is a
    float-domain approximation of R2FFT BFP, so bin values can differ by ~1-4
    LSB and the resulting bfpexp may be off by ±1 relative to R2FFT's
    per-stage decisions.  Set `bfp_compensate=False` to get the raw
    pre-compensation log output (what pipeline_top's `u_logmel.cnn_data_ol`
    produces before the `+2·bfpexp` correction — useful for debugging
    pre-compensation stages).
    """

    def __init__(self, bfp_compensate: bool = True):
        # If False, extract() returns raw pre-compensation log values (what
        # the RTL's u_logmel.cnn_data_ol probe used to expose).  If True, the
        # bfpexp correction from pipeline_top.sv:213–251 is applied on top,
        # matching the new `mel_compensated_o` output.
        self.bfp_compensate = bfp_compensate

        self.win_coeffs = np.array(_load_hex(_WIN_HEX), dtype=np.int32)
        assert len(self.win_coeffs) == N_FFT

        # Mel filterbank: sparse ROM → dense matrix
        raw_coeffs = _load_hex(_DATA_DIR / "mel_coeffs.hex")
        self.mel_starts = _load_hex(_DATA_DIR / "mel_starts.hex")
        self.mel_ends   = _load_hex(_DATA_DIR / "mel_ends.hex")
        self.mel_coeffs = np.array(raw_coeffs, dtype=np.int64).reshape(N_MELS, MAX_COEFFS)

        self.fb_dense = np.zeros((N_BINS, N_MELS), dtype=np.int64)
        for m in range(N_MELS):
            s = self.mel_starts[m]
            e = self.mel_ends[m]
            self.fb_dense[s:e+1, m] = self.mel_coeffs[m, :e - s + 1]

        # Log2 LUT (Q4.12)
        self.log_lut = _load_hex(_DATA_DIR / "log2_lut.hex")
        assert len(self.log_lut) == (1 << LUT_FRAC)

    def _window_frame(self, frame: np.ndarray) -> np.ndarray:
        """RTL stfft windowing: `a_s2_prod[2*IW-2 : IW-1]`.

        In Verilog this means "take the top IW bits of the product, skipping
        the MSB sign bit and the bottom (IW-1) bits".  Equivalent to a signed
        right-shift by (IW-1) bits with truncation (no rounding), then
        reinterpret the bottom IW bits as signed.  No saturation.

        IMPORTANT: this is shift-by-15, not shift-by-16.  Keeping the full
        `shift_amt = IW` (=16) as before halves every windowed sample
        compared to the RTL and propagates a +1-bit bfpexp drift through the
        FFT, which shifts the BFP noise floor by ~2 log2 units and shows up
        as a visible delta against rtl_features_precomp.npy.
        """
        data = frame.astype(np.int64)
        tap  = self.win_coeffs.astype(np.int64)
        product = data * tap

        # Match stfft.sv:128 — a_win_samp_w = a_s2_prod[2*IW-2 : IW-1].
        # That is: arithmetic right shift by (IW-1) = 15 bits, truncating
        # the bottom bits (no rounding), then take the low IW bits.
        shift_amt = _IW - 1   # 15 for IW=16
        shifted = product >> shift_amt

        # Reinterpret the low IW bits as signed.  Top bit (`a_s2_prod[2*IW-1]`,
        # the true sign) is dropped by the bit selection, producing wrap-around
        # for full-magnitude negative products — this is an RTL quirk that
        # only matters at the exact ±full-scale boundary.
        mask = (1 << _OW) - 1
        wrapped = shifted & mask
        sign_bit = 1 << (_OW - 1)
        return np.where(wrapped >= sign_bit, wrapped - (1 << _OW), wrapped).astype(np.int32)

    def _fft_frame(self, windowed: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """RTL-accurate R2FFT emulation with per-stage BFP + 16-bit butterfly.

        Computes a 256-point radix-2 DIF FFT of the real windowed input using
        the exact arithmetic of `radix2Butterfly.sv` (AddAvg, SubAvg, 3-mul
        complex multiply with rounding and saturation) and the per-stage BFP
        bookkeeping of `bfp_bitWidthAcc.sv` / `bfp_bitWidthDetector.sv` /
        `bfp_Shifter.sv`.  Output bins are mapped back to natural DFT order.

        Returns (re_u, im_u, bfpexp) — the unsigned 16-bit representations of
        the first N/2+1 bins plus the accumulated BFP exponent, matching what
        the RTL's DMA bus delivers.
        """
        re_u, im_u, bfpexp = _r2fft_emulate(
            windowed.astype(np.int64), FFT_W=FFT_W
        )
        return re_u, im_u, bfpexp

    def _power(self, re: np.ndarray, im: np.ndarray) -> np.ndarray:
        #power_calc.sv: (re² + im²) >> SHIFT
        half = 1 << (FFT_W - 1)
        r = re.astype(np.int64)
        i = im.astype(np.int64)
        r = np.where(r >= half, r - (1 << FFT_W), r)
        i = np.where(i >= half, i - (1 << FFT_W), i)
        s = (r * r + i * i) >> SHIFT
        return (s & POWER_MASK).astype(np.uint64)

    def _filterbank(self, power: np.ndarray) -> np.ndarray:
        #mel_filterbank.sv: 54-bit accumulator, no saturation.
        p = power.astype(np.int64)
        accum = p @ self.fb_dense
        return (accum & ACCUM_MASK).astype(np.uint64)

    def _log_one(self, energy: int) -> int:
        #log_lut.sv: (floor(log2(e)) << Q_FRAC) + lut[frac]. Returns 0 for energy==0.
        if energy == 0:
            return 0
        lg = int(energy).bit_length() - 1
        max_lg = (1 << (LOG_OUT_W - Q_FRAC)) - 1  # 15
        if lg > max_lg:
            return LOG_MASK  # saturate to 0xFFFF
        mask = (1 << LUT_FRAC) - 1
        if lg >= LUT_FRAC:
            addr = (energy >> (lg - LUT_FRAC)) & mask
        else:
            addr = (energy << (LUT_FRAC - lg)) & mask
        return ((lg << Q_FRAC) + self.log_lut[addr]) & LOG_MASK

    def _log_compress(self, mel_energy: np.ndarray) -> np.ndarray:
        return np.array(
            [self._log_one(int(mel_energy[k])) for k in range(N_MELS)],
            dtype=np.uint16,
        )

    def _bfp_compensate(self, log_vals: np.ndarray, bfpexp: int) -> np.ndarray:
        """Apply the bfpexp correction that pipeline_top.sv:213–251 adds.

          correction = 2 * bfpexp * 2^Q_FRAC
          mel_compensated = saturate_unsigned(log_val + correction, 0, 2^LOG_OUT_W-1)

        The R2FFT right-shifts FFT bins by bfpexp, so logmel computes
        `log2(|stored|²) = log2(|true|²) - 2*bfpexp`.  Adding 2*bfpexp recovers
        `log2(|true|²)` in log2 units, scaled by 2^Q_FRAC for the fixed-point
        representation.
        """
        correction = 2 * int(bfpexp) * (1 << Q_FRAC)  # same as BFP_Q_FRAC in RTL
        widened = log_vals.astype(np.int64) + correction
        hi = (1 << LOG_OUT_W) - 1
        return np.clip(widened, 0, hi).astype(np.uint16)

    def _process_frame(self, frame_samples: np.ndarray) -> np.ndarray:
        #Run one frame. Returns bfpexp-compensated Q_FRAC uint16 if
        #self.bfp_compensate is True, else the raw pre-comp log output.
        windowed = self._window_frame(frame_samples)
        re, im, bfpexp = self._fft_frame(windowed)
        power = self._power(re, im)
        mel_e = self._filterbank(power)
        log_pre = self._log_compress(mel_e)
        if not self.bfp_compensate:
            return log_pre
        return self._bfp_compensate(log_pre, bfpexp)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        #Extract log-mel features (bit-accurate to RTL).
       #Input: PCM audio (clipped to 14-bit). Output: (N_MELS, n_frames) uint16 Q4.12.
        samples = np.clip(np.asarray(audio, dtype=np.int32), -SAMPLE_MAX - 1, SAMPLE_MAX)
        n_samples = len(samples)
        n_frames  = max(0, (n_samples - N_FFT) // HOP_LENGTH + 1)

        if n_frames == 0:
            return np.zeros((N_MELS, 0), dtype=np.uint16)

        out = np.zeros((N_MELS, n_frames), dtype=np.uint16)
        for f in range(n_frames):
            start = f * HOP_LENGTH
            frame = samples[start : start + N_FFT]
            out[:, f] = self._process_frame(frame)
        return out

    def extract_float(self, audio: np.ndarray) -> np.ndarray:
        #Same as extract() but returns float32 in log₂ units.
        return self.extract(audio).astype(np.float32) / (1 << Q_FRAC)

    def get_config(self) -> dict:
        return {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "window_length": WIN_LENGTH,
            "sample_w": SAMPLE_W,
            "fft_w": FFT_W,
            "shift": SHIFT,
            "weight_w": WEIGHT_W,
            "accum_w": ACCUM_W,
            "log_out_w": LOG_OUT_W,
            "q_frac": Q_FRAC,
        }


if __name__ == "__main__":
    import time

    ext = GoldenExtractor()
    print("GoldenExtractor config:", ext.get_config())

    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    chirp = np.sin(2 * np.pi * (200 * t + (7000 - 200) / 2 * t ** 2))
    audio_i16 = (chirp * SAMPLE_MAX).astype(np.int16)

    t0 = time.perf_counter()
    feats = ext.extract(audio_i16)
    dt = time.perf_counter() - t0

    print(f"Input: {len(audio_i16)} samples ({len(audio_i16)/SAMPLE_RATE:.2f}s)")
    print(f"Output: {feats.shape} (n_mels={feats.shape[0]}, n_frames={feats.shape[1]})")
    print(f"Time: {dt*1000:.1f} ms, dtype: {feats.dtype}")
    print(f"Range: [{feats.min()}, {feats.max()}]  "
          f"(Q4.12: [{feats.min()/4096:.3f}, {feats.max()/4096:.3f}] log₂)")

    feats_f = ext.extract_float(audio_i16)
    print(f"Float range: [{feats_f.min():.4f}, {feats_f.max():.4f}] log₂")
    print(f"Non-zero bins: {(feats_f > 0).sum()} / {feats_f.size}")
